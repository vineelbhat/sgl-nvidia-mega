# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Kimi K3 packed-MXFP4 megakernel spec for GB300/SM103."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch
from sglang.srt.megakernels.spec import (
    BuildContext,
    BuiltMegakernel,
    Feature,
    MegakernelSpec,
    Numerics,
    NumericsKind,
    Requirements,
    Step,
)
from sglang.srt.runtime_context import get_model

if TYPE_CHECKING:
    from sglang.srt.megakernels.binding import PreparationView

WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64)
# Keep this independent of the host constant so find_spec rejects incompatibility.
REQUIRED_MEGAKERNEL_API_VERSION = 1

# Names read from SGLang internals. A contract test checks these before GPU use.
SGLANG_READS = {
    "srt/megakernels/spec.py": (
        "BuildContext",
        "BuiltMegakernel",
        "Feature",
        "MULTI_NODE_TP",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "Step",
        "TOLERANCE",
        "binding",
        "deadline_s",
        "mode",
        "prepare",
        "rows",
        "tokens_per_request",
        "view",
        "widths",
    ),
    "srt/megakernels/binding.py": ("PreparationView",),
    "srt/runtime_context.py": ("get_model",),
    "srt/arg_groups/fields/*.py": (
        "context_length",
        "cuda_graph_config",
        "decode_attention_backend",
        "dp_size",
        "enable_page_major_kv_layout",
        "enable_unified_memory",
        "flashinfer_mxfp4_moe_precision",
        "kv_cache_dtype",
        "moe_runner_backend",
        "page_size",
        "prefill_attention_backend",
    ),
    "srt/models/kimi_k3.py": (
        "KimiK3ForConditionalGeneration",
        "language_model",
        "logits_processor",
        "lm_head",
    ),
}


def _deadline_ns(seconds: Any) -> int:
    """Validate the build deadline once and freeze it into compiled-kernel keys."""
    if isinstance(seconds, bool):
        raise ValueError(
            f"Kimi K3 megakernel needs a finite positive deadline: {seconds!r}"
        )
    try:
        value = float(seconds)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Kimi K3 megakernel needs a finite positive deadline: {seconds!r}"
        ) from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"Kimi K3 megakernel needs a finite positive deadline: {seconds!r}"
        )
    if value > ((1 << 63) - 1) / 1_000_000_000:
        raise ValueError(f"Kimi K3 megakernel deadline is out of range: {seconds!r}")
    nanoseconds = int(value * 1_000_000_000)
    if nanoseconds <= 0 or nanoseconds > (1 << 63) - 1:
        raise ValueError(f"Kimi K3 megakernel deadline is out of range: {seconds!r}")
    return nanoseconds


def _config_disable_reasons(cfg: Any, hf_config: Any) -> list[str]:
    reasons: list[str] = []
    graph_config = cfg.cuda_graph_config
    if graph_config is None or graph_config.decode.backend != "full":
        reasons.append("needs the FULL decode CUDA-graph backend")
    algorithm = cfg.speculative_algorithm
    if algorithm is not None:
        if str(algorithm).upper() != "DFLASH":
            reasons.append(f"speculative algorithm {algorithm} is not DFLASH")
        if not cfg.enable_linear_replayssm_spec:
            reasons.append("speculative verify needs --enable-linear-replayssm-spec")
        if (cfg.speculative_eagle_topk or 1) != 1:
            reasons.append("speculative verify needs a linear draft chain (topk 1)")
        if not 2 <= int(cfg.speculative_num_draft_tokens or 0) <= 8:
            reasons.append("speculative verify needs 2-8 draft tokens per request")
    return reasons


def _model_disable_reasons(model: Any, hf_config: Any) -> list[str]:
    reasons: list[str] = []
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        return ["the Kimi K3 language model is absent"]
    layers = tuple(getattr(getattr(language_model, "model", None), "layers", ()))
    if len(layers) != 93:
        reasons.append(f"needs the full 93-layer model, found {len(layers)} layers")
    text_config = language_model.config
    full_ids = tuple(
        int(value) for value in getattr(text_config, "full_attention_layer_ids", ())
    )
    linear_ids = tuple(
        int(value) for value in getattr(text_config, "linear_layer_ids", ())
    )
    if (len(full_ids), len(linear_ids)) != (24, 69):
        reasons.append(
            "needs the 24-MLA/69-KDA Kimi K3 attention split, found "
            f"{len(full_ids)} MLA and {len(linear_ids)} KDA layers"
        )
    methods = []
    for layer in layers:
        if not bool(getattr(layer, "_is_moe_layer", False)):
            continue
        method = getattr(getattr(layer.mlp, "experts", None), "quant_method", None)
        methods.append(type(method).__name__.lower())
    if not methods or not all("mxfp4" in name for name in methods):
        reasons.append("needs packed-MXFP4 routed experts on every MoE layer")
    return reasons


def _call_option(
    args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str, default: Any
) -> Any:
    return kwargs.get(name, args[index] if len(args) > index else default)


class _Built:
    """Top-level model-forward replacement for the exact decode shapes served."""

    draft_sampler = None

    def __init__(self, model: Any, resources: Any, widths: tuple[int, ...]):
        from .executor import KimiK3MegakernelExecutor

        self.model = model
        self.language_model = model.language_model
        self.resources = resources
        self.widths = widths
        self.tokens = int(resources.tokens)
        self.executor = KimiK3MegakernelExecutor(resources)
        # Local validation observers read this without changing the public API.
        model.__dict__["_nvidia_kimi_k3_megakernel"] = self

    def summary(self) -> str:
        if self.tokens > 1:
            requests = [width // self.tokens for width in self.widths]
            return (
                f"TP8 {self.tokens}-token verify of {requests[0]}-{requests[-1]} "
                "requests"
            )
        return "TP8 one-token decode at rows 1-8, 16, 32, and 64"

    def capture_layers(self) -> tuple[int, ...]:
        """The layers whose input mixture DFlash captures, in its column order."""
        layers = self.language_model.model.dspark_layers_to_capture or ()
        return tuple(int(layer) + 1 for layer in layers)

    def _captures(self, forward_batch: Any, rows: int) -> tuple[dict, Any]:
        layers = self.capture_layers()
        if not layers:
            return {}, None
        buffer = getattr(forward_batch, "aux_hidden_states_buffer", None)
        width = 7168 * len(layers)
        if buffer is None:
            buffer = self.resources.scratch["aux"]
        if tuple(buffer.shape[1:]) != (width,) or buffer.dtype != torch.bfloat16:
            raise RuntimeError(
                f"Kimi K3 megakernel aux buffer {tuple(buffer.shape)} {buffer.dtype} "
                f"does not take {len(layers)} captures"
            )
        buffer = buffer[:rows]
        captures = {
            layer: buffer[:, 7168 * k : 7168 * (k + 1)]
            for k, layer in enumerate(layers)
        }
        return captures, buffer

    def plan(
        self, input_ids: Any, forward_batch: Any, *args: Any, **kwargs: Any
    ) -> Step | None:
        get_embedding = bool(_call_option(args, kwargs, 0, "get_embedding", False))
        pp_proxy_tensors = _call_option(args, kwargs, 1, "pp_proxy_tensors", None)
        if get_embedding or pp_proxy_tensors is not None:
            return None
        mode = forward_batch.forward_mode
        rows = int(input_ids.shape[0])
        if rows not in self.widths or forward_batch.contains_mm_inputs():
            return None
        if self.tokens > 1:
            spec_info = forward_batch.spec_info
            if not mode.is_target_verify() or spec_info is None:
                return None
            if int(getattr(spec_info, "draft_token_num", 0)) != self.tokens:
                return None
            if getattr(spec_info, "ragged_verify_layout", None) is not None:
                return None
            if int(forward_batch.batch_size) * self.tokens != rows:
                return None
            return Step(mode="target_verify", rows=rows)
        if not mode.is_decode():
            return None
        if int(forward_batch.batch_size) != rows:
            return None
        return Step(mode="decode", rows=rows)

    def forward(
        self,
        step: Step,
        original: Any,
        input_ids: Any,
        positions: Any,
        forward_batch: Any,
        get_embedding: bool = False,
        pp_proxy_tensors: Any = None,
    ) -> Any:
        if get_embedding or pp_proxy_tensors is not None:
            raise RuntimeError("Kimi K3 megakernel received an unplanned model call")
        mode = "target_verify" if self.tokens > 1 else "decode"
        if step.mode != mode or step.rows not in self.widths:
            raise RuntimeError(f"unsupported Kimi K3 megakernel step: {step}")
        if int(forward_batch.batch_size) * self.tokens != step.rows:
            raise RuntimeError("Kimi K3 megakernel plan and forward batch disagree")
        if step.view is None or step.view.hybrid is None:
            raise RuntimeError("Kimi K3 megakernel received no typed hybrid step view")
        captures, aux = self._captures(forward_batch, step.rows)
        hidden_states = self.executor.run(
            input_ids, None, step.view.hybrid, tokens=self.tokens, captures=captures
        )
        return self.language_model.logits_processor(
            input_ids,
            hidden_states,
            self.language_model.lm_head,
            forward_batch,
            aux,
        )


def _prepare(built: BuiltMegakernel, view: PreparationView) -> None:
    """Bind typed SGLang-owned pools after allocation, before graph warmup, and
    compile the AttnRes variants of the layers DFlash captures."""
    if view.hybrid is None:
        raise ValueError("Kimi K3 megakernel requires a hybrid MLA/KDA pool")
    built = cast(_Built, built)
    resources = built.resources
    resources.bind(view.hybrid)
    layers = built.capture_layers()
    if any(layer >= 93 for layer in layers):
        raise ValueError("Kimi K3 megakernel cannot capture the final norm's input")
    for layer in layers:
        consumer = resources.before[layer]
        if consumer.capture is None:
            consumer.bind_capture()
    if layers and "aux" not in resources.scratch:
        device = resources.scratch["hidden.0"].device
        width = 7168 * len(layers)
        resources.scratch["aux"] = torch.empty(
            (max(resources.batches), width), dtype=torch.bfloat16, device=device
        )


def _build(model: Any, hf_config: Any, context: BuildContext) -> _Built:
    from .providers import step_batches

    tokens = int(context.tokens_per_request)
    if not 1 <= tokens <= 8:
        raise ValueError(f"Kimi K3 megakernel verifies 2-8 tokens, not {tokens}")
    served = step_batches(tokens) if tokens > 1 else WIDTHS
    widths = tuple(
        width for width in context.widths if width in served and width % tokens == 0
    )
    if not widths:
        raise ValueError(
            f"Kimi K3 megakernel found no supported width in {list(context.widths)}"
        )
    if context.binding is None:
        raise ValueError("Kimi K3 megakernel received no SGLang runtime binding")
    deadline_ns = _deadline_ns(context.deadline_s)
    from .compat import validate_runtime_compatibility

    validate_runtime_compatibility()
    from .providers import _cuda_memory, build_resources

    resources = build_resources(
        model.language_model,
        context_length=int(get_model().context_length),
        memory_before=_cuda_memory(),
        deadline_ns=deadline_ns,
        tokens=tokens,
    )
    return _Built(model, resources, widths)


SPEC = MegakernelSpec(
    name="Kimi K3 packed-MXFP4",
    requires=Requirements(
        isas=frozenset({"sm_103"}),
        tp_sizes=frozenset({8}),
        resident_blocks=148,
        dtypes=frozenset({"bfloat16"}),
        decode_graph_backends=frozenset({"full"}),
        server_args={
            "kv_cache_dtype": frozenset({"fp8_e4m3"}),
            "moe_runner_backend": frozenset({"flashinfer_mxfp4"}),
            "flashinfer_mxfp4_moe_precision": frozenset({"default"}),
            "prefill_attention_backend": frozenset({"trtllm_mla"}),
            # cutedsl_mla keeps trtllm_mla's decode metadata
            "decode_attention_backend": frozenset({"trtllm_mla", "cutedsl_mla"}),
            "page_size": frozenset({64}),
            "dp_size": frozenset({1}),
            "enable_unified_memory": frozenset({False}),
            "enable_page_major_kv_layout": frozenset({False}),
        },
        hf_config={"encoder_only": False},
    ),
    features=frozenset(
        {Feature.MULTI_NODE_TP, Feature.SPECULATIVE_DECODING, Feature.KV_OFFLOAD}
    ),
    numerics=Numerics(
        kind=NumericsKind.TOLERANCE,
        tolerances={"forward.selected_logprobs.max_abs": 0.5},
    ),
    max_rows=max(WIDTHS),
    build=_build,
    prepare=_prepare,
    api_version=REQUIRED_MEGAKERNEL_API_VERSION,
    model_disable_reasons=_model_disable_reasons,
    config_disable_reasons=_config_disable_reasons,
)
