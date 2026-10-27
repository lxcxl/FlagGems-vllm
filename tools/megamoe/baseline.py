# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MetaX vLLM 0.24.0 vendor INT8 fused MoE + full EP baseline.

The input contract is the same frozen INT8 + per-token FP32 scales used by
MegaMoE. FusedMoE creates RoutedExperts and loads weights through the native
loader. The public FusedMoEKernel.apply owns allocation, experts and finalize;
only prepare is replaced to dispatch already-quantized activations.

A BF16 metadata carrier determines native output/workspace dtype and shape.
Its contents are never read: prepare uses the bound INT8 tensor and scales.
No input dequantization, first quantizer, router recomputation, or vendor source
patch is part of the timed path. Shared experts and LoRA are unsupported.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import statistics
from pathlib import Path

import torch
import vllm
from packaging.version import Version
from vllm.config import set_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
    prepare_communication_buffer_for_model,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.model_loader.utils import (
    configure_quant_config,
    get_model_cls,
    process_weights_after_loading,
)
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

from benchmark.test_megamoe import TIMING_PROTOCOL
from tools.megamoe.common import (
    _dp_sizes_context,
    _load_prequantized_input,
    _rank_value,
    benchmark,
    load_qwen3_rank_data,
    load_real_expert_weights,
    parse_args,
    resolve_moe_spec,
    run_correctness,
    validate_inputs,
)

VLLM_REQUIRED_VERSION = "0.24.0"
VENDOR_EXPERTS_MODULE = "vllm_metax.model_executor.layers.fused_moe.experts.triton_moe"


def vendor_evidence():
    import vllm_metax.envs as envs
    from vllm_metax.utils.fused_moe import get_triton_experts_cls

    for package, required in (("vllm-metax", "0.24.0"), ("mcoplib", "0.4.9")):
        found = importlib.metadata.version(package)
        if Version(found).base_version != required:
            raise RuntimeError(f"Require {package} {required}, found {found}")
    cls = get_triton_experts_cls()
    if cls.__module__ != VENDOR_EXPERTS_MODULE:
        raise RuntimeError("The baseline requires MetaX vendor TritonExperts")
    if not envs.USE_PRECOMPILED_KERNEL or envs.USE_VLLM_TRITON_EXPERT:
        raise RuntimeError("Require vendor precompiled mcoplib experts")
    if envs.MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE:
        raise RuntimeError("This baseline freezes the mcoplib W8A8 backend")
    source = inspect.getsource(cls.apply)
    if "torch.int8" not in source:
        raise RuntimeError("Installed vendor experts lack native INT8 support")
    modules = [
        VENDOR_EXPERTS_MODULE,
        "vllm_metax.model_executor.layers.fused_moe.fused_moe",
        "vllm_metax.model_executor.layers.fused_moe.utils",
        "mcoplib.triton_fused_moe",
        "vllm.model_executor.layers.fused_moe.modular_kernel",
        "vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep",
    ]
    sources = {}
    for name in modules:
        path = Path(importlib.import_module(name).__file__)
        sources[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "vllm-metax", "mcoplib", "torch")
        },
        "sources": sources,
        "experts_source_patch": False,
        "public_entrypoint": "FusedMoEKernel.apply",
        "prepare_adapter": "prequantized INT8 + FP32 scales; native EP dispatch",
        "output_dtype_carrier": "uninitialized BF16 metadata; data never read",
        "first_input_quantization_in_timed_region": False,
    }


def build_fused_moe(*, vllm_config, spec, layer, device):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(vllm_config.model_config.dtype)
    try:
        with torch.device(device):
            runner = FusedMoE(
                num_experts=int(spec["num_experts"]),
                top_k=int(spec["top_k"]),
                hidden_size=int(spec["hidden_size"]),
                intermediate_size=int(spec["intermediate_size"]),
                params_dtype=vllm_config.model_config.dtype,
                renormalize=bool(spec["renormalize"]),
                quant_config=vllm_config.quant_config,
                prefix=f"model.language_model.layers.{layer}.mlp.experts",
                scoring_func="softmax",
                activation=str(spec["activation"]),
            )
    finally:
        torch.set_default_dtype(old_dtype)
    moe = runner.routed_experts
    moe.expert_mapping = RoutedExperts.make_expert_params_mapping(
        moe,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=int(spec["num_experts"]),
        num_redundant_experts=0,
        routed_experts_prefix="",
    )
    return moe


def build_direct_kernel(moe, *, use_dp_ep, prepare=None):
    from vllm_metax.utils.fused_moe import get_triton_experts_cls

    cls = get_triton_experts_cls()
    if cls.__module__ != VENDOR_EXPERTS_MODULE:
        raise RuntimeError("Non-vendor expert implementation is forbidden")
    experts = cls(moe.moe_config, moe.moe_quant_config)
    if experts.expects_unquantized_inputs:
        raise NotImplementedError("LoRA / deferred input quantization")
    if prepare is None:
        prepare = (
            MoEPrepareAndFinalizeNaiveDPEPModular(
                is_sequence_parallel=moe.moe_config.is_sequence_parallel
            )
            if use_dp_ep
            else MoEPrepareAndFinalizeNoDPEPModular()
        )
    return FusedMoEKernel(prepare, experts)


class _PrequantizedPrepare:
    """Only the first input quantizer is bypassed; native finalize is inherited."""

    bound_input = None
    use_dp_ep = False

    def prepare(
        self,
        a1,
        topk_weights,
        topk_ids,
        num_experts,
        expert_map,
        apply_router_weight_on_input,
        quant_config,
        defer_input_quant=False,
    ):
        if self.bound_input is None:
            raise RuntimeError("No prequantized input bound")
        if apply_router_weight_on_input or defer_input_quant:
            raise NotImplementedError("Input router weighting / deferred quantization")
        if (
            quant_config.quant_dtype != torch.int8
            or not quant_config.per_act_token_quant
            or quant_config.block_shape is not None
        ):
            raise NotImplementedError("Requires dynamic per-token INT8 W8A8")
        hidden_q, hidden_scale = self.bound_input
        if a1.dtype != torch.bfloat16 or a1.shape != hidden_q.shape:
            raise ValueError("Invalid BF16 output metadata carrier")
        if self.use_dp_ep:
            hidden_q, topk_weights, topk_ids, scales = get_ep_group().dispatch(
                hidden_q,
                topk_weights,
                topk_ids,
                is_sequence_parallel=self.is_sequence_parallel,
                extra_tensors=[hidden_scale],
            )
            if len(scales) != 1:
                raise RuntimeError("INT8 scale dispatch returned an invalid result")
            hidden_scale = scales[0]
        return hidden_q, hidden_scale, None, topk_ids, topk_weights


class _PrequantizedDPEP(_PrequantizedPrepare, MoEPrepareAndFinalizeNaiveDPEPModular):
    use_dp_ep = True


class _PrequantizedLocal(_PrequantizedPrepare, MoEPrepareAndFinalizeNoDPEPModular):
    pass


class PrequantizedInt8MoE:
    def __init__(self, moe, local_tokens, output_dtype):
        if output_dtype != torch.bfloat16:
            raise NotImplementedError("Validated output dtype is BF16")
        self.moe = moe
        self.use_dp_ep = moe.moe_config.dp_size > 1 and moe.moe_config.use_ep
        self.prepare = (
            _PrequantizedDPEP(is_sequence_parallel=moe.moe_config.is_sequence_parallel)
            if self.use_dp_ep
            else _PrequantizedLocal()
        )
        self.kernel = build_direct_kernel(
            moe, use_dp_ep=self.use_dp_ep, prepare=self.prepare
        )
        self.metadata_carrier = torch.empty(
            (local_tokens, moe.w2_weight.size(1)),
            dtype=output_dtype,
            device=moe.w13_weight.device,
        )

    @property
    def experts_name(self):
        cls = self.kernel.fused_experts.__class__
        return f"{cls.__module__}.{cls.__qualname__}"

    @torch.inference_mode()
    def __call__(self, hidden_q, hidden_scale, topk_weights, topk_ids):
        if self.prepare.bound_input is not None:
            raise RuntimeError("Concurrent calls on one adapter are unsupported")
        if hidden_q.dtype != torch.int8 or hidden_scale.dtype != torch.float32:
            raise TypeError("Expected INT8 activations and FP32 scales")
        if hidden_scale.shape != (hidden_q.shape[0], 1):
            raise ValueError("Expected one scale per token")
        self.prepare.bound_input = (hidden_q, hidden_scale)
        try:
            return self.kernel.apply(
                hidden_states=self.metadata_carrier,
                w1=self.moe.w13_weight,
                w2=self.moe.w2_weight,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                activation=self.moe.activation,
                global_num_experts=self.moe.global_num_experts,
                expert_map=self.moe.expert_map,
                apply_router_weight_on_input=False,
            )
        finally:
            self.prepare.bound_input = None


@torch.inference_mode()
def audit_adapter(adapter, hidden_q, hidden_scale, topk_weights, topk_ids):
    """Untimed execution audit: two vendor GEMMs and only intermediate quant."""
    from unittest.mock import patch

    expert_module = importlib.import_module(VENDOR_EXPERTS_MODULE)
    fused_module = importlib.import_module(
        "vllm_metax.model_executor.layers.fused_moe.fused_moe"
    )
    native_gemm = fused_module.fused_moe_triton_kernel
    native_quantize = expert_module.moe_kernel_quantize_input
    gemms, quantizers = [], []

    def gemm(*args, **kwargs):
        gemms.append(
            {
                "input_dtype": str(args[1].dtype),
                "input_shape": list(args[1].shape),
                "weight_dtype": str(args[2].dtype),
                "output_dtype": str(args[3].dtype),
                "use_int8_w8a8": bool(kwargs.get("use_int8_w8a8")),
            }
        )
        return native_gemm(*args, **kwargs)

    def quantize(*args, **kwargs):
        quantizers.append(str(args[0].dtype))
        return native_quantize(*args, **kwargs)

    adapter.metadata_carrier.fill_(float("nan"))
    with patch.object(fused_module, "fused_moe_triton_kernel", gemm), patch.object(
        expert_module, "moe_kernel_quantize_input", quantize
    ):
        poisoned = adapter(hidden_q, hidden_scale, topk_weights, topk_ids).clone()
    adapter.metadata_carrier.zero_()
    clean = adapter(hidden_q, hidden_scale, topk_weights, topk_ids)
    if len(gemms) != 2 or any(
        g["input_dtype"] != "torch.int8"
        or g["weight_dtype"] != "torch.int8"
        or g["output_dtype"] != "torch.bfloat16"
        or not g["use_int8_w8a8"]
        for g in gemms
    ):
        raise AssertionError(f"Expected two vendor W8A8 GEMMs, found {gemms}")
    if quantizers != ["torch.bfloat16"]:
        raise AssertionError(f"Unexpected quantization calls: {quantizers}")
    equal = bool(torch.equal(poisoned, clean))
    finite = bool(torch.isfinite(clean).all())
    if not equal or not finite:
        raise AssertionError("Metadata carrier affected output or output is nonfinite")
    return {
        "vendor_gemm_calls": gemms,
        "intermediate_quantizer_input_dtypes": quantizers,
        "metadata_carrier_nan_invariance": equal,
        "output_finite": finite,
    }


def main() -> None:
    args = parse_args()
    tokens_list = (
        [int(x) for x in args.tokens_list.split(",")]
        if args.tokens_list
        else [args.num_tokens]
    )
    args.num_tokens = max(tokens_list)
    reports = []
    if args.correctness_only and args.skip_correctness:
        raise ValueError(
            "--correctness-only and --skip-correctness are mutually exclusive"
        )
    if Version(vllm.__version__).base_version != VLLM_REQUIRED_VERSION:
        raise RuntimeError(
            f"This METAX baseline requires vLLM {VLLM_REQUIRED_VERSION}, found {vllm.__version__}"
        )
    if "MACA_PATH" not in os.environ:
        raise RuntimeError(
            "METAX_SDK is not set; refusing to run the METAX baseline on another platform"
        )
    import vllm_metax.envs as mx_envs

    vendor_flags = {
        "USE_PRECOMPILED_KERNEL": mx_envs.USE_PRECOMPILED_KERNEL,
        "MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE": mx_envs.MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE,
        "MACA_VLLM_ENABLE_MCTLASS_PYTHON_API": mx_envs.MACA_VLLM_ENABLE_MCTLASS_PYTHON_API,
        "USE_VLLM_TRITON_EXPERT": mx_envs.USE_VLLM_TRITON_EXPERT,
    }
    if vendor_flags["USE_VLLM_TRITON_EXPERT"]:
        raise RuntimeError("Upstream expert debug mode is not a vendor baseline")
    if vendor_flags["MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE"]:
        vendor_backend = "MCTLASS W8A8"
    elif vendor_flags["USE_PRECOMPILED_KERNEL"]:
        vendor_backend = "mcoplib.triton_fused_moe"
    else:
        vendor_backend = "vllm_metax Triton JIT"
    evidence = vendor_evidence()
    if (
        not args.model.is_dir()
        or not (args.shared_data_dir / "manifest.json").is_file()
    ):
        raise FileNotFoundError(
            "model directory or INT8 shared-data manifest is missing"
        )

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and args.all2all_backend != "allgather_reducescatter":
        raise RuntimeError(
            "pre-quantized INT8 scales are validated only with allgather_reducescatter"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("METAX CUDA-compatible runtime is unavailable")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    init_workspace_manager(device, num_ubatches=1)

    engine_args = EngineArgs(
        model=str(args.model),
        skip_tokenizer_init=True,
        load_format="safetensors",
        tensor_parallel_size=1,
        data_parallel_size=world_size,
        enable_expert_parallel=world_size > 1,
        enable_eplb=False,
        expert_placement_strategy="linear",
        all2all_backend=args.all2all_backend,
        distributed_executor_backend="external_launcher",
        enforce_eager=False,
        moe_backend="triton",
    )
    vllm_config = engine_args.create_engine_config()
    if vllm_config.quant_config is None:
        raise RuntimeError("INT8 model has no quantization_config")
    spec = resolve_moe_spec(vllm_config.model_config.hf_text_config)
    configure_quant_config(
        vllm_config.quant_config, get_model_cls(vllm_config.model_config)
    )
    initialized = False
    report = None
    try:
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size, rank, "env://", local_rank, args.dist_backend
            )
            initialized = True
            initialize_model_parallel(1, 1, 1, 1, backend=args.dist_backend)
            dataset = load_qwen3_rank_data(
                args.shared_data_dir,
                rank=rank,
                num_ranks=world_size,
                num_tokens=args.num_tokens,
                hidden=int(spec["hidden_size"]),
                intermediate=int(spec["intermediate_size"]),
                num_experts=int(spec["num_experts"]),
                topk=int(spec["top_k"]),
                torch=torch,
            )
            manifest, shared, local = (
                dataset["manifest"],
                dataset["shared"],
                dataset["local"],
            )
            moe = build_fused_moe(
                vllm_config=vllm_config, spec=spec, layer=args.layer, device=device
            )
            moe.eval()
            local_ids = load_real_expert_weights(moe=moe, local=local, spec=spec)
            process_weights_after_loading(moe, vllm_config.model_config, device)
            prepare_communication_buffer_for_model(moe)
            moe.moe_quant_config = moe.quant_method.get_fused_moe_quant_config(moe)
            qcfg = moe.moe_quant_config
            if qcfg is None or not qcfg.use_int8_w8a8 or qcfg.quant_dtype != torch.int8:
                raise RuntimeError("model must resolve to dynamic INT8 W8A8")
            if not qcfg.per_act_token_quant or qcfg.block_shape is not None:
                raise RuntimeError(
                    "METAX baseline requires dynamic per-token, channelwise INT8"
                )
            full_shared = shared
            for current_tokens in tokens_list:
                args.num_tokens = current_tokens
                shared = {
                    key: tensor[:current_tokens] for key, tensor in full_shared.items()
                }
                for repeat in range(args.repeats):
                    hidden_bf16, hidden_q, hidden_scale, input_source = (
                        _load_prequantized_input(
                            shared=shared, rank=rank, device=device
                        )
                    )
                    topk_ids = (
                        _rank_value(shared["topk_idx"], rank).to(device).contiguous()
                    )
                    topk_weights = (
                        _rank_value(shared["topk_weights"], rank)
                        .to(device)
                        .contiguous()
                    )
                    validate_inputs(
                        hidden_q, hidden_scale, topk_weights, topk_ids, spec
                    )
                    adapter = PrequantizedInt8MoE(
                        moe, args.num_tokens, vllm_config.model_config.dtype
                    )
                    correctness = None
                    if not args.skip_correctness:
                        reference_kernel = build_direct_kernel(
                            moe, use_dp_ep=adapter.use_dp_ep
                        )
                        correctness = run_correctness(
                            moe=moe,
                            adapter=adapter,
                            reference_kernel=reference_kernel,
                            vllm_config=vllm_config,
                            hidden_bf16=hidden_bf16,
                            hidden_q=hidden_q,
                            hidden_scale=hidden_scale,
                            topk_weights=topk_weights,
                            topk_ids=topk_ids,
                            diff_tol=args.diff_tol,
                            device=device,
                        )
                    if correctness is not None:
                        import math

                        if not math.isfinite(correctness["relative_l2_error"]):
                            raise AssertionError("Nonfinite reference error")
                        if repeat == 0:
                            with set_forward_context(
                                attn_metadata=None,
                                vllm_config=vllm_config,
                                num_tokens=current_tokens,
                            ):
                                with _dp_sizes_context(moe):
                                    correctness["execution_audit"] = audit_adapter(
                                        adapter,
                                        hidden_q,
                                        hidden_scale,
                                        topk_weights,
                                        topk_ids,
                                    )
                    performance = None
                    if not args.correctness_only:
                        samples, rank_means = benchmark(
                            moe=moe,
                            adapter=adapter,
                            vllm_config=vllm_config,
                            hidden_q=hidden_q,
                            hidden_scale=hidden_scale,
                            topk_weights=topk_weights,
                            topk_ids=topk_ids,
                            warmup=args.warmup,
                            iters=args.iters,
                            device=device,
                        )
                        slowest = max(rank_means)
                        performance = {
                            "warmup": args.warmup,
                            "iterations": args.iters,
                            "timing": "CUDA events; individual sample completion; full native EP",
                            "timing_protocol": TIMING_PROTOCOL,
                            "per_rank_mean_ms": rank_means,
                            "primary_latency_ms": slowest,
                            "mean_of_iteration_rank_max_ms": statistics.mean(samples),
                            "p50_ms": sorted(samples)[len(samples) // 2],
                            "min_ms": min(samples),
                            "max_ms": max(samples),
                            "global_tokens_per_second": world_size
                            * args.num_tokens
                            / (slowest / 1e3),
                        }
                    if rank == 0:
                        report = {
                            "implementation": "MetaX vendor vLLM 0.24.0 pre-quantized INT8 fused MoE + EP",
                            "vendor_evidence": evidence,
                            "vendor_backend": vendor_backend,
                            "vendor_flags": vendor_flags,
                            "vllm_version": vllm.__version__,
                            "torch_version": torch.__version__,
                            "gpu": torch.cuda.get_device_name(device),
                            "model_config": str(args.model),
                            "shared_data": str(args.shared_data_dir),
                            "source_model": manifest.get("source_model"),
                            "source_revision": manifest.get("source_revision"),
                            "source_layer": manifest.get("layer"),
                            "world_size": world_size,
                            "tokens_per_rank": args.num_tokens,
                            "hidden_size": spec["hidden_size"],
                            "intermediate_size": spec["intermediate_size"],
                            "num_experts": spec["num_experts"],
                            "local_experts": len(local_ids),
                            "top_k": spec["top_k"],
                            "input_dtype": str(hidden_q.dtype),
                            "input_scale_dtype": str(hidden_scale.dtype),
                            "input_source": input_source,
                            "weight_dtype": str(moe.w13_weight.dtype),
                            "weight_quantization": "channelwise",
                            "output_dtype": str(vllm_config.model_config.dtype),
                            "all2all_backend": args.all2all_backend,
                            "first_input_quantization_in_timed_region": False,
                            "experts_implementation": adapter.experts_name,
                            "correctness_reference": (
                                "same METAX vLLM modular kernel with BF16 "
                                "input quantized before experts"
                            ),
                            "routing": manifest.get("routing", {}).get(
                                "source", "unknown"
                            ),
                            "correctness": correctness,
                            "performance": performance,
                        }
                    if args.capture_dir is not None and repeat == 0:
                        import hashlib

                        args.capture_dir.mkdir(parents=True, exist_ok=True)
                        with set_forward_context(
                            attn_metadata=None,
                            vllm_config=vllm_config,
                            num_tokens=current_tokens,
                        ):
                            with _dp_sizes_context(moe):
                                captured = adapter(
                                    hidden_q, hidden_scale, topk_weights, topk_ids
                                ).cpu()
                        tensors = dict(
                            hidden_q=hidden_q,
                            hidden_scale=hidden_scale,
                            topk_weights=topk_weights,
                            topk_ids=topk_ids,
                        )
                        digests = {
                            key: hashlib.sha256(
                                value.cpu()
                                .contiguous()
                                .view(torch.uint8)
                                .numpy()
                                .tobytes()
                            ).hexdigest()
                            for key, value in tensors.items()
                        }
                        torch.save(
                            dict(output=captured, input_sha256=digests),
                            args.capture_dir / f"t{current_tokens}_rank{rank}.pt",
                        )
                    if rank == 0:
                        report["repeat"] = repeat
                        reports.append(report)
                        print("BASELINE_RECORD " + json.dumps(report), flush=True)
                        if args.json_output:
                            args.json_output.parent.mkdir(parents=True, exist_ok=True)
                            args.json_output.write_text(json.dumps(reports, indent=2))
    finally:
        try:
            if initialized:
                try:
                    destroy_model_parallel()
                finally:
                    destroy_distributed_environment()
        finally:
            reset_workspace_manager()
    if report is not None:
        print("\n" + "=" * 88)
        print("METAX vLLM 0.24.0 | direct INT8-input FusedMoE + EP | final report")
        print("=" * 88)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
