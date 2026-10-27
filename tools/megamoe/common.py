# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared input loading, native reference and frozen eager timing protocol."""

from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Iterator

import torch
import torch.distributed as dist
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

from benchmark.test_megamoe import load_qwen3_canonical

INT8_DTYPE = torch.int8


def _get_tensor(mapping: dict[str, torch.Tensor], *names: str) -> torch.Tensor:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, torch.Tensor):
            return value
    raise KeyError(f"none of the required tensor keys are present: {names}")


def _normalize_channel_scale(scale: torch.Tensor, *, name: str) -> torch.Tensor:
    scale = scale.float() if scale.dtype != torch.float32 else scale
    if scale.ndim == 2:
        scale = scale.unsqueeze(-1)
    if scale.ndim != 3 or scale.shape[-1] != 1:
        raise ValueError(f"{name} must be [E,N] or [E,N,1], got {tuple(scale.shape)}")
    return scale.contiguous()


def _first_present(obj, names: Iterable[str], default=None):
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def resolve_moe_spec(hf_config) -> dict[str, object]:
    values = {
        "hidden_size": _first_present(hf_config, ("hidden_size", "d_model")),
        "intermediate_size": _first_present(
            hf_config,
            ("moe_intermediate_size", "expert_intermediate_size", "intermediate_size"),
        ),
        "num_experts": _first_present(
            hf_config, ("num_experts", "n_routed_experts", "num_local_experts")
        ),
        "top_k": _first_present(
            hf_config,
            (
                "num_experts_per_tok",
                "num_experts_per_token",
                "num_selected_experts",
                "moe_top_k",
            ),
        ),
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError("Could not infer MoE fields: " + ", ".join(missing))
    return {
        **{name: int(value) for name, value in values.items()},
        "renormalize": bool(_first_present(hf_config, ("norm_topk_prob",), True)),
        "activation": str(_first_present(hf_config, ("hidden_act",), "silu")),
    }


def _local_expert_ids(moe: FusedMoE, num_experts: int) -> list[int]:
    expert_map = getattr(moe, "_expert_map", None)
    if expert_map is None:
        return list(range(num_experts))
    return (
        torch.nonzero(expert_map.detach().cpu() >= 0, as_tuple=False).flatten().tolist()
    )


def _iter_real_expert_weights(
    *, local: dict[str, torch.Tensor], local_ids: list[int], intermediate_size: int
) -> Iterator[tuple[str, torch.Tensor]]:
    l1 = _get_tensor(local, "l1_weight_int8", "l1_weight_q")
    l2 = _get_tensor(local, "l2_weight_int8", "l2_weight_q")
    l1_scale = _normalize_channel_scale(
        _get_tensor(local, "l1_weight_scale", "l1_weight_scales"),
        name="l1_weight_scale",
    )
    l2_scale = _normalize_channel_scale(
        _get_tensor(local, "l2_weight_scale", "l2_weight_scales"),
        name="l2_weight_scale",
    )
    if l1.dtype != INT8_DTYPE or l2.dtype != INT8_DTYPE:
        raise TypeError(
            f"expected INT8 expert weights, got l1={l1.dtype}, l2={l2.dtype}"
        )
    if l1.shape[0] != len(local_ids) or l2.shape[0] != len(local_ids):
        raise ValueError(
            f"dataset/vLLM local-expert mismatch: {l1.shape[0]}, {l2.shape[0]}, {len(local_ids)}"
        )
    if l1.ndim != 3 or l1.shape[1] != 2 * intermediate_size:
        raise ValueError(f"expected l1 shape [E,2I,H], got {tuple(l1.shape)}")
    if l2.ndim != 3 or l2.shape[1] != l1.shape[2] or l2.shape[2] != intermediate_size:
        raise ValueError(f"expected l2 shape [E,H,I], got {tuple(l2.shape)}")
    if l1_scale.shape[:2] != l1.shape[:2] or l2_scale.shape[:2] != l2.shape[:2]:
        raise ValueError("INT8 scale and weight shapes do not match")
    for slot, expert_id in enumerate(local_ids):
        stem = str(expert_id)
        yield f"{stem}.gate_proj.weight", l1[slot, :intermediate_size]
        yield f"{stem}.up_proj.weight", l1[slot, intermediate_size:]
        yield f"{stem}.down_proj.weight", l2[slot]
        yield f"{stem}.gate_proj.weight_scale", l1_scale[slot, :intermediate_size]
        yield f"{stem}.up_proj.weight_scale", l1_scale[slot, intermediate_size:]
        yield f"{stem}.down_proj.weight_scale", l2_scale[slot]


def load_real_expert_weights(
    *, moe: FusedMoE, local: dict[str, torch.Tensor], spec: dict[str, object]
) -> list[int]:
    local_ids = _local_expert_ids(moe, int(spec["num_experts"]))
    loaded = list(
        moe.load_weights(
            _iter_real_expert_weights(
                local=local,
                local_ids=local_ids,
                intermediate_size=int(spec["intermediate_size"]),
            )
        )
    )
    required = {"w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"}
    missing = sorted(required - set(loaded))
    if missing:
        raise RuntimeError(f"vLLM did not load required INT8 parameters: {missing}")
    return local_ids


def _rank_value(value, rank: int):
    if isinstance(value, (list, tuple)):
        return value[rank]
    if isinstance(value, torch.Tensor) and value.ndim >= 3:
        return value[rank]
    return value


def _load_prequantized_input(*, shared: dict, rank: int, device):
    hidden_bf16 = (
        _rank_value(shared["hidden_states_bf16"], rank).to(device).contiguous()
    )
    input_key = next(
        (
            k
            for k in ("input_int8", "input_int8_values", "hidden_states_int8")
            if k in shared
        ),
        None,
    )
    scale_key = next(
        (
            k
            for k in ("input_scales", "input_int8_scales", "hidden_states_scale")
            if k in shared
        ),
        None,
    )
    if input_key is None or scale_key is None:
        raise KeyError(
            "METAX baseline requires pre-quantized input_int8 and input scale tensors"
        )
    hidden_q = _rank_value(shared[input_key], rank).to(device).contiguous()
    hidden_scale = (
        _rank_value(shared[scale_key], rank)
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )
    return (
        hidden_bf16,
        hidden_q,
        hidden_scale,
        f"shared-dataset:{input_key}+{scale_key}",
    )


def validate_inputs(hidden_q, hidden_scale, topk_weights, topk_ids, spec):
    tokens = hidden_q.shape[0]
    if hidden_q.dtype != INT8_DTYPE or not hidden_q.is_contiguous():
        raise TypeError(
            f"input must be contiguous torch.int8, got {hidden_q.dtype}, {hidden_q.is_contiguous()}"
        )
    if hidden_scale.dtype != torch.float32 or tuple(hidden_scale.shape) != (tokens, 1):
        raise TypeError(
            f"input scales must be FP32 {(tokens, 1)}, got {hidden_scale.dtype}, {tuple(hidden_scale.shape)}"
        )
    expected_topk = (tokens, int(spec["top_k"]))
    if topk_ids.dtype != torch.int64 or tuple(topk_ids.shape) != expected_topk:
        raise TypeError(f"topk ids must be int64 {expected_topk}")
    if (
        topk_weights.dtype != torch.float32
        or tuple(topk_weights.shape) != expected_topk
    ):
        raise TypeError(f"topk weights must be FP32 {expected_topk}")


def _dp_sizes_context(moe: FusedMoE):
    metadata = get_forward_context().dp_metadata
    if metadata is None:
        return nullcontext()
    return metadata.sp_local_sizes(moe.moe_config.sp_size)


def _all_reduce_scalar(value: float, *, op, device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


@torch.inference_mode()
def run_correctness(
    *,
    moe,
    adapter,
    reference_kernel,
    vllm_config,
    hidden_bf16,
    hidden_q,
    hidden_scale,
    topk_weights,
    topk_ids,
    diff_tol,
    device,
):
    qcfg = moe.moe_quant_config
    stock_q, stock_scale = moe_kernel_quantize_input(
        hidden_bf16,
        qcfg.a1_scale,
        qcfg.quant_dtype,
        qcfg.per_act_token_quant,
        qcfg.block_shape,
    )
    mismatch = int(
        _all_reduce_scalar(
            float((stock_q.contiguous() != hidden_q).sum()),
            op=dist.ReduceOp.SUM,
            device=device,
        )
    )
    scale_error = _all_reduce_scalar(
        float((stock_scale.float() - hidden_scale).abs().max()),
        op=dist.ReduceOp.MAX,
        device=device,
    )
    dist.barrier()
    with set_forward_context(
        attn_metadata=None, vllm_config=vllm_config, num_tokens=hidden_q.shape[0]
    ):
        with _dp_sizes_context(moe):
            reference = reference_kernel.apply(
                hidden_states=hidden_bf16,
                w1=moe.w13_weight,
                w2=moe.w2_weight,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                activation=moe.activation,
                global_num_experts=moe.global_num_experts,
                expert_map=moe.expert_map,
                apply_router_weight_on_input=moe.apply_router_weight_on_input,
            )
            output = adapter(hidden_q, hidden_scale, topk_weights, topk_ids)
    torch.cuda.synchronize(device)
    if output.dtype != hidden_bf16.dtype or output.shape != hidden_bf16.shape:
        raise RuntimeError(
            f"invalid output dtype/shape: {output.dtype} {tuple(output.shape)}"
        )
    delta = output.float() - reference.float()
    diff_sq = _all_reduce_scalar(
        float(delta.square().sum()), op=dist.ReduceOp.SUM, device=device
    )
    ref_sq = _all_reduce_scalar(
        float(reference.float().square().sum()), op=dist.ReduceOp.SUM, device=device
    )
    rel_l2 = math.sqrt(diff_sq / max(ref_sq, 1e-30))
    if rel_l2 > diff_tol:
        raise AssertionError(f"relative L2 error {rel_l2:.6g} exceeds {diff_tol:.6g}")
    return {
        "pass": True,
        "relative_l2_error": rel_l2,
        "max_absolute_error": _all_reduce_scalar(
            float(delta.abs().max()), op=dist.ReduceOp.MAX, device=device
        ),
        "input_int8_value_mismatches": mismatch,
        "input_scale_max_absolute_error": scale_error,
    }


@torch.inference_mode()
def benchmark(
    *,
    moe,
    adapter,
    vllm_config,
    hidden_q,
    hidden_scale,
    topk_weights,
    topk_ids,
    warmup,
    iters,
    device,
):
    if warmup < 0 or iters <= 0:
        raise ValueError("--warmup must be >= 0 and --iters must be > 0")
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    with set_forward_context(
        attn_metadata=None, vllm_config=vllm_config, num_tokens=hidden_q.shape[0]
    ):
        with _dp_sizes_context(moe):
            for _ in range(warmup):
                adapter(hidden_q, hidden_scale, topk_weights, topk_ids)
            torch.cuda.synchronize(device)
            dist.barrier()
            for start, end in zip(starts, ends):
                start.record()
                adapter(hidden_q, hidden_scale, topk_weights, topk_ids)
                end.record()
                end.synchronize()
                dist.barrier()
    local = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(starts, ends)],
        dtype=torch.float64,
        device=device,
    )
    rank_max = local.clone()
    dist.all_reduce(rank_max, op=dist.ReduceOp.MAX)
    means = [
        torch.empty(1, dtype=torch.float64, device=device)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(means, local.mean().reshape(1))
    return rank_max.cpu().tolist(), [float(x.item()) for x in means]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--shared-data-dir", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--num-tokens", type=int, default=512)
    p.add_argument("--tokens-list", default="")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--json-output", type=Path)
    p.add_argument("--capture-dir", type=Path)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--diff-tol", type=float, default=0.005)
    p.add_argument("--correctness-only", action="store_true")
    p.add_argument("--skip-correctness", action="store_true")
    p.add_argument(
        "--all2all-backend",
        default="allgather_reducescatter",
        choices=("allgather_reducescatter", "naive"),
    )
    p.add_argument("--dist-backend", default="nccl")
    return p.parse_args()


def load_qwen3_rank_data(
    data_dir,
    *,
    rank: int,
    num_ranks: int,
    num_tokens: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    topk: int,
    torch,
):
    """Select a deterministic tokens-per-rank prefix and linear EP weight slice."""
    dataset = load_qwen3_canonical(data_dir, torch=torch)
    shape = dataset["manifest"]["shape"]
    requested = {
        "hidden_size": hidden,
        "moe_intermediate_size": intermediate,
        "num_experts": num_experts,
        "num_experts_per_tok": topk,
    }
    mismatches = {
        key: {"dataset": shape.get(key), "requested": value}
        for key, value in requested.items()
        if int(shape.get(key, -1)) != int(value)
    }
    if mismatches:
        raise ValueError(f"canonical shape mismatch: {mismatches}")
    if num_ranks not in (1, 2, 4, 8) or num_experts % num_ranks:
        raise ValueError(f"unsupported EP world size: {num_ranks}")
    if not 0 <= rank < num_ranks:
        raise ValueError(f"rank {rank} is outside [0, {num_ranks})")
    capacity = int(shape["tokens_per_rank_capacity"])
    if not 0 < num_tokens <= capacity:
        raise ValueError(f"num_tokens={num_tokens} is outside [1, {capacity}]")

    shared = {
        key: value[rank, :num_tokens].contiguous()
        for key, value in dataset["shared"].items()
    }
    experts_per_rank = num_experts // num_ranks
    begin = rank * experts_per_rank
    end = begin + experts_per_rank
    local = {
        key: value[begin:end].contiguous() for key, value in dataset["experts"].items()
    }
    return {"manifest": dataset["manifest"], "shared": shared, "local": local}
