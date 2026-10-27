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

"""Benchmark the multi-rank MetaX MegaMoE operator.

This harness owns MPI/mxSHMEM setup, inputs, correctness checks and timing.
Build the host bridge ahead of time and set MEGAMOE_HOST_LIB to its shared library.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import torch
import triton
import triton.language as tl
from triton import knobs

TIMING_PROTOCOL = "pr774-isolated-events-5-mean-v1"


@triton.jit
def _local_peer(storage, table, size: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(storage + idx, 0, mask=idx < size)
    if tl.program_id(0) == 0:
        tl.store(table, storage.to(tl.int64))


@triton.jit
def _init_state(state, epoch_counter):
    tl.store(state + tl.arange(0, 32), 0)
    tl.store(epoch_counter, 0)


@lru_cache(maxsize=1)
def _occupancy_query():
    # PyTorch/FlagTree has already loaded the platform runtime.
    runtime = ctypes.CDLL(None)
    query = runtime.mcModuleOccupancyMaxActiveBlocksPerMultiprocessor
    query.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    return query


@lru_cache(maxsize=128)
def resident_blocks(function, num_warps, shared):
    """Query the compiled kernel's resident CTA count, without launching it."""
    from triton.runtime import driver

    count = ctypes.c_int()
    threads = num_warps * driver.active.get_current_target().warp_size
    status = _occupancy_query()(
        ctypes.byref(count), ctypes.c_void_p(function), threads, shared
    )
    if status or count.value < 1:
        raise RuntimeError(f"MegaMoE occupancy query failed: {status}, {count.value}")
    return count.value


@lru_cache(maxsize=1)
def _device_libraries():
    """The caller supplies SDK-matched, prebuilt device libraries."""
    return (
        ("megamoe_sync", os.environ["MEGAMOE_DEVICE_LIB"]),
        ("mxshmem", os.environ["MXSHMEM_DEVICE_LIB"]),
    )


_shmem_modules = {}


@contextmanager
def _shmem_module(compiled, shmem):
    """Reference-count registrations across overlapping caller workspaces."""
    key = (torch.cuda.current_device(), compiled.module)
    if key not in _shmem_modules:
        for name in ("mxshmemx_cumodule_init", "mxshmemx_cumodule_finalize"):
            function = getattr(shmem, name)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_int
        status = shmem.mxshmemx_cumodule_init(ctypes.c_void_p(compiled.module))
        if status:
            raise RuntimeError(f"mxSHMEM module initialization failed: {status}")
        _shmem_modules[key] = [compiled, 0]
    _shmem_modules[key][1] += 1
    try:
        yield
    finally:
        torch.cuda.synchronize()
        _shmem_modules[key][1] -= 1
        if _shmem_modules[key][1] == 0:
            status = shmem.mxshmemx_cumodule_finalize(ctypes.c_void_p(compiled.module))
            del _shmem_modules[key]
            if status:
                raise RuntimeError(f"mxSHMEM module finalization failed: {status}")


@contextmanager
def create_workspace(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    w1_scale,
    w2_scale,
    *,
    shmem,
    extern_libs=None,
    max_tokens=None,
    rank=0,
    world_size=1,
    allocate_symmetric=None,
    release_symmetric=None,
    quant_scale_lut=None,
    a1_scale=None,
    expert_map=None,
    out=None,
    workers=None,
    ctas_per_sm=2,
    local_output=None,
    block_m=None,
    block_n=128,
    block_k=64,
    stages=2,
):
    """Test-owned allocation and residency planning; nothing runs in op timing.

    ctas_per_sm is a configurable tuning budget, not a device or algorithm limit.
    workers overrides the requested total grid. The compiled occupancy may reduce
    it; all grid CTAs must reside simultaneously because the kernel spin-waits.
    Each EP rank must construct matching symmetric layouts/capacities in order.
    """
    from flaggems_vllm.runtime.backend._metax.mega.megamoe.api import MegaMoEContext
    from flaggems_vllm.runtime.backend._metax.mega.megamoe.kernel import (
        _launch_args,
        _megamoe,
    )

    capacity = max(hidden_states.shape[0], 1) if max_tokens is None else max_tokens
    top_k = topk_ids.shape[1]
    e, _, h = w1.shape
    i = w2.shape[2]
    sms = torch.cuda.get_device_properties(w1.device).multi_processor_count
    density = capacity * top_k / e
    bm = block_m or (16 if density < 16 else 32 if density < 32 else 64)
    if local_output is None:
        local_output = capacity * top_k >= sms * bm
    requested = (
        workers if workers is not None else sms * (ctas_per_sm if local_output else 1)
    )
    if requested < 1:
        raise ValueError("workspace requires a positive CTA budget")
    align = lambda n: triton.cdiv(n, 256) * 256
    xoff = 256
    soff = align(xoff + capacity * h)
    roff = align(soff + capacity * 4)
    woff = align(roff + capacity * top_k * 4)
    layout = (align(woff + capacity * top_k * 4), xoff, soff, roff, woff)
    storage = []

    def empty(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=w1.device)

    if allocate_symmetric is None:
        if world_size != 1:
            raise ValueError("EP setup requires a symmetric allocator")

        def allocate_symmetric(nbytes):
            allocation = empty((nbytes,), torch.uint8)
            table = empty((1,), torch.int64)
            _local_peer[(triton.cdiv(nbytes, 1024),)](allocation, table, nbytes, 1024)
            storage.append(allocation)
            return table

    try:
        inputs = allocate_symmetric(layout[0] * world_size)
        partials = allocate_symmetric(top_k * capacity * h * 2)
        signals = allocate_symmetric(3 * world_size * 4)
        blocks = triton.cdiv(capacity * top_k * world_size, bm) + e
        rows = blocks * bm
        chunks = triton.cdiv(capacity * top_k, 64)
        buffers = (
            empty((32,), torch.int32),
            empty((1,), torch.int32),
            empty((world_size * chunks * e,), torch.int32),
            empty((world_size * chunks * e,), torch.int32),
            empty((e * world_size,), torch.int32),
            empty((e,), torch.int32),
            empty((blocks, 2), torch.int32),
            empty((rows, 3), torch.int32),
            empty((rows, h), torch.int8),
            empty((rows,), torch.float32),
            empty((rows,), torch.float32),
            empty((rows, i), torch.bfloat16),
            empty((rows, i), torch.int8),
            empty((rows,), torch.float32),
        )
        expert_partials = (
            empty((min(blocks, 4 * requested) * bm, h), torch.bfloat16)
            if local_output
            else partials
        )
        _init_state[(1,)](buffers[0], buffers[1])
        ws = MegaMoEContext(
            rank=rank,
            world_size=world_size,
            capacity=capacity,
            top_k=top_k,
            model_shape=(e, h, i),
            layout=layout,
            workers=requested,
            bm=bm,
            bn=block_n,
            bk=block_k,
            stages=stages,
            local_output=local_output,
            inputs=inputs,
            partials=partials,
            signals=signals,
            buffers=buffers,
            expert_partials=expert_partials,
            lut=quant_scale_lut,
            extern_libs=(
                tuple(extern_libs) if extern_libs is not None else _device_libraries()
            ),
        )
        with torch.cuda.device(w1.device):
            while True:
                _, args, options = _launch_args(
                    hidden_states,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    w1_scale,
                    w2_scale,
                    workspace=ws,
                    a1_scale=a1_scale,
                    expert_map=expert_map,
                    out=out,
                )
                compiled = _megamoe.warmup(*args, grid=(ws.workers,), **options)
                compiled._init_handles()
                limit = sms * resident_blocks(
                    compiled.function,
                    compiled.metadata.num_warps,
                    compiled.metadata.shared,
                )
                if ws.workers <= limit:
                    break
                ws = replace(ws, workers=limit)
        with _shmem_module(compiled, shmem):
            yield ws
    finally:
        torch.cuda.synchronize(w1.device)
        if release_symmetric is not None:
            release_symmetric()
        storage.clear()


class KernelTimer:
    """Time exactly one MegaMoE launch; keep stream barriers outside events.

    Warmup/JIT, argument binding and occupancy queries precede the enter hook.
    The unchanged device kernel includes its input publication, route planning,
    compute, partial return and internal rank barriers.
    """

    def __init__(
        self, barrier: Optional[Callable[[torch.cuda.Stream], None]] = None
    ) -> None:
        self.barrier = barrier
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.armed = False
        self.launches = 0

    def __enter__(self) -> KernelTimer:
        knobs.runtime.launch_enter_hook.add(self._enter)
        knobs.runtime.launch_exit_hook.add(self._exit)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.armed = False
        knobs.runtime.launch_exit_hook.remove(self._exit)
        knobs.runtime.launch_enter_hook.remove(self._enter)

    def arm(self) -> None:
        self.launches = 0
        self.armed = True

    def _enter(self, metadata) -> None:
        if not self.armed:
            return
        name = metadata.get()["name"]
        assert name == "_megamoe", f"unexpected kernel inside timed call: {name}"
        stream = torch.cuda.current_stream()
        if self.barrier is not None:
            self.barrier(stream)
        self.start.record(stream)
        self.launches += 1

    def _exit(self, metadata) -> None:
        if not self.armed:
            return
        stream = torch.cuda.current_stream()
        self.end.record(stream)
        if self.barrier is not None:
            self.barrier(stream)

    def finish(self) -> float:
        """Return milliseconds after completing this individual sample."""
        assert self.launches == 1, "each sample must contain one MegaMoE launch"
        self.end.synchronize()
        self.armed = False
        return self.start.elapsed_time(self.end)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_qwen3_canonical(data_dir, *, torch, verify_sha256: bool | None = None):
    """Load shared activations/routes and global expert tensors from a manifest."""
    root = Path(data_dir).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if verify_sha256 is None:
        verify_sha256 = os.environ.get("MEGAMOE_DATA_VERIFY_SHA256", "0") == "1"
    tensors = {}
    for name in ("shared", "experts"):
        entry = manifest["files"][name]
        path = root / entry["name"]
        if verify_sha256 and _sha256(path) != entry["sha256"]:
            raise ValueError(f"SHA256 mismatch for {path}")
        tensors[name] = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
    return {"root": root, "manifest": manifest, **tensors}


def vllm_numerical_reference(
    l1_tok, l1_sf, l1_w, w1_q, w1_sf, w2_q, w2_sf, block_expert_tab, MAX_BLOCKS, BLOCK_M
):
    import torch
    import triton.language as tl

    EPR, _, K = w1_q.shape
    INTER = w2_q.shape[-1]
    POOL_TOKENS = l1_tok.shape[0]
    dev = l1_tok.device
    from vllm.model_executor.layers.fused_moe import fused_moe as vf

    ref_weights_l1 = (
        w1_q.reshape(EPR, INTER // 8, 2, 8, K)
        .permute(0, 2, 1, 3, 4)
        .reshape(EPR, 2 * INTER, K)
        .contiguous()
    )
    ref_scales_l1 = (
        w1_sf[:, 0]
        .reshape(EPR, INTER // 8, 2, 8)
        .permute(0, 2, 1, 3)
        .reshape(EPR, 2 * INTER, 1)
        .contiguous()
    )
    order = torch.arange(POOL_TOKENS, device=dev, dtype=torch.int32)
    expert_ids = block_expert_tab
    padded = torch.tensor([MAX_BLOCKS * BLOCK_M], device=dev, dtype=torch.int32)
    cfg = dict(
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=128,
        GROUP_SIZE_M=16,
        num_warps=4,
        num_stages=1,
    )
    ref_gu = torch.zeros((POOL_TOKENS, 1, 2 * INTER), device=dev, dtype=torch.bfloat16)
    vf.invoke_fused_moe_triton_kernel(
        l1_tok,
        ref_weights_l1,
        ref_gu,
        l1_sf[:POOL_TOKENS].view(-1, 1),
        ref_scales_l1,
        None,
        order,
        expert_ids,
        padded,
        False,
        1,
        cfg,
        tl.bfloat16,
        False,
        True,
        False,
        False,
        True,
    )
    ref_act = torch.empty((POOL_TOKENS, INTER), device=dev, dtype=torch.bfloat16)
    torch.ops._C.silu_and_mul(ref_act, ref_gu.view(POOL_TOKENS, 2 * INTER))
    # Engine initialization patches _int8_quantize to this native operation.
    # Call it explicitly: importing the upstream helper alone uses a different
    # ties-away Triton quantizer and is not the actual MetaX EP baseline.
    from vllm import _custom_ops as ops

    ref_q, ref_sf, _ = ops.scaled_int8_quant(ref_act)
    ref_out = torch.zeros((POOL_TOKENS, 1, K), device=dev, dtype=torch.bfloat16)
    vf.invoke_fused_moe_triton_kernel(
        ref_q,
        w2_q,
        ref_out,
        ref_sf,
        w2_sf[:, 0].unsqueeze(-1).contiguous(),
        l1_w.view(-1, 1),
        order,
        expert_ids,
        padded,
        True,
        1,
        cfg,
        tl.bfloat16,
        False,
        True,
        False,
        False,
        True,
    )
    return ref_q, ref_sf, ref_out[:, 0], ref_act, ref_gu[:, 0]


def reference(x, x_scale, w1, w2, s1, s2, routes, weights):
    """CPU integer GEMMs and BF16 rounding, with dynamic symmetric quantization."""
    x, x_scale, w1, w2, s1, s2, routes, weights = (
        t.cpu() for t in (x, x_scale, w1, w2, s1, s2, routes, weights)
    )
    i = w2.shape[-1]
    result = torch.zeros(x.shape, dtype=torch.float32)
    numerator = torch.tensor(127.0)

    for token in range(x.shape[0]):
        for slot in range(routes.shape[1]):
            expert = int(routes[token, slot])
            if expert < 0:
                continue
            gu = (w1[expert].int() @ x[token].int()).float()
            gu = (gu * x_scale[token] * s1[expert]).bfloat16().float()
            gate, up = gu[:i], gu[i:]
            silu = (gate / (1 + torch.exp(-gate))).bfloat16().float()
            act = (silu * up).bfloat16().float()
            amax = act.abs().max()
            scale = amax / 127
            # Tensor reverse division uses reciprocal * 127, changing ties.
            inv = torch.div(numerator, amax) if amax else 0.0
            q = (act * inv).round().to(torch.int8)
            partial = (w2[expert].int() @ q.int()).float()
            partial = (partial * scale * s2[expert] * weights[token, slot]).bfloat16()
            result[token] += partial.float()
    return result.bfloat16()


def make_inputs(t, h, i, experts, top_k, device="cuda"):
    gen = torch.Generator().manual_seed(71)
    x = torch.randint(-32, 33, (t, h), dtype=torch.int8, generator=gen)
    w1 = torch.randint(-32, 33, (experts, 2 * i, h), dtype=torch.int8, generator=gen)
    w2 = torch.randint(-32, 33, (experts, h, i), dtype=torch.int8, generator=gen)
    xs = torch.rand((t,), generator=gen) * 0.02 + 0.005
    s1 = torch.rand((experts, 2 * i), generator=gen) * 0.02 + 0.005
    s2 = torch.rand((experts, h), generator=gen) * 0.02 + 0.005
    routes = torch.stack(
        [torch.randperm(experts, generator=gen)[:top_k] for _ in range(t)]
    )
    weights = torch.rand((t, top_k), generator=gen)
    return tuple(z.to(device) for z in (x, xs, w1, w2, s1, s2, routes, weights))


@triton.jit
def zero_peer(table, rank: tl.constexpr, size: tl.constexpr, BLOCK: tl.constexpr):
    ptr = tl.load(table + rank).to(tl.pointer_type(tl.uint8))
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + idx, 0, mask=idx < size)


@triton.jit(do_not_specialize=["tokens"])
def read_local_partials(
    table,
    output,
    tokens,
    RANK: tl.constexpr,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    idx = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    row, col = idx // H, idx % H
    slot, token = row // tokens, row % tokens
    pointer = tl.load(table + RANK).to(tl.pointer_type(tl.bfloat16))
    value = tl.load(
        pointer + (slot * CAPACITY + token) * H + col,
        mask=idx < tokens * TOPK * H,
        other=0,
    )
    tl.store(output + idx, value, mask=idx < tokens * TOPK * H)


def snapshot_partials(ws, tokens):
    h = ws.model_shape[1]
    snapshot = torch.empty(
        (ws.top_k, tokens, h), device=ws.buffers[0].device, dtype=torch.bfloat16
    )
    if snapshot.numel():
        read_local_partials[(triton.cdiv(snapshot.numel(), 1024),)](
            ws.partials, snapshot, tokens, ws.rank, h, ws.top_k, ws.capacity
        )
    return snapshot.cpu()


def check_native(ws, w1, w2, s1, s2, local, lib):
    diagnostics = {}
    try:

        state, _, _, _, _, _, plan, metadata, pool, sf, weights, _, q, qs = ws.buffers
        e, _, h = w1.shape
        i = w2.shape[-1]
        blocks = int(state[1].cpu())
        gate = w1[:, :i].reshape(e, i // 8, 8, h)
        up = w1[:, i:].reshape(e, i // 8, 8, h)
        packed_w1 = torch.stack((gate, up), 2).reshape(e, 2 * i, h)
        gate_s = s1[:, :i].reshape(e, i // 8, 8)
        up_s = s1[:, i:].reshape(e, i // 8, 8)
        packed_s1 = torch.stack((gate_s, up_s), 2).reshape(e, 1, 2 * i)
        rq, rs, rp, _, _ = vllm_numerical_reference(
            pool,
            sf,
            weights,
            packed_w1,
            packed_s1,
            w2,
            s2[:, None, :],
            plan[:blocks, 0].contiguous(),
            blocks,
            ws.bm,
        )
        rows = torch.arange(blocks * ws.bm, device=w1.device)
        indices = rows[rows % ws.bm < plan[rows // ws.bm, 1]]
        diagnostics["native_q_bad"] = int((q[indices] != rq[indices]).sum())
        diagnostics["native_sf_bad"] = int((qs[indices] != rs[indices, 0]).sum())
        diagnostics["native_sf_nonfinite"] = int((~torch.isfinite(qs[indices])).sum())
        meta, expected = metadata[indices].cpu().long(), rp[indices].cpu()
        diagnostics["native_metadata_bad"] = int(
            (
                (meta[:, 0] < 0)
                | (meta[:, 0] >= ws.world_size)
                | (meta[:, 1] < 0)
                | (meta[:, 1] >= local.shape[1])
                | (meta[:, 2] < 0)
                | (meta[:, 2] >= ws.top_k)
            ).sum()
        )
    except Exception as error:
        diagnostics["native_error"] = f"{type(error).__name__}: {error}"
    # A rank-local oracle failure must not strand peers inside the broadcasts.
    ready = "native_error" not in diagnostics and not diagnostics.get(
        "native_metadata_bad", 0
    )
    if not lib.mxshmem_all_success_wrapper(int(ready)):
        diagnostics["native_exact"] = False
        return diagnostics
    bad, nonfinite, max_abs = 0, 0, 0.0
    for source in range(ws.world_size):
        received = local if source == ws.rank else torch.empty_like(local)
        lib.mxshmem_broadcast_bytes_wrapper(
            received.data_ptr(), received.numel() * received.element_size(), source
        )
        selected = meta[:, 0] == source
        got = received[meta[selected, 2], meta[selected, 1]]
        target = expected[selected]
        bad += int((got != target).sum())
        nonfinite += int((~torch.isfinite(got)).sum())
        if got.numel():
            max_abs = max(max_abs, float((got.float() - target.float()).abs().max()))
    diagnostics.update(
        native_partial_bad=bad,
        native_partial_nonfinite=nonfinite,
        native_partial_max_abs=max_abs,
    )
    diagnostics["native_exact"] = not any(
        diagnostics[key]
        for key in (
            "native_q_bad",
            "native_sf_bad",
            "native_sf_nonfinite",
            "native_metadata_bad",
            "native_partial_bad",
            "native_partial_nonfinite",
        )
    )
    return diagnostics


def check_output(output, local, routes, fixture):
    actual = output.cpu()
    routes = routes.cpu()
    combined = torch.zeros_like(actual, dtype=torch.float32)
    sum_abs = torch.zeros_like(combined)
    for slot in range(local.shape[0]):
        selected = routes[:, slot] >= 0
        partial = local[slot, selected].float()
        combined[selected] += partial
        sum_abs[selected] += partial.abs()
    target = fixture["output"].float()
    delta = actual.float() - target
    rel_l2 = (
        float(
            delta.double().square().sum()
            / target.double().square().sum().clamp_min(1e-30)
        )
        ** 0.5
    )
    diagnostics = dict(
        rel_l2=rel_l2,
        max_abs=float(delta.abs().max()) if delta.numel() else 0.0,
        combine_bad=int((actual != combined.to(torch.bfloat16)).sum()),
        reduction_bound_bad=int((delta.abs() > sum_abs * (8.0 / 256.0) + 1e-8).sum()),
        output_nonfinite=int((~torch.isfinite(delta)).sum()),
    )
    diagnostics["output_ok"] = (
        rel_l2 <= 0.005
        and not diagnostics["combine_bad"]
        and not diagnostics["reduction_bound_bad"]
        and not diagnostics["output_nonfinite"]
    )
    return diagnostics


def check_dispatch(ws, shared, tokens):
    state, _, _, _, _, _, plan, metadata, pool, sf, weights, *_ = ws.buffers
    blocks = int(state[1].cpu())
    if not 0 <= blocks <= plan.shape[0]:
        raise ValueError(f"invalid active block count: {blocks}")
    plan = plan[:blocks].cpu().long()
    rows = torch.arange(blocks * ws.bm)
    rows = rows[rows % ws.bm < plan[rows // ws.bm, 1]]
    meta = metadata[rows.to(metadata.device)].cpu().long()
    routes = shared["topk_idx"][: ws.world_size, :tokens]
    local_experts = ws.model_shape[0]
    first_expert = ws.rank * local_experts
    owned = (routes >= first_expert) & (routes < first_expert + local_experts)
    expected_keys = torch.nonzero(owned.reshape(-1)).flatten()
    metadata_bad = int(
        (
            (meta[:, 0] < 0)
            | (meta[:, 0] >= ws.world_size)
            | (meta[:, 1] < 0)
            | (meta[:, 1] >= tokens)
            | (meta[:, 2] < 0)
            | (meta[:, 2] >= ws.top_k)
        ).sum()
    )
    diagnostics = dict(
        dispatch_expected_routes=expected_keys.numel(),
        dispatch_actual_routes=meta.shape[0],
        dispatch_metadata_bad=metadata_bad,
        dispatch_ok=False,
    )
    if metadata_bad:
        return diagnostics
    source, token, slot = meta.unbind(1)
    actual_keys = (source * tokens + token) * ws.top_k + slot
    diagnostics.update(
        dispatch_route_set_bad=int(
            not torch.equal(actual_keys.sort().values, expected_keys)
        ),
        dispatch_expert_bad=int(
            (routes[source, token, slot] != first_expert + plan[rows // ws.bm, 0]).sum()
        ),
        dispatch_token_bad=int(
            (
                pool[rows.to(pool.device)].cpu() != shared["input_int8"][source, token]
            ).sum()
        ),
        dispatch_scale_bad=int(
            (
                sf[rows.to(sf.device)].cpu()
                != shared["input_scales"][source, token].reshape(-1)
            ).sum()
        ),
        dispatch_weight_bad=int(
            (
                weights[rows.to(weights.device)].cpu()
                != shared["topk_weights"][source, token, slot]
            ).sum()
        ),
    )
    diagnostics["dispatch_ok"] = not any(
        value for name, value in diagnostics.items() if name.endswith("_bad")
    )
    return diagnostics


def validate(ws, w1, w2, s1, s2, output, routes, fixture, shared, lib, native, phase):
    diagnostics = dict(
        kind="correctness",
        correctness_phase=phase,
        rank=ws.rank,
        world=ws.world_size,
        tokens=output.shape[0],
        input_hashes_match=True,
        native_checked=native,
        output_ok=False,
        dispatch_ok=False,
    )
    # Inspect destination-owned storage before either oracle can affect GPU state.
    local = None
    try:
        local = snapshot_partials(ws, output.shape[0])
        diagnostics.update(check_output(output, local, routes, fixture))
        diagnostics.update(check_dispatch(ws, shared, output.shape[0]))
    except Exception as error:
        diagnostics["validation_error"] = f"{type(error).__name__}: {error}"
    snapshots_ready = lib.mxshmem_all_success_wrapper(
        int(
            local is not None
            and "validation_error" not in diagnostics
            and not diagnostics.get("dispatch_metadata_bad", 0)
        )
    )
    if native and snapshots_ready:
        diagnostics.update(check_native(ws, w1, w2, s1, s2, local, lib))
    elif native:
        diagnostics["native_exact"] = False
    passed = bool(
        lib.mxshmem_all_success_wrapper(
            int(
                diagnostics["output_ok"]
                and diagnostics["dispatch_ok"]
                and "validation_error" not in diagnostics
                and (not native or diagnostics["native_exact"])
            )
        )
    )
    diagnostics["status"] = "PASS" if passed else "FAIL"
    print(json.dumps(diagnostics), flush=True)
    return passed


def measure(run, lib, rank, tokens, warmup, iters, repeats):
    """PR #774 cadence: complete every CUDA-event sample before the next."""

    lib.mxshmemx_barrier_wrapper.argtypes = [ctypes.c_void_p]
    lib.mxshmemx_barrier_wrapper.restype = None

    def barrier(stream):
        lib.mxshmemx_barrier_wrapper(ctypes.c_void_p(stream.cuda_stream))

    means = []
    with KernelTimer(barrier) as timer:
        for repeat in range(repeats):
            for _ in range(warmup):
                run()
            torch.cuda.synchronize()
            lib.mxshmem_all_success_wrapper(1)
            times = []
            for _ in range(iters):
                timer.arm()
                run()
                times.append(timer.finish() * 1000.0)
                # Complete the untimed stream barrier before the host collective.
                torch.cuda.synchronize()
                lib.mxshmem_all_success_wrapper(1)
            mean_us = sum(times) / len(times)
            means.append(mean_us)
            # All device work is complete before the repeat-level host collective.
            torch.cuda.synchronize()
            lib.mxshmem_all_success_wrapper(1)
            print(
                f"BENCH_REPEAT rank={rank} tokens={tokens} repeat={repeat} "
                f"mean_us={mean_us:.9f}",
                flush=True,
            )
    return means


def _run_worker():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--num-ranks", type=int)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--intermediate", type=int, default=512)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument(
        "--host-lib", default=os.environ.get("MEGAMOE_HOST_LIB", "libmegamoe_host.so")
    )
    ap.add_argument("--data")
    ap.add_argument("--references")
    ap.add_argument("--scale-lut")
    ap.add_argument("--tokens", default="1,128,256")
    ap.add_argument("--check-native", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--stress-batched-partials", action="store_true")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    if args.warmup < 0 or args.iters < 1 or args.repeats < 1:
        ap.error("warmup must be nonnegative; iters and repeats must be positive")
    if args.data and (not args.references or not args.scale_lut):
        ap.error("--data requires --references and --scale-lut")
    lib = ctypes.CDLL(args.host_lib)
    lib.mxshmem_alloc_bytes_wrapper.argtypes = [ctypes.c_size_t]
    lib.mxshmem_alloc_bytes_wrapper.restype = ctypes.c_void_p
    lib.mxshmem_ptr_wrapper.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.mxshmem_ptr_wrapper.restype = ctypes.c_void_p
    lib.mxshmem_free_wrapper.argtypes = [ctypes.c_void_p]
    lib.mxshmem_broadcast_bytes_wrapper.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.mxshmem_all_success_wrapper.argtypes = [ctypes.c_int]
    lib.mxshmem_all_success_wrapper.restype = ctypes.c_int
    # Initialize native schemas before library compatibility schemas.
    importlib.import_module("vllm._custom_ops")
    from flaggems_vllm import megamoe

    torch.cuda.set_device(int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]))
    # MACA's SHMEM stream setup needs the selected device's primary context.
    context_guard = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    lib.mxshmem_init_wrapper()
    rank, world = lib.mxshmem_mype_wrapper(), lib.mxshmem_npes_wrapper()
    if args.num_ranks is not None:
        assert world == args.num_ranks
    allocations = []
    failures = 0

    def release_allocations():
        torch.cuda.synchronize()
        lib.mxshmem_barrier_all_wrapper()
        for allocation in allocations:
            lib.mxshmem_free_wrapper(allocation)
        allocations.clear()

    def allocate(nbytes):
        lib.mxshmem_barrier_all_wrapper()
        pointer = lib.mxshmem_alloc_bytes_wrapper(nbytes)
        if not lib.mxshmem_all_success_wrapper(int(bool(pointer))):
            raise RuntimeError("symmetric allocation failed on one or more ranks")
        allocations.append(pointer)
        addresses = [
            pointer if peer == rank else lib.mxshmem_ptr_wrapper(pointer, peer)
            for peer in range(world)
        ]
        if not lib.mxshmem_all_success_wrapper(int(all(addresses))):
            raise RuntimeError("symmetric peer mapping failed on one or more ranks")
        table = torch.tensor(addresses, device="cuda", dtype=torch.int64)
        zero_peer[(triton.cdiv(nbytes, 1024),)](table, rank, nbytes, 1024)
        torch.cuda.synchronize()
        lib.mxshmem_barrier_all_wrapper()
        return table

    if args.data:

        dataset = load_qwen3_canonical(args.data, torch=torch)
        shared, experts = dataset["shared"], dataset["experts"]
        e = experts["l1_weight_int8"].shape[0] // world
        owned = slice(rank * e, (rank + 1) * e)
        w1 = experts["l1_weight_int8"][owned].to("cuda")
        w2 = experts["l2_weight_int8"][owned].to("cuda")
        s1 = experts["l1_weight_scale"][owned].reshape(e, -1).to("cuda")
        s2 = experts["l2_weight_scale"][owned].reshape(e, -1).to("cuda")
        sizes = [int(t) for t in args.tokens.split(",")]
        lut = torch.frombuffer(
            bytearray(Path(args.scale_lut).read_bytes()), dtype=torch.int32
        ).to("cuda")
        for t in sizes:
            diagnostics = dict(rank=rank, world=world, tokens=t)
            try:
                fixture = torch.load(
                    Path(args.references) / f"t{t}_rank{rank}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                inputs = dict(
                    hidden_q=shared["input_int8"][rank, :t],
                    hidden_scale=shared["input_scales"][rank, :t],
                    topk_ids=shared["topk_idx"][rank, :t],
                    topk_weights=shared["topk_weights"][rank, :t],
                )
                hashes = {
                    name: hashlib.sha256(
                        tensor.contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                    for name, tensor in inputs.items()
                }
                diagnostics["input_hash_bad"] = [
                    name
                    for name, digest in hashes.items()
                    if digest != fixture["input_sha256"][name]
                ]
                diagnostics["input_hashes_match"] = not diagnostics["input_hash_bad"]
            except Exception as error:
                diagnostics["input_error"] = f"{type(error).__name__}: {error}"
                diagnostics["input_hashes_match"] = False
            if not lib.mxshmem_all_success_wrapper(
                int(diagnostics["input_hashes_match"])
            ):
                failures += 1
                diagnostics.update(kind="shape_summary", status="FAIL")
                print(json.dumps(diagnostics), flush=True)
                continue
            x, xs, routes, weights = (
                inputs[name].to("cuda")
                for name in ("hidden_q", "hidden_scale", "topk_ids", "topk_weights")
            )
            # Capacity is a deployment choice, so benchmark each real shape with
            # its own capacity; a 16384-token workspace distorts every small case.
            with create_workspace(
                x,
                w1,
                w2,
                weights,
                routes,
                s1,
                s2,
                a1_scale=xs,
                max_tokens=t,
                rank=rank,
                world_size=world,
                allocate_symmetric=allocate,
                quant_scale_lut=lut,
                release_symmetric=release_allocations,
                shmem=lib,
            ) as ws:
                out = torch.empty(
                    (t, x.shape[1]), device=x.device, dtype=torch.bfloat16
                )

                def run():
                    return megamoe(
                        x,
                        w1,
                        w2,
                        weights,
                        routes,
                        s1,
                        s2,
                        workspace=ws,
                        a1_scale=xs,
                        out=out,
                    )

                y = run()
                torch.cuda.synchronize()
                pre_passed = validate(
                    ws,
                    w1,
                    w2,
                    s1,
                    s2,
                    y,
                    routes,
                    fixture,
                    shared,
                    lib,
                    args.check_native,
                    "pre",
                )
                post_passed = None
                if args.benchmark and pre_passed:
                    means = measure(
                        run, lib, rank, t, args.warmup, args.iters, args.repeats
                    )
                    # Validate the final measured invocation without replacing its output.
                    post_passed = validate(
                        ws,
                        w1,
                        w2,
                        s1,
                        s2,
                        out,
                        routes,
                        fixture,
                        shared,
                        lib,
                        args.check_native,
                        "post",
                    )
                    print(
                        json.dumps(
                            dict(
                                kind="benchmark",
                                rank=rank,
                                world=world,
                                tokens=t,
                                warmup=args.warmup,
                                iters=args.iters,
                                repeats=args.repeats,
                                mean_us=means,
                                timing="CUDA events at launch hooks; individual sample completion",
                                timing_protocol=TIMING_PROTOCOL,
                                external_barriers_in_timing=False,
                                internal_barriers_in_timing=True,
                                includes=(
                                    "dispatch, dynamic routing, both GEMMs, "
                                    "activation, quantization, combine"
                                ),
                                capacity=ws.capacity,
                                block_m=ws.bm,
                                correctness="PASS" if post_passed else "FAIL",
                            )
                        ),
                        flush=True,
                    )
                passed = pre_passed and (post_passed is not False)
                diagnostics.update(
                    kind="shape_summary",
                    status="PASS" if passed else "FAIL",
                    pre_passed=pre_passed,
                    post_passed=post_passed,
                    native_checked=args.check_native,
                    capacity=ws.capacity,
                    block_m=ws.bm,
                    block_n=ws.bn,
                    block_k=ws.bk,
                    workers=ws.workers,
                )
                print(json.dumps(diagnostics), flush=True)
                if not passed:
                    failures += 1
                del run, ws, y, out
    elif args.synthetic:
        assert args.num_experts % world == 0
        for t in map(int, args.tokens.split(",")):
            x, xs, full_w1, full_w2, full_s1, full_s2, routes, weights = make_inputs(
                t,
                args.hidden,
                args.intermediate,
                args.num_experts,
                args.topk,
                device="cpu",
            )
            e = args.num_experts // world
            owned = slice(rank * e, (rank + 1) * e)
            x, xs, routes, weights = (
                value.to("cuda") for value in (x, xs, routes, weights)
            )
            w1, w2, s1, s2 = (
                value[owned].to("cuda")
                for value in (full_w1, full_w2, full_s1, full_s2)
            )
            del full_w1, full_w2, full_s1, full_s2
            with create_workspace(
                x,
                w1,
                w2,
                weights,
                routes,
                s1,
                s2,
                a1_scale=xs,
                max_tokens=t,
                rank=rank,
                world_size=world,
                allocate_symmetric=allocate,
                release_symmetric=release_allocations,
                shmem=lib,
            ) as ws:
                out = torch.empty((t, args.hidden), device="cuda", dtype=torch.bfloat16)

                def run():
                    return megamoe(
                        x,
                        w1,
                        w2,
                        weights,
                        routes,
                        s1,
                        s2,
                        workspace=ws,
                        a1_scale=xs,
                        out=out,
                    )

                # Warm the explicit workspace specialization before timing, including --warmup=0.
                run()
                means = measure(
                    run, lib, rank, t, args.warmup, args.iters, args.repeats
                )
                if not lib.mxshmem_all_success_wrapper(int(torch.isfinite(out).all())):
                    failures += 1
                for repeat, mean in enumerate(means):
                    print(
                        f"MEGAMOE_RESULT scope=kernel rank={rank} world={world} "
                        f"repeat={repeat} latency_us={mean:.9f}",
                        flush=True,
                    )
                del run, out
    else:
        sizes = (1025, 33, 0, 129) if args.stress_batched_partials else (19, 3, 0, 17)
        capacity = (
            torch.cuda.get_device_properties(0).multi_processor_count * 64
            if args.stress_batched_partials
            else 19
        )
        x, xs, full_w1, full_w2, full_s1, full_s2, routes, weights = make_inputs(
            max(sizes), 128, 70, world * 3, 2
        )
        owned = slice(rank * 3, (rank + 1) * 3)
        w1, w2, s1, s2 = (a[owned] for a in (full_w1, full_w2, full_s1, full_s2))
        with create_workspace(
            x,
            w1,
            w2,
            weights,
            routes,
            s1,
            s2,
            a1_scale=xs,
            workers=2 if args.stress_batched_partials else None,
            max_tokens=capacity,
            rank=rank,
            world_size=world,
            allocate_symmetric=allocate,
            release_symmetric=release_allocations,
            shmem=lib,
        ) as ws:
            for iteration in range(8):
                t = sizes[(iteration + rank) % 4]
                current = routes[:t].clone()
                current[:, 0] = (iteration + 1) % (world * 3)
                current[:, 1] = -1
                y = megamoe(
                    x[:t],
                    w1,
                    w2,
                    weights[:t],
                    current,
                    s1,
                    s2,
                    workspace=ws,
                    a1_scale=xs[:t],
                )
                expected = reference(
                    x[:t],
                    xs[:t],
                    full_w1,
                    full_w2,
                    full_s1,
                    full_s2,
                    current,
                    weights[:t],
                ).to("cuda")
                close = torch.isclose(y, expected, rtol=0.02, atol=0.015)
                good = bool(close.all())
                if not good:
                    bad = (~close).nonzero()[:8]
                    print(
                        json.dumps(
                            dict(
                                rank=rank,
                                iteration=iteration,
                                kind="dynamic_mismatch",
                                count=int((~close).sum()),
                                coordinates=bad.cpu().tolist(),
                                actual=y[bad[:, 0], bad[:, 1]].float().cpu().tolist(),
                                expected=expected[bad[:, 0], bad[:, 1]]
                                .float()
                                .cpu()
                                .tolist(),
                                relative_l2=float(
                                    (y.float() - expected.float()).norm()
                                    / expected.float().norm().clamp_min(1e-10)
                                ),
                            )
                        ),
                        flush=True,
                    )
                if not lib.mxshmem_all_success_wrapper(int(good)):
                    failures += 1
                    print(
                        json.dumps(
                            dict(rank=rank, iteration=iteration, status="DYNAMIC_FAIL")
                        ),
                        flush=True,
                    )
            print(
                json.dumps(
                    dict(
                        rank=rank,
                        world=world,
                        iterations=8,
                        status="DYNAMIC_FAIL" if failures else "DYNAMIC_PASS",
                    )
                ),
                flush=True,
            )
            t = sizes[1] if rank % 2 == 0 else sizes[0]
            current = routes[:t].clone()
            out = torch.empty((t, 128), device="cuda", dtype=torch.bfloat16)
            megamoe(
                x[:t],
                w1,
                w2,
                weights[:t],
                current,
                s1,
                s2,
                workspace=ws,
                a1_scale=xs[:t],
                out=out,
            )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                megamoe(
                    x[:t],
                    w1,
                    w2,
                    weights[:t],
                    current,
                    s1,
                    s2,
                    workspace=ws,
                    a1_scale=xs[:t],
                    out=out,
                )
            graph_failures = 0
            for iteration in range(4):
                current[:, 0] = iteration % (world * 3)
                current[:, 1] = -1
                graph.replay()
                expected = reference(
                    x[:t],
                    xs[:t],
                    full_w1,
                    full_w2,
                    full_s1,
                    full_s2,
                    current,
                    weights[:t],
                ).to("cuda")
                good = bool(torch.isclose(out, expected, rtol=0.02, atol=0.015).all())
                if not lib.mxshmem_all_success_wrapper(int(good)):
                    graph_failures += 1
                    print(
                        json.dumps(
                            dict(rank=rank, iteration=iteration, status="GRAPH_FAIL")
                        ),
                        flush=True,
                    )
            failures += graph_failures
            print(
                json.dumps(
                    dict(
                        rank=rank,
                        world=world,
                        replays=4,
                        status="GRAPH_FAIL" if graph_failures else "GRAPH_PASS",
                    )
                ),
                flush=True,
            )
            del graph, ws
    release_allocations()
    lib.mxshmem_finalize_wrapper()
    del context_guard
    return int(failures > 0)


# Standalone workers enter before the benchmark framework imports package schemas.
if __name__ == "__main__":
    raise SystemExit(_run_worker())


import pytest  # noqa: E402

from . import base  # noqa: E402
from .conftest import Config  # noqa: E402
from .consts import DEFAULT_ITER_TIME, DEFAULT_WARMUP_TIME, BenchMode  # noqa: E402

OP_NAME = "megamoe"
NUM_RANKS = 8
DEFAULT_WARMUP = 20
DEFAULT_ITERS = 100
DEFAULT_REPEATS = 5
DEFAULT_SHAPES = [
    (512, 2048, 512, 256, 8, NUM_RANKS),
    (1024, 2048, 512, 256, 8, NUM_RANKS),
    (2048, 2048, 512, 256, 8, NUM_RANKS),
]
_RESULT = re.compile(
    r"MEGAMOE_RESULT scope=kernel rank=(\d+) world=(\d+) repeat=(\d+) latency_us=([0-9.]+)"
)


@dataclass(frozen=True)
class _BenchmarkConfig:
    num_ranks: int
    tokens: int
    hidden: int
    intermediate: int
    num_experts: int
    topk: int
    warmup: int = DEFAULT_WARMUP
    iterations: int = DEFAULT_ITERS
    repeats: int = DEFAULT_REPEATS
    timeout: int = 1800
    host_lib: str = os.environ.get("MEGAMOE_HOST_LIB", "libmegamoe_host.so")


def _launch_megamoe(config: _BenchmarkConfig) -> float:
    command = [
        str(Path(os.environ.get("MACA_PATH", "/opt/maca")) / "ompi/bin/mpirun"),
        "--allow-run-as-root",
        "--mca",
        "btl",
        "self,tcp",
        "--bind-to",
        "none",
        "-np",
        str(config.num_ranks),
        sys.executable,
        "-u",
        "-m",
        "benchmark.test_megamoe",
        "--worker",
        "--synthetic",
        "--host-lib",
        config.host_lib,
        "--num-ranks",
        str(config.num_ranks),
        "--tokens",
        str(config.tokens),
        "--hidden",
        str(config.hidden),
        "--intermediate",
        str(config.intermediate),
        "--num-experts",
        str(config.num_experts),
        "--topk",
        str(config.topk),
        "--warmup",
        str(config.warmup),
        "--iters",
        str(config.iterations),
        "--repeats",
        str(config.repeats),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=config.timeout)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command, output=stdout)
    means = {
        (int(rank), int(repeat)): float(latency) / 1e3
        for rank, world, repeat, latency in _RESULT.findall(stdout)
        if int(world) == config.num_ranks
    }
    expected = {
        (rank, repeat)
        for rank in range(config.num_ranks)
        for repeat in range(config.repeats)
    }
    if set(means) != expected:
        raise RuntimeError(f"missing rank/repeat results:\n{stdout}")
    return statistics.mean(
        max(means[rank, repeat] for rank in range(config.num_ranks))
        for repeat in range(config.repeats)
    )


def _iteration_count(option, configured, default, fallback):
    explicit = any(arg == option or arg.startswith(f"{option}=") for arg in sys.argv)
    return configured if explicit or configured != default else fallback


class MegaMoEBenchmark(base.Benchmark):
    DEFAULT_METRICS = ["latency"]
    DEFAULT_DTYPES = [torch.int8]
    DEFAULT_SHAPE_DESC = "tokens, hidden, intermediate, num_experts, topk, num_ranks"

    def __init__(self):
        super().__init__(
            op_name=OP_NAME,
            torch_op=_launch_megamoe,
            gems_op=_launch_megamoe,
            dtypes=self.DEFAULT_DTYPES,
        )

    def set_shapes(self, shape_file_path=None):
        self.shapes = DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, dtype):
        yield from self.shapes

    def get_latency(
        self, op, tokens, hidden, intermediate, num_experts, topk, num_ranks
    ):
        assert Config.mode is BenchMode.KERNEL
        return op(
            _BenchmarkConfig(
                num_ranks=num_ranks,
                tokens=tokens,
                hidden=hidden,
                intermediate=intermediate,
                num_experts=num_experts,
                topk=topk,
                warmup=_iteration_count(
                    "--warmup",
                    Config.warm_up,
                    DEFAULT_WARMUP_TIME,
                    DEFAULT_WARMUP,
                ),
                iterations=_iteration_count(
                    "--iter",
                    Config.repetition,
                    DEFAULT_ITER_TIME,
                    DEFAULT_ITERS,
                ),
            )
        )


@pytest.mark.megamoe
def test_megamoe():
    if Config.query:
        MegaMoEBenchmark().run()
        return
    from flaggems_vllm.runtime import device

    if device.vendor_name != "metax":
        pytest.skip("requires MetaX")
    if torch.cuda.device_count() < NUM_RANKS:
        pytest.skip(f"requires {NUM_RANKS} visible GPUs")
    MegaMoEBenchmark().run()
