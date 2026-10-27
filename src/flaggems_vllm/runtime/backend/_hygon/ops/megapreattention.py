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

"""Hygon BF16 MegaPreAttention, derived from NVIDIA PR #1007 (89b0a512).

Portable dot and system-scope P2P replace PTX, TMA and WGMMA.
IPC payloads use fine-grained memory; only the local GEMM staging is cached. All ranks
must call in the same order on one stream. Prepare workspace before capture.
The two epoch barriers protect both publication and reuse of peer payloads.
"""

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import triton
import triton.language as tl

__all__ = ["megapreattention", "clear_megapreattention_cache"]


@dataclass(frozen=True)
class _Config:
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    mrope_section: tuple[int, int, int]
    eps: float
    hidden_size: int

    @property
    def q_size(self):
        return self.num_q_heads * self.head_dim

    @property
    def kv_size(self):
        return self.num_kv_heads * self.head_dim

    @property
    def packed_size(self):
        return 2 * (self.q_size + self.kv_size)


@dataclass
class _Inputs:
    hidden: torch.Tensor
    weight: torch.Tensor
    positions: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    rope: torch.Tensor
    cache: torch.Tensor
    slots: torch.Tensor
    group: object
    config: _Config


@triton.jit
def _store_head(
    acc,
    query,
    key,
    value,
    gate,
    qnorm,
    knorm,
    positions,
    rope,
    cache,
    slots,
    row_start,
    pid_m,
    pid_n,
    LOCAL_T: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    POS_AXIS,
    POS_TOKEN,
    ROPE_STRIDE,
    SLOT_STRIDE,
    CACHE_BLOCK,
    CACHE_HEAD,
    CACHE_TOKEN,
    CACHE_DIM,
    CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SECTION_H: tl.constexpr,
    SECTION_W: tl.constexpr,
    EPS: tl.constexpr = 1.0e-6,
):
    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token = row_start + row
    valid = row < LOCAL_T
    cols = tl.arange(0, 256)
    raw = acc.to(tl.bfloat16)
    # Preserve the materialized BF16 projection and norm rounding boundaries.
    if pid_n < 2 * Q_HEADS and pid_n % 2 == 1:
        head = pid_n // 2
        tl.store(
            gate + token[:, None] * (Q_HEADS * 256) + head * 256 + cols[None, :],
            raw,
            mask=valid[:, None],
        )
    elif pid_n >= 2 * Q_HEADS + KV_HEADS:
        head = pid_n - 2 * Q_HEADS - KV_HEADS if KV_HEADS > 1 else 0
        tl.store(
            value + token[:, None] * (KV_HEADS * 256) + head * 256 + cols[None, :],
            raw,
            mask=valid[:, None],
        )
        slot = tl.load(slots + token * SLOT_STRIDE, mask=valid, other=-1).to(tl.int64)
        target = (
            cache
            + (slot // CACHE_BLOCK_SIZE)[:, None] * CACHE_BLOCK
            + head * CACHE_HEAD
            + (slot % CACHE_BLOCK_SIZE)[:, None] * CACHE_TOKEN
        )
        tl.store(
            target + (256 + cols[None, :]) * CACHE_DIM,
            raw,
            mask=(valid & (slot >= 0))[:, None],
        )
    else:
        is_key = pid_n >= 2 * Q_HEADS if KV_HEADS > 1 else pid_n == 2 * Q_HEADS
        head = pid_n // 2
        weight = tl.load(tl.where(is_key, knorm, qnorm) + cols).to(tl.float32)
        x = raw.to(tl.float32)
        scale = tl.rsqrt(tl.sum(x * x, 1) / 256.0 + EPS)
        normalized = (
            (x * scale[:, None] * (weight[None, :] + 1.0))
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        frequency = tl.arange(0, 32)
        axis = tl.where(
            (frequency % 3 == 1) & (frequency < 3 * SECTION_H),
            1,
            tl.where((frequency % 3 == 2) & (frequency < 3 * SECTION_W), 2, 0),
        )
        pos = tl.load(
            positions + axis[None, :] * POS_AXIS + token[:, None] * POS_TOKEN,
            valid[:, None],
            other=0,
        )
        co = tl.load(
            rope + pos * ROPE_STRIDE + frequency[None, :], valid[:, None], other=1
        ).to(tl.float32)
        si = tl.load(
            rope + pos * ROPE_STRIDE + 32 + frequency[None, :], valid[:, None], other=0
        ).to(tl.float32)
        # Only the first 64 dimensions rotate; pass the other 192 through.
        # Static splits avoid a full-width gather and unused rotary arithmetic.
        low128, high128 = tl.split(normalized.reshape(BLOCK_M, 2, 128).permute(0, 2, 1))
        low64, high64 = tl.split(low128.reshape(BLOCK_M, 2, 64).permute(0, 2, 1))
        first, second = tl.split(low64.reshape(BLOCK_M, 2, 32).permute(0, 2, 1))
        first_out = first * co - second * si
        second_out = second * co + first * si
        rotated64 = tl.join(first_out, second_out).permute(0, 2, 1).reshape(BLOCK_M, 64)
        result128 = tl.join(rotated64, high64).permute(0, 2, 1).reshape(BLOCK_M, 128)
        result = tl.join(result128, high128).permute(0, 2, 1).reshape(BLOCK_M, 256)
        if is_key:
            kv_head = pid_n - 2 * Q_HEADS if KV_HEADS > 1 else 0
            tl.store(
                key + token[:, None] * (KV_HEADS * 256) + kv_head * 256 + cols[None, :],
                result,
                mask=valid[:, None],
            )
            slot = tl.load(slots + token * SLOT_STRIDE, mask=valid, other=-1).to(
                tl.int64
            )
            target = (
                cache
                + (slot // CACHE_BLOCK_SIZE)[:, None] * CACHE_BLOCK
                + kv_head * CACHE_HEAD
                + (slot % CACHE_BLOCK_SIZE)[:, None] * CACHE_TOKEN
            )
            tl.store(
                target + cols[None, :] * CACHE_DIM,
                result,
                mask=(valid & (slot >= 0))[:, None],
            )
        else:
            tl.store(
                query + token[:, None] * (Q_HEADS * 256) + head * 256 + cols[None, :],
                result,
                mask=valid[:, None],
            )


@triton.jit
def _init_metadata(flags, epoch, peers, ADDRESSES: tl.constexpr, WORLD: tl.constexpr):
    ranks = tl.arange(0, WORLD)
    tl.store(flags + ranks, 0)
    tl.store(epoch, 0)
    for i in tl.static_range(WORLD):
        tl.store(peers + i, ADDRESSES[i])


@triton.jit
def _peer_barrier(
    flags, epoch, peers, rank, DATA_WORDS: tl.constexpr, WORLD: tl.constexpr
):
    target = tl.atomic_add(epoch, 1, sem="relaxed") + 1
    ranks = tl.arange(0, WORLD)
    remote = tl.load(peers + ranks).to(tl.pointer_type(tl.int64))
    tl.atomic_xchg(remote + DATA_WORDS + rank, target, sem="release", scope="sys")
    waiting = tl.full((), 1, tl.int32)
    while waiting != 0:
        seen = tl.atomic_add(flags + ranks, 0, sem="acquire", scope="sys")
        waiting = tl.sum((seen < target).to(tl.int32), 0)
    tl.debug_barrier()


@triton.jit
def _push(hidden, peers, rank, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    target = tl.load(peers + tl.program_id(1)).to(tl.pointer_type(tl.bfloat16))
    x = tl.load(hidden + offset, offset < N, other=0)
    tl.store(target + rank * N + offset, x, offset < N)


@triton.jit
def _project(
    hidden,
    weight,
    query,
    key,
    value,
    gate,
    qnorm,
    knorm,
    positions,
    rope,
    cache,
    slots,
    POS_AXIS,
    POS_TOKEN,
    ROPE_STRIDE,
    SLOT_STRIDE,
    CACHE_BLOCK,
    CACHE_HEAD,
    CACHE_TOKEN,
    CACHE_DIM,
    M: tl.constexpr,
    HIDDEN: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    SECTION_H: tl.constexpr,
    SECTION_W: tl.constexpr,
    EPS: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m: tl.constexpr = tl.cdiv(M, BM)
    heads: tl.constexpr = 2 * (Q_HEADS + KV_HEADS)
    group = pid // (GROUP_M * heads)
    first_m = group * GROUP_M
    group_m = tl.minimum(num_m - first_m, GROUP_M)
    within = pid % (GROUP_M * heads)
    pm = first_m + within % group_m
    pn = within // group_m
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * 256 + tl.arange(0, 256)
    kk = tl.arange(0, BK)
    a_ptr = hidden + rows[:, None] * HIDDEN + kk[None, :]
    b_ptr = weight + cols[None, :] * HIDDEN + kk[:, None]
    acc = tl.zeros((BM, 256), tl.float32)
    for k in range(HIDDEN // BK):
        a = tl.load(a_ptr, rows[:, None] < M, other=0)
        b = tl.load(b_ptr)
        acc = tl.dot(a, b, acc)
        a_ptr += BK
        b_ptr += BK
    _store_head(
        acc,
        query,
        key,
        value,
        gate,
        qnorm,
        knorm,
        positions,
        rope,
        cache,
        slots,
        0,
        pm,
        pn,
        M,
        Q_HEADS,
        KV_HEADS,
        POS_AXIS,
        POS_TOKEN,
        ROPE_STRIDE,
        SLOT_STRIDE,
        CACHE_BLOCK,
        CACHE_HEAD,
        CACHE_TOKEN,
        CACHE_DIM,
        CACHE_BLOCK_SIZE,
        BM,
        SECTION_H,
        SECTION_W,
        EPS,
    )


@triton.jit
def _cache_local(source, target, N: tl.constexpr, BLOCK: tl.constexpr):
    # No peer accesses this ordinary allocation; IPC stays fine-grained.
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    data = tl.load(source + i, i < N, other=0)
    tl.store(target + i, data, i < N)


class _DevicePointer:
    def __init__(self, owner, offset, shape, dtype):
        self.owner, self.offset, self.shape, self.dtype = owner, offset, shape, dtype
        self.device = owner.device

    def data_ptr(self):
        return self.owner.data_ptr() + self.offset

    def numel(self):
        return math.prod(self.shape)

    def fill_(self, value):
        _fill_buffer[(triton.cdiv(self.numel(), 1024),)](
            self, value, self.numel(), BLOCK=1024, num_warps=4
        )
        return self


@triton.jit
def _fill_buffer(pointer, value, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(pointer + i, value, i < N)


class _HipAllocation:
    """Fine-grained device allocation for system-scope IPC atomics."""

    def __init__(self, size, device):
        import ctypes
        import ctypes.util

        self.device = device
        self.size = size
        self.lib = ctypes.CDLL(ctypes.util.find_library("amdhip64") or "libamdhip64.so")
        self.lib.hipExtMallocWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        self.lib.hipFree.argtypes = [ctypes.c_void_p]
        self.pointer = ctypes.c_void_p()
        result = self.lib.hipExtMallocWithFlags(ctypes.byref(self.pointer), size, 1)
        if result:
            raise RuntimeError(f"hipExtMallocWithFlags(fine-grained) failed: {result}")

    def data_ptr(self):
        return self.pointer.value

    def close(self):
        result = self.lib.hipFree(self.pointer)
        if result:
            raise RuntimeError(f"hipFree failed: {result}")
        self.pointer = None


_ipc_mappings = {}


class _HipMapping:
    """HIP IPC mapping of a live device allocation; no torch data movement.

    Export the allocation
    base and exchange the byte offset, rather than exporting the interior ptr.
    All owners remain alive until collective close, including captured graphs.
    """

    def __init__(self, allocation, group):
        import ctypes
        import ctypes.util

        class Handle(ctypes.Structure):
            _fields_ = [("reserved", ctypes.c_byte * 64)]

        self.ctypes, self.Handle = ctypes, Handle
        self.lib = ctypes.CDLL(ctypes.util.find_library("amdhip64") or "libamdhip64.so")
        self.lib.hipGetErrorString.restype = ctypes.c_char_p
        self.lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(Handle), ctypes.c_void_p]
        self.lib.hipIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            Handle,
            ctypes.c_uint,
        ]
        self.lib.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        self.lib.hipMemGetAddressRange.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        ]
        self.allocation, self.rank = allocation, dist.get_rank(group)
        self.remote = []
        base, size, handle = ctypes.c_void_p(), ctypes.c_size_t(), Handle()
        self._check(
            self.lib.hipMemGetAddressRange(
                ctypes.byref(base), ctypes.byref(size), allocation.data_ptr()
            )
        )
        self._check(self.lib.hipIpcGetMemHandle(ctypes.byref(handle), base))
        offset = allocation.data_ptr() - base.value
        descriptor = (ctypes.string_at(ctypes.byref(handle), 64), offset)
        handles = [None] * dist.get_world_size(group)
        dist.all_gather_object(handles, descriptor, group=group)
        self.buffer_ptrs = []
        for rank, (raw, byte_offset) in enumerate(handles):
            if rank == self.rank:
                self.buffer_ptrs.append(allocation.data_ptr())
            else:
                key = (allocation.device, raw)
                if key in _ipc_mappings:
                    pointer, references = _ipc_mappings[key]
                    _ipc_mappings[key] = (pointer, references + 1)
                else:
                    pointer = ctypes.c_void_p()
                    self._check(
                        self.lib.hipIpcOpenMemHandle(
                            ctypes.byref(pointer), Handle.from_buffer_copy(raw), 1
                        )
                    )
                    _ipc_mappings[key] = (pointer, 1)
                self.remote.append(key)
                self.buffer_ptrs.append(pointer.value + byte_offset)

    def _check(self, code):
        if code:
            raise RuntimeError(
                f"HIP IPC error {code}: {self.lib.hipGetErrorString(code).decode()}"
            )

    def get_buffer(self, rank, shape, dtype, storage_offset=0):
        if rank != self.rank:
            raise ValueError("only local tensor views are exposed")
        elements = math.prod(shape)
        itemsize = 2 if dtype == torch.bfloat16 else 8
        start = storage_offset * itemsize
        if start < 0 or start + elements * itemsize > self.allocation.size:
            raise ValueError("IPC buffer view exceeds allocation")
        return _DevicePointer(self.allocation, start, shape, dtype)

    def close(self):
        for key in self.remote:
            pointer, references = _ipc_mappings[key]
            if references == 1:
                self._check(self.lib.hipIpcCloseMemHandle(pointer))
                del _ipc_mappings[key]
            else:
                _ipc_mappings[key] = (pointer, references - 1)
        self.remote.clear()
        self.allocation = None


class _Workspace:
    """A fixed-capacity generation; graph references survive growth."""

    def __init__(self, x, capacity=None):
        self.group = x.group
        self.device = x.hidden.device
        self.rank, self.world = dist.get_rank(x.group), dist.get_world_size(x.group)
        self.capacity = capacity or x.positions.shape[1]
        self.data_words = self.capacity * x.config.hidden_size // 4
        size = self.data_words * 8 + self.world * 8
        # Fail collectively before any kernel if ranks chose different contracts.
        signature = (self.capacity, x.config)
        signatures = [None] * self.world
        dist.all_gather_object(signatures, signature, group=x.group)
        if any(s != signature for s in signatures):
            raise ValueError("all ranks must use matching shapes and configuration")
        self.allocation = _HipAllocation(size, self.device)
        self.mapping = _HipMapping(self.allocation, x.group)
        self.buffer = self.mapping.get_buffer(
            self.rank, (self.capacity, x.config.hidden_size), torch.bfloat16
        )
        self.flags = self.mapping.get_buffer(
            self.rank, (self.world,), torch.int64, storage_offset=self.data_words
        )
        self.epoch = torch.empty((), device=self.device, dtype=torch.int64)
        self.peers = torch.empty(self.world, device=self.device, dtype=torch.uint64)
        _init_metadata[(1,)](
            self.flags,
            self.epoch,
            self.peers,
            tuple(self.mapping.buffer_ptrs),
            self.world,
            num_warps=1,
        )
        torch.cuda.synchronize(self.device)
        dist.barrier(group=x.group)
        self.local_buffer = torch.empty(
            (self.capacity, x.config.hidden_size),
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.captured = False

    def __call__(self, x):
        c, tokens = x.config, x.positions.shape[1]
        self.captured |= torch.cuda.is_current_stream_capturing()
        q = torch.empty(
            (tokens, c.q_size), dtype=x.hidden.dtype, device=x.hidden.device
        )
        k = torch.empty(
            (tokens, c.kv_size), dtype=x.hidden.dtype, device=x.hidden.device
        )
        v, gate = torch.empty_like(k), torch.empty_like(q)
        # The first barrier prevents fast ranks overwriting slow ranks' previous
        # GEMM inputs. The second publishes every push CTA before consumers run.
        for phase in range(2):
            _peer_barrier[(1,)](
                self.flags,
                self.epoch,
                self.peers,
                self.rank,
                DATA_WORDS=self.data_words,
                WORLD=self.world,
                num_warps=1,
            )
            if phase == 0:
                _push[(triton.cdiv(x.hidden.numel(), 2048), self.world)](
                    x.hidden,
                    self.peers,
                    self.rank,
                    N=x.hidden.numel(),
                    BLOCK=2048,
                    num_warps=4,
                )
        _cache_local[(triton.cdiv(tokens * c.hidden_size, 4096),)](
            self.buffer,
            self.local_buffer,
            N=tokens * c.hidden_size,
            BLOCK=4096,
            num_warps=4,
        )
        kernel = _project[
            (triton.cdiv(tokens, 128) * 2 * (c.num_q_heads + c.num_kv_heads),)
        ](
            self.local_buffer,
            x.weight,
            q,
            k,
            v,
            gate,
            x.q_norm,
            x.k_norm,
            x.positions,
            x.rope,
            x.cache,
            x.slots,
            *x.positions.stride(),
            x.rope.stride(0),
            x.slots.stride(0),
            *x.cache.stride(),
            M=tokens,
            HIDDEN=c.hidden_size,
            Q_HEADS=c.num_q_heads,
            KV_HEADS=c.num_kv_heads,
            CACHE_BLOCK_SIZE=x.cache.shape[2],
            SECTION_H=c.mrope_section[1],
            SECTION_W=c.mrope_section[2],
            EPS=c.eps,
            BM=128,
            BK=64,
            GROUP_M=8,
            num_warps=8,
            num_stages=2,
            enable_fp_fusion=False,
        )
        self.resources = dict(
            registers=kernel.n_regs,
            spills=kernel.n_spills,
            shared_bytes=kernel.metadata.shared,
        )
        return q, k, v, gate.view(tokens, c.num_q_heads, 256)

    def close(self):
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.group)
            self.mapping.close()
            dist.barrier(group=self.group)
            self.allocation.close()
            self.local_buffer = None
            self.buffer = self.flags = self.mapping = self.allocation = None


_windows = {}
_retired = []


def _check(condition, message, unsupported=False):
    if not condition:
        raise (NotImplementedError if unsupported else ValueError)(message)


def megapreattention(
    hidden,
    weight,
    positions,
    q_norm,
    k_norm,
    cos_sin_cache,
    kv_cache,
    slot_mapping,
    group,
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    mrope_section,
    eps=1e-6,
):
    """BF16 Hygon AllGather, gated QKV, Gemma QK norm, MRoPE and cache write.

    Signature and rounding boundaries follow NV PR #1007. Global head counts
    are sharded/replicated by TP2/4. hidden is contiguous [T/TP,H]; packed
    weight is [2*(localQ+localKV)*256,H], Q/gate interleaved within each head,
    followed by all K then all V. Positions [3,T] and slots [T] are int64.
    Norm weights [256] and RoPE [P,64] are BF16. Cache [blocks,localKV,B,512]
    accepts nonoverlapping strides. Negative slots skip cache writes; valid
    slots must be unique/in bounds and positions must index RoPE. These value
    preconditions are not scanned on the host. Inputs must not alias cache.

    Returns Q[T,localQ*256], K/V[T,localKV*256], gate[T,localQ,256]; gate is
    before activation. Only cache is mutated. Forward only; no autograd.
    Warm up each specialization/capacity before capture. Calls sharing a
    workspace must execute serially on one stream with identical rank order.
    Reset dependent graphs before collective clear_megapreattention_cache().
    No per-rank online tuning of the collective protocol is permitted.
    """
    _check(
        hidden.ndim == 2 and hidden.device.type == "cuda",
        "hidden must be a CUDA/HIP matrix",
    )
    _check(torch.version.hip is not None, "Hygon HIP backend required", True)
    tp = dist.get_world_size(group)
    local_tokens, hidden_size = hidden.shape
    _check(tp in (2, 4), "only TP2/4 supported", True)
    _check(
        local_tokens > 0 and local_tokens % 128 == 0,
        "per-rank tokens must be a positive multiple of 128",
        True,
    )
    _check(
        hidden_size >= 256 and hidden_size % 64 == 0,
        "hidden width must be >=256 and divisible by 64",
        True,
    )
    _check(head_dim == 256 and rotary_dim == 64, "only D256/R64 supported", True)
    _check(num_heads > 0 and num_heads % tp == 0, "Q heads must divide TP")
    _check(
        num_kv_heads > 0
        and (num_kv_heads % tp == 0 if num_kv_heads >= tp else tp % num_kv_heads == 0),
        "KV heads must shard or replicate evenly",
    )
    sections = tuple(mrope_section)
    _check(
        len(sections) == 3
        and all(isinstance(v, int) and v >= 0 for v in sections)
        and sum(sections) == 32,
        "MRoPE sections must be three nonnegative integers summing to 32",
    )
    _check(math.isfinite(eps) and eps >= 0, "eps must be finite and nonnegative")
    tokens = local_tokens * tp
    c = _Config(
        num_heads // tp,
        max(1, num_kv_heads // tp),
        head_dim,
        rotary_dim,
        sections,
        eps,
        hidden_size,
    )
    device = hidden.device
    _check(
        hidden.is_contiguous() and weight.is_contiguous(),
        "hidden and weight must be contiguous",
        True,
    )
    _check(weight.shape == (c.packed_size, hidden_size), "packed weight shape mismatch")
    _check(
        positions.shape == (3, tokens) and slot_mapping.shape == (tokens,),
        "positions/slots shape mismatch",
    )
    _check(
        positions.dtype == slot_mapping.dtype == torch.int64,
        "positions/slots must be int64",
        True,
    )
    _check(
        positions.device == slot_mapping.device == device,
        "all tensors must share device",
    )
    _check(
        all(v.shape == (256,) and v.is_contiguous() for v in (q_norm, k_norm)),
        "norm weights must be contiguous [256]",
    )
    _check(
        cos_sin_cache.ndim == 2
        and cos_sin_cache.shape[0] > 0
        and cos_sin_cache.shape[1] == 64
        and cos_sin_cache.stride(1) == 1,
        "RoPE must be [positions,64]",
    )
    _check(
        kv_cache.ndim == 4
        and kv_cache.shape[0] > 0
        and kv_cache.shape[1] == c.num_kv_heads
        and kv_cache.shape[2] > 0
        and kv_cache.shape[3] == 512,
        "cache shape mismatch",
    )
    # Sufficient metadata-only nonoverlap check, including permuted dense caches.
    span = 1
    for stride, size in sorted(zip(kv_cache.stride(), kv_cache.shape)):
        if size > 1:
            _check(stride >= span, "overlapping cache strides unsupported", True)
            span = stride * (size - 1) + span
    values = (hidden, weight, q_norm, k_norm, cos_sin_cache, kv_cache)
    _check(all(v.dtype == torch.bfloat16 for v in values), "only BF16 supported", True)
    _check(all(v.device == device for v in values), "all tensors must share device")
    _check(not any(v.requires_grad for v in values), "inference only", True)
    cache_storage = kv_cache.untyped_storage().data_ptr()
    _check(
        not any(
            v.untyped_storage().data_ptr() == cache_storage
            for v in (
                hidden,
                weight,
                q_norm,
                k_norm,
                cos_sin_cache,
                positions,
                slot_mapping,
            )
        ),
        "cache must not alias inputs",
        True,
    )
    inputs = _Inputs(
        hidden,
        weight,
        positions,
        q_norm,
        k_norm,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        group,
        c,
    )
    key = (group, device, c)
    with torch.cuda.device(device):
        current = _windows.get(key)
        if current is None or tokens > current.capacity:
            _check(
                not torch.cuda.is_current_stream_capturing(),
                "warm up workspace before Graph capture",
            )
            capacity = tokens if current is None else max(tokens, 2 * current.capacity)
            replacement = _Workspace(inputs, capacity)
            if current is not None:
                if current.captured:
                    _retired.append(current)
                else:
                    current.close()
            _windows[key] = current = replacement
        return current(inputs)


def clear_megapreattention_cache():
    """Reset dependent graphs, then call collectively before destroying group."""
    for window in [*_retired, *_windows.values()]:
        window.close()
    _retired.clear()
    _windows.clear()
