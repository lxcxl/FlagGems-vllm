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

"""W8A8 distributed MoE with runtime routing and one resident-grid launch.

Weights use the usual [expert, output, input] layout, with gate then up in w1.
The caller owns the EP group and symmetric allocator. A workspace is reused
serially on one stream by every rank; it never reads routing data on the host.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ._sync import fence_sys, native_rcp


@triton.jit
def _peer(table, rank, dtype: tl.constexpr):
    return tl.load(table + rank).to(tl.pointer_type(dtype))


@triton.jit
def _acquire(pointer):
    return tl.atomic_cas(pointer, -1, -1, sem="acquire")


@triton.jit(noinline=True)
def _barrier(state, slot: tl.constexpr, WORKERS: tl.constexpr, rounds=1):
    tl.debug_barrier()
    tl.atomic_add(state + 8 + slot, 1, sem="acq_rel")
    while _acquire(state + 8 + slot) < WORKERS * rounds:
        pass
    tl.debug_barrier()


@triton.jit(noinline=True)
def _rank_barrier(
    signals, slot: tl.constexpr, epoch, RANK: tl.constexpr, WORLD: tl.constexpr
):
    fence_sys()
    tl.debug_barrier()
    for peer in tl.static_range(WORLD):
        tl.store(
            _peer(signals, peer, tl.int32) + slot * WORLD + RANK,
            epoch,
            cache_modifier=".wt",
        )
    fence_sys()
    for source in tl.static_range(WORLD):
        flag = _peer(signals, RANK, tl.int32) + slot * WORLD + source
        while tl.load(flag, volatile=True) != epoch:
            pass
    tl.debug_barrier()
    fence_sys()


@triton.jit
def _input_pointer(
    inputs,
    source,
    RANK: tl.constexpr,
    STRIDE: tl.constexpr,
    OFFSET: tl.constexpr,
    DTYPE: tl.constexpr,
):
    return (_peer(inputs, RANK, tl.int8) + source * STRIDE + OFFSET).to(
        tl.pointer_type(DTYPE)
    )


@triton.jit(noinline=True)
def _push(
    x,
    x_scale,
    route,
    weight,
    inputs,
    lut,
    tokens,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    SS: tl.constexpr,
    RS0: tl.constexpr,
    RS1: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    WORKERS: tl.constexpr,
    STRIDE: tl.constexpr,
    XOFF: tl.constexpr,
    SOFF: tl.constexpr,
    ROFF: tl.constexpr,
    WOFF: tl.constexpr,
    QUANTIZED: tl.constexpr,
    NATIVE: tl.constexpr,
    HPAD: tl.constexpr,
    TPAD: tl.constexpr,
    worker,
):
    cols = tl.arange(0, HPAD)
    slots = tl.arange(0, TPAD)
    for row in tl.range(worker, tokens, WORKERS):
        value = tl.load(x + row * XS0 + cols * XS1, mask=cols < H, other=0)
        if QUANTIZED:
            scale = tl.load(x_scale + row * SS)
            q = value
        else:
            amax = tl.max(tl.abs(value.to(tl.float32)))
            if NATIVE:
                bits = amax.to(tl.int32, bitcast=True) >> 16
                scale = tl.load(lut + bits, mask=bits < 32640, other=0).to(
                    tl.float32, bitcast=True
                )
                inv = native_rcp(amax) * 127.0
            else:
                scale = tl.div_rn(amax, 127.0)
                inv = tl.div_rn(127.0, amax)
            scaled = tl.where(amax == 0, 0.0, value.to(tl.float32) * inv)
            q = libdevice.nearbyint(scaled).to(tl.int8)
        ids = tl.load(route + row * RS0 + slots * RS1, mask=slots < TOPK, other=-1)
        weights = tl.load(weight + row * WS0 + slots * WS1, mask=slots < TOPK, other=0)
        for peer in tl.static_range(WORLD):
            base = _peer(inputs, peer, tl.int8) + RANK * STRIDE
            tl.store(base + XOFF + row * H + cols, q, mask=cols < H)
            tl.store((base + SOFF).to(tl.pointer_type(tl.float32)) + row, scale)
            tl.store(
                (base + ROFF).to(tl.pointer_type(tl.int32)) + row * TOPK + slots,
                ids.to(tl.int32),
                mask=slots < TOPK,
            )
            tl.store(
                (base + WOFF).to(tl.pointer_type(tl.float32)) + row * TOPK + slots,
                weights.to(tl.float32),
                mask=slots < TOPK,
            )
    if worker == 0:
        for peer in tl.static_range(WORLD):
            base = _peer(inputs, peer, tl.int8) + RANK * STRIDE
            tl.store(base.to(tl.pointer_type(tl.int32)), tokens)
    fence_sys()


@triton.jit(noinline=True)
def _histogram(
    inputs,
    expert_map,
    chunk_counts,
    RANK: tl.constexpr,
    E: tl.constexpr,
    EPAD: tl.constexpr,
    CHUNKS: tl.constexpr,
    CHUNK: tl.constexpr,
    TOPK: tl.constexpr,
    STRIDE: tl.constexpr,
    ROFF: tl.constexpr,
    MAPPED: tl.constexpr,
    task,
):
    source = task // CHUNKS
    chunk = task % CHUNKS
    count = tl.load(_input_pointer(inputs, source, RANK, STRIDE, 0, tl.int32))
    rid = chunk * CHUNK + tl.arange(0, CHUNK)
    expert = tl.load(
        _input_pointer(inputs, source, RANK, STRIDE, ROFF, tl.int32) + rid,
        mask=rid < count * TOPK,
        other=-1,
    )
    if MAPPED:
        local = tl.load(expert_map + expert, mask=expert >= 0, other=-1)
    else:
        local = expert - RANK * E
    take = (rid < count * TOPK) & (local >= 0) & (local < E)
    es = tl.arange(0, EPAD)
    counts = tl.sum(((local[:, None] == es[None, :]) & take[:, None]).to(tl.int32), 0)
    tl.store(chunk_counts + (source * CHUNKS + chunk) * E + es, counts, mask=es < E)


@triton.jit(noinline=True)
def _prefix(
    chunk_counts,
    chunk_offsets,
    counts,
    E: tl.constexpr,
    WORLD: tl.constexpr,
    CHUNKS: tl.constexpr,
    CPAD: tl.constexpr,
    task,
):
    source, expert = task // E, task % E
    chunks = tl.arange(0, CPAD)
    values = tl.load(
        chunk_counts + (source * CHUNKS + chunks) * E + expert,
        mask=chunks < CHUNKS,
        other=0,
    )
    tl.store(
        chunk_offsets + (source * CHUNKS + chunks) * E + expert,
        tl.cumsum(values) - values,
        mask=chunks < CHUNKS,
    )
    tl.store(counts + expert * WORLD + source, tl.sum(values))


@triton.jit(noinline=True)
def _plan(
    counts,
    expert_base,
    state,
    E: tl.constexpr,
    EPAD: tl.constexpr,
    WORLD: tl.constexpr,
    BM: tl.constexpr,
):
    es = tl.arange(0, EPAD)
    rows = tl.full((EPAD,), 0, tl.int32)
    for source in tl.static_range(WORLD):
        rows += tl.load(counts + es * WORLD + source, mask=es < E, other=0)
    blocks = tl.cdiv(rows, BM)
    tl.store(expert_base + es, tl.cumsum(blocks) - blocks, mask=es < E)
    tl.store(state + 1, tl.sum(blocks))


@triton.jit(noinline=True)
def _block_metadata(
    plan,
    counts,
    expert_base,
    block,
    E: tl.constexpr,
    EPAD: tl.constexpr,
    WORLD: tl.constexpr,
    BM: tl.constexpr,
):
    es = tl.arange(0, EPAD)
    starts = tl.load(expert_base + es, mask=es < E, other=2147483647)
    expert = tl.sum(((starts <= block) & (es < E)).to(tl.int32)) - 1
    first = tl.load(expert_base + expert)
    total = tl.full((), 0, tl.int32)
    for source in tl.static_range(WORLD):
        total += tl.load(counts + expert * WORLD + source)
    tl.store(plan + block * 2, expert)
    tl.store(plan + block * 2 + 1, tl.minimum(BM, total - (block - first) * BM))


@triton.jit(noinline=True)
def _payload(
    inputs,
    expert_map,
    counts,
    offsets,
    expert_base,
    row_meta,
    scales,
    weights,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    E: tl.constexpr,
    CHUNKS: tl.constexpr,
    CHUNK: tl.constexpr,
    TOPK: tl.constexpr,
    STRIDE: tl.constexpr,
    ROFF: tl.constexpr,
    SOFF: tl.constexpr,
    WOFF: tl.constexpr,
    BM: tl.constexpr,
    MAPPED: tl.constexpr,
    task,
):
    source, chunk = task // CHUNKS, task % CHUNKS
    lanes = tl.arange(0, CHUNK)
    rid = chunk * CHUNK + lanes
    count = tl.load(_input_pointer(inputs, source, RANK, STRIDE, 0, tl.int32))
    expert = tl.load(
        _input_pointer(inputs, source, RANK, STRIDE, ROFF, tl.int32) + rid,
        mask=rid < count * TOPK,
        other=-1,
    )
    if MAPPED:
        local = tl.load(expert_map + expert, mask=expert >= 0, other=-1)
    else:
        local = expert - RANK * E
    take = (rid < count * TOPK) & (local >= 0) & (local < E)
    intra = tl.sum(
        (
            (lanes[:, None] > lanes[None, :])
            & (expert[:, None] == expert[None, :])
            & take[None, :]
        ).to(tl.int32),
        1,
    )
    offset = (
        tl.load(offsets + (source * CHUNKS + chunk) * E + local, mask=take, other=0)
        + intra
    )
    prior = tl.full((CHUNK,), 0, tl.int32)
    for previous in tl.static_range(WORLD):
        value = tl.load(counts + local * WORLD + previous, mask=take, other=0)
        prior += tl.where(previous < source, value, 0)
    row = tl.load(expert_base + local, mask=take, other=0) * BM + prior + offset
    tl.store(row_meta + row * 3, source, mask=take)
    tl.store(row_meta + row * 3 + 1, rid // TOPK, mask=take)
    tl.store(row_meta + row * 3 + 2, rid % TOPK, mask=take)
    scale = tl.load(
        _input_pointer(inputs, source, RANK, STRIDE, SOFF, tl.float32) + rid // TOPK,
        mask=take,
        other=0,
    )
    weight = tl.load(
        _input_pointer(inputs, source, RANK, STRIDE, WOFF, tl.float32) + rid,
        mask=take,
        other=0,
    )
    tl.store(scales + row, scale, mask=take)
    tl.store(weights + row, weight, mask=take)


@triton.jit(noinline=True)
def _copy_tokens(
    inputs,
    row_meta,
    plan,
    tokens,
    task,
    RANK: tl.constexpr,
    STRIDE: tl.constexpr,
    XOFF: tl.constexpr,
    H: tl.constexpr,
    HPAD: tl.constexpr,
    BM: tl.constexpr,
    ROWS: tl.constexpr,
):
    block, tile = task // (BM // ROWS), task % (BM // ROWS)
    rows = tile * ROWS + tl.arange(0, ROWS)
    valid = rows < tl.load(plan + block * 2 + 1)
    pool_row = block * BM + rows
    source = tl.load(row_meta + pool_row * 3, mask=valid, other=0)
    token = tl.load(row_meta + pool_row * 3 + 1, mask=valid, other=0)
    cols = tl.arange(0, HPAD)
    address = _input_pointer(inputs, source, RANK, STRIDE, XOFF, tl.int8)
    value = tl.load(
        address[:, None] + token[:, None] * H + cols[None, :],
        mask=valid[:, None] & (cols[None, :] < H),
        other=0,
    )
    tl.store(
        tokens + pool_row[:, None] * H + cols[None, :],
        value,
        mask=valid[:, None] & (cols[None, :] < H),
    )


@triton.jit(noinline=True)
def _gemm1(
    plan,
    tokens,
    scales,
    w1,
    w1_scale,
    activation,
    H: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    STAGES: tl.constexpr,
    WE: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    task,
):
    tiles: tl.constexpr = triton.cdiv(I, BN // 2)
    block, tile = task // tiles, task % tiles
    expert = tl.load(plan + block * 2)
    valid = tl.load(plan + block * 2 + 1)
    rows = block * BM + tl.arange(0, BM)
    rm = tl.arange(0, BM) < valid
    pairs = tl.arange(0, BN)
    out_col = tile * (BN // 2) + (pairs // 16) * 8 + pairs % 8
    col = out_col + tl.where(pairs % 16 >= 8, I, 0)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.int32)
    for kb in tl.range(triton.cdiv(H, BK), num_stages=STAGES):
        k = kb * BK + kk
        a = tl.load(
            tokens + rows[:, None] * H + k[None, :],
            mask=rm[:, None] & (k[None, :] < H),
            other=0,
        )
        b = tl.load(
            w1 + expert * WE + k[:, None] * WK + col[None, :] * WN,
            mask=(k[:, None] < H) & (out_col[None, :] < I),
            other=0,
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    sf = tl.load(scales + rows, mask=rm, other=0)
    ws = tl.load(w1_scale + expert * SE + col * SN, mask=out_col < I, other=0)
    value = (
        (acc.to(tl.float32) * sf[:, None] * ws[None, :]).to(tl.bfloat16).to(tl.float32)
    )
    paired = tl.permute(tl.reshape(value, (BM, BN // 16, 2, 8)), (0, 1, 3, 2))
    gate, up = tl.split(paired)
    gate, up = tl.reshape(gate, (BM, BN // 2)), tl.reshape(up, (BM, BN // 2))
    silu = tl.div_rn(gate, 1.0 + tl.exp(-gate)).to(tl.bfloat16).to(tl.float32)
    cols = tile * (BN // 2) + tl.arange(0, BN // 2)
    tl.store(
        activation + rows[:, None] * I + cols[None, :],
        (silu * up).to(tl.bfloat16),
        mask=rm[:, None] & (cols[None, :] < I),
    )


@triton.jit(noinline=True)
def _quantize(
    plan,
    activation,
    quantized,
    scales,
    lut,
    task,
    BM: tl.constexpr,
    I: tl.constexpr,
    IPAD: tl.constexpr,
    NATIVE: tl.constexpr,
):
    rows = task * 8 + tl.arange(0, 8)
    valid = rows % BM < tl.load(plan + (rows // BM) * 2 + 1)
    cols = tl.arange(0, IPAD)
    x = tl.load(
        activation + rows[:, None] * I + cols[None, :],
        mask=valid[:, None] & (cols[None, :] < I),
        other=0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(x), 1)
    if NATIVE:
        bits = amax.to(tl.int32, bitcast=True) >> 16
        scale = tl.load(lut + bits, mask=bits < 32640, other=0).to(
            tl.float32, bitcast=True
        )
        inv = native_rcp(amax) * 127.0
    else:
        scale = tl.div_rn(amax, 127.0)
        inv = tl.div_rn(127.0, amax)
    value = tl.where(amax[:, None] == 0, 0.0, x * inv[:, None])
    q = libdevice.nearbyint(value).to(tl.int8)
    tl.store(
        quantized + rows[:, None] * I + cols[None, :],
        q,
        mask=valid[:, None] & (cols[None, :] < I),
    )
    tl.store(scales + rows, scale, mask=valid)


@triton.jit(noinline=True)
def _gemm2(
    plan,
    weights,
    acts,
    scales,
    w2,
    w2_scale,
    row_meta,
    partials,
    task,
    base_row,
    H: tl.constexpr,
    I: tl.constexpr,
    CAPACITY: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    STAGES: tl.constexpr,
    WE: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    LOCAL_OUTPUT: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(H, BN)
    block, tile = task // tiles, task % tiles
    expert = tl.load(plan + block * 2)
    valid = tl.load(plan + block * 2 + 1)
    rows = block * BM + tl.arange(0, BM)
    rm = tl.arange(0, BM) < valid
    cols = tile * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.int32)
    for kb in tl.range(triton.cdiv(I, BK), num_stages=STAGES):
        k = kb * BK + kk
        a = tl.load(
            acts + rows[:, None] * I + k[None, :],
            mask=rm[:, None] & (k[None, :] < I),
            other=0,
        )
        b = tl.load(
            w2 + expert * WE + k[:, None] * WK + cols[None, :] * WN,
            mask=(k[:, None] < I) & (cols[None, :] < H),
            other=0,
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    sf = tl.load(scales + rows, mask=rm, other=0)
    ws = tl.load(w2_scale + expert * SE + cols * SN, mask=cols < H, other=0)
    router = tl.load(weights + rows, mask=rm, other=0)
    value = acc.to(tl.float32) * sf[:, None] * ws[None, :] * router[:, None]
    if LOCAL_OUTPUT:
        tl.store(
            partials + (rows[:, None] - base_row) * H + cols[None, :],
            value.to(tl.bfloat16),
            mask=rm[:, None] & (cols[None, :] < H),
        )
    else:
        source = tl.load(row_meta + rows * 3, mask=rm, other=0)
        token = tl.load(row_meta + rows * 3 + 1, mask=rm, other=0)
        slot = tl.load(row_meta + rows * 3 + 2, mask=rm, other=0)
        output = _peer(partials, source, tl.bfloat16)
        tl.store(
            output[:, None]
            + (slot[:, None] * CAPACITY + token[:, None]) * H
            + cols[None, :],
            value.to(tl.bfloat16),
            mask=rm[:, None] & (cols[None, :] < H),
        )


@triton.jit(noinline=True)
def _scatter_partials(
    plan,
    metadata,
    local,
    partials,
    task,
    base_row,
    CAPACITY: tl.constexpr,
    H: tl.constexpr,
    HPAD: tl.constexpr,
    BM: tl.constexpr,
    ROWS: tl.constexpr,
):
    block, tile = task // (BM // ROWS), task % (BM // ROWS)
    rr = tile * ROWS + tl.arange(0, ROWS)
    valid = rr < tl.load(plan + block * 2 + 1)
    row = block * BM + rr
    source = tl.load(metadata + row * 3, mask=valid, other=0)
    token = tl.load(metadata + row * 3 + 1, mask=valid, other=0)
    slot = tl.load(metadata + row * 3 + 2, mask=valid, other=0)
    col = tl.arange(0, HPAD)
    value = tl.load(
        local + (row[:, None] - base_row) * H + col[None, :],
        mask=valid[:, None] & (col[None, :] < H),
        other=0,
    )
    output = _peer(partials, source, tl.bfloat16)
    tl.store(
        output[:, None]
        + (slot[:, None] * CAPACITY + token[:, None]) * H
        + col[None, :],
        value,
        mask=valid[:, None] & (col[None, :] < H),
    )


@triton.jit(do_not_specialize=["tokens"])
def _megamoe(
    x,
    x_scale,
    w1,
    w2,
    w1_scale,
    w2_scale,
    routes,
    weights,
    expert_map,
    out,
    inputs,
    partials,
    signals,
    state,
    epoch_counter,
    chunk_counts,
    chunk_offsets,
    counts,
    expert_base,
    plan,
    row_meta,
    pool,
    pool_scale,
    pool_weight,
    activation,
    quantized,
    act_scale,
    expert_partials,
    lut,
    tokens,
    H: tl.constexpr,
    I: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    CAPACITY: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    WORKERS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    STAGES: tl.constexpr,
    COPY_ROWS: tl.constexpr,
    STRIDE: tl.constexpr,
    XOFF: tl.constexpr,
    SOFF: tl.constexpr,
    ROFF: tl.constexpr,
    WOFF: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    SS: tl.constexpr,
    RS0: tl.constexpr,
    RS1: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    W1E: tl.constexpr,
    W1N: tl.constexpr,
    W1K: tl.constexpr,
    W2E: tl.constexpr,
    W2N: tl.constexpr,
    W2K: tl.constexpr,
    S1E: tl.constexpr,
    S1N: tl.constexpr,
    S2E: tl.constexpr,
    S2N: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    QUANTIZED: tl.constexpr,
    NATIVE: tl.constexpr,
    MAPPED: tl.constexpr,
    LOCAL_OUTPUT: tl.constexpr,
):
    worker = tl.program_id(0)
    epoch = tl.load(epoch_counter) + 1
    if worker == 0:
        idx = tl.arange(0, 32)
        tl.store(state + idx, 0, mask=(idx >= 8) & (idx < (28 if LOCAL_OUTPUT else 24)))
        tl.debug_barrier()
        tl.atomic_xchg(state, epoch, sem="release")
    while _acquire(state) != epoch:
        pass
    tl.debug_barrier()
    _push(
        x,
        x_scale,
        routes,
        weights,
        inputs,
        lut,
        tokens,
        H,
        TOPK,
        XS0,
        XS1,
        SS,
        RS0,
        RS1,
        WS0,
        WS1,
        RANK,
        WORLD,
        WORKERS,
        STRIDE,
        XOFF,
        SOFF,
        ROFF,
        WOFF,
        QUANTIZED,
        NATIVE,
        triton.next_power_of_2(H),
        triton.next_power_of_2(TOPK),
        worker,
    )
    _barrier(state, 0, WORKERS)
    if worker == 0:
        _rank_barrier(signals, 0, epoch, RANK, WORLD)
        tl.atomic_xchg(state + 2, epoch, sem="release")
    while _acquire(state + 2) != epoch:
        pass
    tl.debug_barrier()
    CHUNK: tl.constexpr = 64
    CHUNKS: tl.constexpr = triton.cdiv(CAPACITY * TOPK, CHUNK)
    EPAD: tl.constexpr = triton.next_power_of_2(E)
    for task in tl.range(worker, WORLD * CHUNKS, WORKERS):
        _histogram(
            inputs,
            expert_map,
            chunk_counts,
            RANK,
            E,
            EPAD,
            CHUNKS,
            CHUNK,
            TOPK,
            STRIDE,
            ROFF,
            MAPPED,
            task,
        )
    _barrier(state, 1, WORKERS)
    for task in tl.range(worker, WORLD * E, WORKERS):
        _prefix(
            chunk_counts,
            chunk_offsets,
            counts,
            E,
            WORLD,
            CHUNKS,
            triton.next_power_of_2(CHUNKS),
            task,
        )
    _barrier(state, 2, WORKERS)
    if worker == 0:
        _plan(counts, expert_base, state, E, EPAD, WORLD, BM)
    _barrier(state, 3, WORKERS)
    blocks = tl.load(state + 1)
    for block in tl.range(worker, blocks, WORKERS):
        _block_metadata(plan, counts, expert_base, block, E, EPAD, WORLD, BM)
    for task in tl.range(worker, WORLD * CHUNKS, WORKERS):
        _payload(
            inputs,
            expert_map,
            counts,
            chunk_offsets,
            expert_base,
            row_meta,
            pool_scale,
            pool_weight,
            RANK,
            WORLD,
            E,
            CHUNKS,
            CHUNK,
            TOPK,
            STRIDE,
            ROFF,
            SOFF,
            WOFF,
            BM,
            MAPPED,
            task,
        )
    _barrier(state, 4, WORKERS)
    for task in tl.range(worker, blocks * (BM // COPY_ROWS), WORKERS):
        _copy_tokens(
            inputs,
            row_meta,
            plan,
            pool,
            task,
            RANK,
            STRIDE,
            XOFF,
            H,
            triton.next_power_of_2(H),
            BM,
            COPY_ROWS,
        )
    _barrier(state, 5, WORKERS)
    for task in tl.range(worker, blocks * triton.cdiv(I, BN // 2), WORKERS):
        _gemm1(
            plan,
            pool,
            pool_scale,
            w1,
            w1_scale,
            activation,
            H,
            I,
            BM,
            BN,
            BK,
            STAGES,
            W1E,
            W1N,
            W1K,
            S1E,
            S1N,
            task,
        )
    _barrier(state, 6, WORKERS)
    for task in tl.range(worker, blocks * (BM // 8), WORKERS):
        _quantize(
            plan,
            activation,
            quantized,
            act_scale,
            lut,
            task,
            BM,
            I,
            triton.next_power_of_2(I),
            NATIVE,
        )
    _barrier(state, 7, WORKERS)
    if LOCAL_OUTPUT and blocks > 2 * WORKERS:
        # Compute CTAs produce two bounded buffers; transport CTAs consume them.
        # Each slot has monotonic ready/free counts, reset once per invocation.
        COMM: tl.constexpr = max(1, WORKERS // 6)
        COMPUTE: tl.constexpr = WORKERS - COMM
        BATCH: tl.constexpr = 2 * WORKERS
        for first in tl.range(0, blocks, BATCH):
            batch = first // BATCH
            slot = batch % 2
            generation = batch // 2
            active = tl.minimum(BATCH, blocks - first)
            base_row = (first - slot * BATCH) * BM
            if worker < COMPUTE:
                while _acquire(state + 26 + slot) < COMM * generation:
                    pass
                tl.debug_barrier()
                for task in tl.range(worker, active * triton.cdiv(H, BN), COMPUTE):
                    _gemm2(
                        plan,
                        pool_weight,
                        quantized,
                        act_scale,
                        w2,
                        w2_scale,
                        row_meta,
                        expert_partials,
                        task + first * triton.cdiv(H, BN),
                        base_row,
                        H,
                        I,
                        CAPACITY,
                        BM,
                        BN,
                        BK,
                        STAGES,
                        W2E,
                        W2N,
                        W2K,
                        S2E,
                        S2N,
                        True,
                    )
                tl.debug_barrier()
                tl.atomic_add(state + 24 + slot, 1, sem="acq_rel")
            else:
                while _acquire(state + 24 + slot) < COMPUTE * (generation + 1):
                    pass
                tl.debug_barrier()
                for task in tl.range(worker - COMPUTE, active * (BM // 8), COMM):
                    _scatter_partials(
                        plan,
                        row_meta,
                        expert_partials,
                        partials,
                        task + first * (BM // 8),
                        base_row,
                        CAPACITY,
                        H,
                        triton.next_power_of_2(H),
                        BM,
                        8,
                    )
                tl.debug_barrier()
                tl.atomic_add(state + 26 + slot, 1, sem="acq_rel")
    else:
        SERIAL_BATCH: tl.constexpr = (
            4 * WORKERS
            if LOCAL_OUTPUT
            else triton.cdiv(CAPACITY * TOPK * WORLD, BM) + E
        )
        for first in tl.range(0, blocks, SERIAL_BATCH):
            active = tl.minimum(SERIAL_BATCH, blocks - first)
            for task in tl.range(worker, active * triton.cdiv(H, BN), WORKERS):
                _gemm2(
                    plan,
                    pool_weight,
                    quantized,
                    act_scale,
                    w2,
                    w2_scale,
                    row_meta,
                    expert_partials,
                    task + first * triton.cdiv(H, BN),
                    first * BM,
                    H,
                    I,
                    CAPACITY,
                    BM,
                    BN,
                    BK,
                    STAGES,
                    W2E,
                    W2N,
                    W2K,
                    S2E,
                    S2N,
                    LOCAL_OUTPUT,
                )
            if LOCAL_OUTPUT:
                rounds = first // SERIAL_BATCH + 1
                _barrier(state, 10, WORKERS, rounds)
                for task in tl.range(worker, active * (BM // 8), WORKERS):
                    _scatter_partials(
                        plan,
                        row_meta,
                        expert_partials,
                        partials,
                        task + first * (BM // 8),
                        first * BM,
                        CAPACITY,
                        H,
                        triton.next_power_of_2(H),
                        BM,
                        8,
                    )
                _barrier(state, 11, WORKERS, rounds)
    fence_sys()
    _barrier(state, 8, WORKERS)
    if worker == 0:
        _rank_barrier(signals, 1, epoch, RANK, WORLD)
        tl.atomic_xchg(state + 3, epoch, sem="release")
    while _acquire(state + 3) != epoch:
        pass
    tl.debug_barrier()
    cols = tl.arange(0, triton.next_power_of_2(H))
    for token in tl.range(worker, tokens, WORKERS):
        result = tl.full((triton.next_power_of_2(H),), 0, tl.float32)
        for slot in tl.static_range(TOPK):
            expert = tl.load(routes + token * RS0 + slot * RS1)
            value = tl.load(
                _peer(partials, RANK, tl.bfloat16)
                + (slot * CAPACITY + token) * H
                + cols,
                mask=(cols < H) & (expert >= 0),
                other=0,
            )
            result += value.to(tl.float32)
        tl.store(out + token * OS0 + cols * OS1, result.to(tl.bfloat16), mask=cols < H)
    _barrier(state, 9, WORKERS)
    if worker == 0:
        _rank_barrier(signals, 2, epoch, RANK, WORLD)
        tl.atomic_xchg(state + 4, epoch, sem="release")
    while _acquire(state + 4) != epoch:
        pass
    tl.debug_barrier()
    if worker == 0:
        tl.store(epoch_counter, epoch)


def _launch_args(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    w1_scale,
    w2_scale,
    *,
    workspace,
    a1_scale=None,
    expert_map=None,
    out=None,
):
    """Compute distributed SwiGLU MoE; communication is included in this call.

    hidden_states: [T,H] int8 with a1_scale, or BF16 (quantized in the kernel).
    w1/w2: int8 [E_local,2I,H] / [E_local,H,I]; scales [E_local,2I] / [E_local,H]
    (a trailing singleton dimension is also accepted). Routes/weights: [T,top_k].
    Expert IDs are global; -1 drops a route. By default experts are partitioned
    contiguously; expert_map optionally maps global IDs to local IDs, or -1.
    Output is BF16 [T,H]. Every rank participates, including ranks with T=0.
    """
    ws = workspace
    tokens, h = hidden_states.shape
    e, _, i = ws.model_shape
    if tokens > ws.capacity:
        raise ValueError("input exceeds the workspace token capacity")
    if topk_ids.shape != (tokens, ws.top_k) or topk_weights.shape != topk_ids.shape:
        raise ValueError("routes and weights must have shape [tokens, top_k]")
    if (
        hidden_states.dtype not in (torch.int8, torch.bfloat16)
        or w1.dtype != torch.int8
        or w2.dtype != torch.int8
    ):
        raise NotImplementedError("MegaMoE requires W8A8 with int8 or BF16 input")
    if hidden_states.dtype == torch.int8 and a1_scale is None:
        raise ValueError("INT8 input requires per-token a1_scale")
    if out is None:
        out = torch.empty(
            (tokens, h), dtype=torch.bfloat16, device=hidden_states.device
        )
    if h != ws.model_shape[1] or w1.shape != (e, 2 * i, h) or w2.shape != (e, h, i):
        raise ValueError("weights and input must match the workspace dimensions")
    if (
        out.shape != hidden_states.shape
        or out.dtype != torch.bfloat16
        or out.device != hidden_states.device
    ):
        raise ValueError("out must be BF16 with the input shape and device")
    xscale = a1_scale if a1_scale is not None else hidden_states
    lut = ws.lut if ws.lut is not None else ws.buffers[0]
    mapping = expert_map if expert_map is not None else topk_ids
    options = dict(
        H=h,
        I=i,
        E=e,
        TOPK=ws.top_k,
        CAPACITY=ws.capacity,
        RANK=ws.rank,
        WORLD=ws.world_size,
        WORKERS=ws.workers,
        BM=ws.bm,
        BN=ws.bn,
        BK=ws.bk,
        STAGES=ws.stages,
        COPY_ROWS=4,
        STRIDE=ws.layout[0],
        XOFF=ws.layout[1],
        SOFF=ws.layout[2],
        ROFF=ws.layout[3],
        WOFF=ws.layout[4],
        XS0=hidden_states.stride(0),
        XS1=hidden_states.stride(1),
        SS=xscale.stride(0),
        RS0=topk_ids.stride(0),
        RS1=topk_ids.stride(1),
        WS0=topk_weights.stride(0),
        WS1=topk_weights.stride(1),
        W1E=w1.stride(0),
        W1N=w1.stride(1),
        W1K=w1.stride(2),
        W2E=w2.stride(0),
        W2N=w2.stride(1),
        W2K=w2.stride(2),
        S1E=w1_scale.stride(0),
        S1N=w1_scale.stride(1),
        S2E=w2_scale.stride(0),
        S2N=w2_scale.stride(1),
        OS0=out.stride(0),
        OS1=out.stride(1),
        QUANTIZED=hidden_states.dtype == torch.int8,
        NATIVE=ws.lut is not None,
        MAPPED=expert_map is not None,
        LOCAL_OUTPUT=ws.local_output,
        num_warps=4,
        num_stages=ws.stages,
        launch_cooperative_grid=True,
        extern_libs=dict(ws.extern_libs),
    )
    args = (
        hidden_states,
        xscale,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_ids,
        topk_weights,
        mapping,
        out,
        ws.inputs,
        ws.partials,
        ws.signals,
        *ws.buffers,
        ws.expert_partials,
        lut,
        tokens,
    )
    return out, args, options


def launch_megamoe(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    w1_scale,
    w2_scale,
    *,
    workspace,
    a1_scale=None,
    expert_map=None,
    out=None,
):
    out, args, options = _launch_args(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        workspace=workspace,
        a1_scale=a1_scale,
        expert_map=expert_map,
        out=out,
    )
    with torch.cuda.device(hidden_states.device):
        _megamoe[(workspace.workers,)](*args, **options)
    return out
