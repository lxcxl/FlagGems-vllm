# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Capture the unchanged candidate inside its caller-owned workspace lifetime."""

import argparse
import ctypes
import json
import sys
from pathlib import Path

import torch

import benchmark.test_megamoe as eager

PROTOCOL = "pr774-full-ep-both-cudagraph-5-mean-v2"
p = argparse.ArgumentParser(add_help=False)
p.add_argument("--graph-audit-dir", type=Path, required=True)
p.add_argument("--graph-profile", action="store_true")
extra, rest = p.parse_known_args()
sys.argv = [sys.argv[0], *rest]
extra.graph_audit_dir.mkdir(parents=True, exist_ok=True)


def measure(run, lib, rank, tokens, warmup, iters, repeats):
    values = dict(
        zip(run.__code__.co_freevars, (x.cell_contents for x in run.__closure__))
    )
    inputs = tuple(values[k] for k in ("x", "xs", "weights", "routes"))
    out, ws = values["out"], values["ws"]
    device = out.device
    audit = dict(
        rank=rank,
        world=ws.world_size,
        tokens=tokens,
        checks=[],
        capacity=ws.capacity,
        workers=ws.workers,
        block_m=ws.bm,
        block_n=ws.bn,
        block_k=ws.bk,
        timing_protocol=PROTOCOL,
    )
    lib.mxshmemx_barrier_wrapper.argtypes = [ctypes.c_void_p]
    lib.mxshmemx_barrier_wrapper.restype = None

    def sync():
        torch.cuda.synchronize(device)
        assert lib.mxshmem_all_success_wrapper(1)

    def barrier():
        lib.mxshmemx_barrier_wrapper(
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        )

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(max(warmup, 3)):
            run()
    torch.cuda.current_stream(device).wait_stream(stream)
    sync()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run()
    assert captured.data_ptr() == out.data_ptr()
    sync()
    snapshots = [x.clone() for x in inputs]
    original = None

    def validate(phase):
        graph.replay()
        actual = out.clone()
        sync()
        expected = run().clone()
        sync()
        mismatch = int((actual != expected).sum())
        finite = bool(torch.isfinite(actual).all())
        changed = int((expected != original).sum()) if original is not None else 0
        check = dict(
            phase=phase,
            bit_mismatches=mismatch,
            finite=finite,
            changed_from_original=changed,
        )
        audit["checks"].append(check)
        assert lib.mxshmem_all_success_wrapper(int(mismatch == 0 and finite)), check
        return expected

    original = validate("original_pre")
    if rank == 0:
        values["x"].neg_()
        values["xs"].mul_(1.125)
    validate("remote_input_changed")
    for x, saved in zip(inputs, snapshots):
        x.copy_(saved)
    if rank == 0:
        routes = values["routes"]
        ids = (torch.arange(routes.shape[1], device=device) + ws.model_shape[0]) % (
            ws.model_shape[0] * ws.world_size
        )
        routes.copy_(ids.expand_as(routes))
        values["weights"].mul_(0.5)
    validate("remote_routes_changed")
    for x, saved in zip(inputs, snapshots):
        x.copy_(saved)
    validate("original_restored_pre")
    if rank == 0:
        assert all(
            c["changed_from_original"] > 0
            for c in audit["checks"]
            if c["phase"].startswith("remote_")
        )
    sync()
    if extra.graph_profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            graph.replay()
            torch.cuda.synchronize(device)
        prof.export_chrome_trace(
            str(
                extra.graph_audit_dir
                / f"replay-np{ws.world_size}-t{tokens}-rank{rank}.json"
            )
        )
        audit["profile_kernel_names"] = [
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        ]
    means = []
    for repeat in range(repeats):
        for _ in range(warmup):
            graph.replay()
        sync()
        times = []
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        for _ in range(iters):
            barrier()
            start.record()
            graph.replay()
            end.record()
            barrier()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
            sync()
        mean = sum(times) / len(times)
        means.append(mean)
        print(
            f"BENCH_REPEAT rank={rank} tokens={tokens} repeat={repeat} mean_us={mean:.9f}",
            flush=True,
        )
    # Inspect measured output before any eager refresh.
    mismatch = int((out != original).sum())
    finite = bool(torch.isfinite(out).all())
    audit["checks"].append(
        dict(phase="post_measure", bit_mismatches=mismatch, finite=finite)
    )
    assert lib.mxshmem_all_success_wrapper(int(mismatch == 0 and finite))
    audit["capture_complete"] = True
    path = extra.graph_audit_dir / f"np{ws.world_size}-rank{rank}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(audit) + "\n")
    print("GRAPH_VALIDATION " + json.dumps(audit), flush=True)
    sync()
    graph.reset()
    return means


def emit(*args, **kwargs):
    if len(args) == 1 and isinstance(args[0], str) and args[0].startswith("{"):
        record = json.loads(args[0])
        if record.get("kind") == "benchmark":
            record["timing"] = (
                "CUDA events around full graph.replay; individual sample completion"
            )
            record["execution_mode"] = "cuda_graph"
            args = (json.dumps(record),)
    print(*args, **kwargs)


eager.print = emit
eager.measure = measure
eager.TIMING_PROTOCOL = PROTOCOL
eager._run_worker()
