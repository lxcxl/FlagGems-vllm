"""Full native EP CUDA Graph replay; imports the immutable eager baseline."""

import argparse
import faulthandler
import gc
import hashlib
import json
import signal
import sys
import weakref
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

from tools.megamoe import baseline as eager

faulthandler.register(signal.SIGUSR1, all_threads=True)

PROTOCOL = "pr774-full-ep-both-cudagraph-5-mean-v2"
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--graph-audit-dir", type=Path, required=True)
parser.add_argument("--graph-profile", action="store_true")
extra, passthrough = parser.parse_known_args()
sys.argv = [sys.argv[0], *passthrough]
extra.graph_audit_dir.mkdir(parents=True, exist_ok=True)
AUDITS = []
ARGS = None
RawAdapter = eager.PrequantizedInt8MoE
LIVE_ADAPTERS = weakref.WeakSet()
original_destroy = eager.destroy_model_parallel
original_parse = eager.parse_args


def parse_args():
    global ARGS
    ARGS = original_parse()
    return ARGS


class GraphAdapter(RawAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        LIVE_ADAPTERS.add(self)

    @torch.inference_mode()
    def __call__(self, *tensors):
        graph = getattr(self, "_captured_graph", None)
        if graph is None:
            return super().__call__(*tensors)
        assert all(
            a.data_ptr() == b.data_ptr() for a, b in zip(tensors, self._static_inputs)
        )
        graph.replay()
        return self.output


@contextmanager
def count_capture_collectives():
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    counts = Counter()
    originals = {}
    for name in ("all_gather", "all_gatherv", "reduce_scatter", "reduce_scatterv"):
        original = getattr(PyNcclCommunicator, name, None)
        if original is None:
            continue
        originals[name] = original

        def wrapped(self, *args, _name=name, _original=original, **kwargs):
            if torch.cuda.is_current_stream_capturing():
                counts[_name] += 1
            return _original(self, *args, **kwargs)

        setattr(PyNcclCommunicator, name, wrapped)
    try:
        yield counts
    finally:
        for name, original in originals.items():
            setattr(PyNcclCommunicator, name, original)


def digest(tensor):
    return hashlib.sha256(
        tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


@torch.inference_mode()
def graph_benchmark(
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
    from vllm.distributed.parallel_state import graph_capture

    rank, world = dist.get_rank(), dist.get_world_size()
    tensors = (hidden_q, hidden_scale, topk_weights, topk_ids)
    tokens = hidden_q.shape[0]
    audit = dict(
        rank=rank,
        world=world,
        tokens=tokens,
        sequence=len(AUDITS),
        execution_mode="cuda_graph",
        timing_protocol=PROTOCOL,
        checks=[],
        input_sha256=[digest(t) for t in tensors],
    )

    def raw():
        return RawAdapter.__call__(adapter, *tensors)

    with eager.set_forward_context(
        attn_metadata=None, vllm_config=vllm_config, num_tokens=tokens
    ):
        with eager._dp_sizes_context(moe):
            # Warm all JIT/native kernels and the communicator on a side stream.
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(max(warmup, 3)):
                    raw()
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with graph_capture(device=device) as capture_context:
                with count_capture_collectives() as counts:
                    with torch.cuda.graph(graph, stream=capture_context.stream):
                        captured_output = raw()
            torch.cuda.synchronize(device)
            dist.barrier()
            audit["captured_collectives"] = dict(counts)
            assert (
                sum(v for k, v in counts.items() if k.startswith("all_gather")) > 0
            ), counts
            assert (
                sum(v for k, v in counts.items() if k.startswith("reduce_scatter")) > 0
            ), counts
            snapshots = [t.clone() for t in tensors]
            original_expected = None

            def validate(phase):
                # Replay FIRST: eager must not refresh any shared communication
                # or scratch buffer before the replay being validated.
                graph.replay()
                actual = captured_output.clone()
                torch.cuda.synchronize(device)
                expected = raw().clone()
                torch.cuda.synchronize(device)
                mismatch = int((actual != expected).sum())
                relative_l2 = float(
                    (actual.float() - expected.float()).norm()
                    / expected.float().norm().clamp_min(1e-20)
                )
                finite = bool(torch.isfinite(actual).all())
                changed = (
                    int((expected != original_expected).sum())
                    if original_expected is not None
                    else 0
                )
                check = dict(
                    phase=phase,
                    bit_mismatches=mismatch,
                    rel_l2=relative_l2,
                    finite=finite,
                    changed_from_original=changed,
                )
                audit["checks"].append(check)
                good = torch.tensor(int(mismatch == 0 and finite), device=device)
                dist.all_reduce(good, op=dist.ReduceOp.MIN)
                if not good.item():
                    print("GRAPH_VALIDATION_FAIL " + json.dumps(check), flush=True)
                    raise AssertionError("Graph differs from full eager EP output")
                return expected

            original_expected = validate("original_pre")
            # Only source rank 0 changes: its computation is routed to remote experts.
            if rank == 0:
                hidden_q.neg_()
                hidden_scale.mul_(1.125)
            validate("remote_input_changed")
            for tensor, saved in zip(tensors, snapshots):
                tensor.copy_(saved)
            if rank == 0:
                local_experts = moe.w13_weight.shape[0]
                changed_ids = (
                    torch.arange(topk_ids.shape[1], device=device) + local_experts
                ) % (local_experts * world)
                topk_ids.copy_(changed_ids.expand_as(topk_ids))
                topk_weights.mul_(0.5)
            validate("remote_routes_changed")
            for tensor, saved in zip(tensors, snapshots):
                tensor.copy_(saved)
            validate("original_restored_pre")
            assert [digest(t) for t in tensors] == audit["input_sha256"]
            # Check that mutation really exercised a data-dependent result on source 0.
            if rank == 0:
                assert all(
                    c["changed_from_original"] > 0
                    for c in audit["checks"]
                    if c["phase"] in ("remote_input_changed", "remote_routes_changed")
                )
            dist.barrier()
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
                        / f"replay-np{world}-t{tokens}-seq{len(AUDITS)}-rank{rank}.json"
                    )
                )
                audit["profile_kernel_names"] = [
                    event.name
                    for event in prof.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA
                ]
            for _ in range(warmup):
                graph.replay()
            torch.cuda.synchronize(device)
            dist.barrier()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            for start, end in zip(starts, ends):
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                dist.barrier()
            # Validate the last measured output BEFORE executing any new MoE call.
            measured_mismatches = int((captured_output != original_expected).sum())
            finite = bool(torch.isfinite(captured_output).all())
            audit["checks"].append(
                dict(
                    phase="post_measure",
                    bit_mismatches=measured_mismatches,
                    finite=finite,
                    rel_l2=0.0 if measured_mismatches == 0 else None,
                )
            )
            good = torch.tensor(int(measured_mismatches == 0 and finite), device=device)
            dist.all_reduce(good, op=dist.ReduceOp.MIN)
            assert good.item(), "last measured Graph output differs"
            local = torch.tensor(
                [s.elapsed_time(e) for s, e in zip(starts, ends)],
                dtype=torch.float64,
                device=device,
            )
            rank_max = local.clone()
            dist.all_reduce(rank_max, op=dist.ReduceOp.MAX)
            means = [
                torch.empty(1, dtype=torch.float64, device=device) for _ in range(world)
            ]
            dist.all_gather(means, local.mean().reshape(1))
            adapter.output = captured_output
            adapter._captured_graph = graph
            adapter._static_inputs = tensors
            audit["capture_complete"] = True
            AUDITS.append(audit)
            with (extra.graph_audit_dir / f"np{world}-rank{rank}.jsonl").open("a") as f:
                f.write(json.dumps(audit) + "\n")
            print("GRAPH_VALIDATION " + json.dumps(audit), flush=True)
            dist.barrier()
            return rank_max.cpu().tolist(), [float(x.item()) for x in means]


def destroy_after_graphs():
    # MCCL communicators can retain captured graph user objects. Destroy the
    # graph execs before communicator teardown, while the device is alive.
    torch.cuda.synchronize()
    for adapter in list(LIVE_ADAPTERS):
        graph = getattr(adapter, "_captured_graph", None)
        if graph is not None:
            graph.reset()
            adapter._captured_graph = None
            adapter._static_inputs = None
    gc.collect()
    torch.cuda.synchronize()
    print("GRAPH_CLEANUP before_communicator_destroy", flush=True)
    original_destroy()
    print("GRAPH_CLEANUP communicator_destroyed", flush=True)


eager.destroy_model_parallel = destroy_after_graphs
eager.parse_args = parse_args
eager.PrequantizedInt8MoE = GraphAdapter
eager.benchmark = graph_benchmark
eager.TIMING_PROTOCOL = PROTOCOL
eager.main()
if int(__import__("os").environ.get("RANK", "0")) == 0 and ARGS.json_output is not None:
    records = json.loads(ARGS.json_output.read_text())
    assert len(records) == len(AUDITS)
    for record, audit in zip(records, AUDITS):
        assert record["tokens_per_rank"] == audit["tokens"]
        record["execution_mode"] = "cuda_graph"
        record["performance"][
            "timing"
        ] = "CUDA events around full native EP graph.replay; individual sample completion"
        record["graph_validation"] = audit
    ARGS.json_output.write_text(json.dumps(records, indent=2) + "\n")
