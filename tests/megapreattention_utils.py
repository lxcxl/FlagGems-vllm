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

"""Independent torch oracle and Hygon layer3 test adapter (test-only).

Oracle is derived from frozen NV PR #1007; it gathers original hidden shards
and never reads candidate communication buffers or intermediate projections.
"""
import importlib.util
import json
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@dataclass(frozen=True)
class Config:
    num_q_heads: int
    num_kv_heads: int
    head_dim: int = 256
    rotary_dim: int = 64
    mrope_section: tuple[int, int, int] = (11, 11, 10)
    eps: float = 1e-6
    global_kv_heads: int = 2

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
class Inputs:
    hidden: torch.Tensor
    weight: torch.Tensor
    positions: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    rope: torch.Tensor
    cache: torch.Tensor
    slots: torch.Tensor
    scale: torch.Tensor
    group: object
    config: Config


@dataclass
class Output:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    gate: torch.Tensor  # [tokens, local_q_heads, head_dim], possibly strided.
    workspace: tuple = ()

    def tensors(self):
        return self.query, self.key, self.value, self.gate


def independent_reference(inputs):
    """Oracle starts at original shards; never reads candidate scratch/results."""
    c, t = inputs.config, inputs.positions.shape[1]
    hidden = inputs.hidden.new_empty((t, inputs.hidden.shape[1]))
    dist.all_gather_into_tensor(hidden, inputs.hidden, group=inputs.group)
    qkv = hidden @ inputs.weight.t()
    qgate, key, value = qkv.split((2 * c.q_size, c.kv_size, c.kv_size), dim=-1)
    query, gate = qgate.view(t, c.num_q_heads, 2 * c.head_dim).chunk(2, dim=-1)
    frequency = torch.arange(c.rotary_dim // 2, device="cuda")
    axis = torch.zeros_like(frequency)
    axis[1 : 3 * c.mrope_section[1] : 3] = 1
    axis[2 : 3 * c.mrope_section[2] : 3] = 2
    position = inputs.positions[axis].t()
    cos = inputs.rope[position, frequency].float().unsqueeze(1)
    sin = inputs.rope[position, frequency + c.rotary_dim // 2].float().unsqueeze(1)

    def norm_rope(x, weight):
        x = x.float()
        normalized = (
            (
                x
                * torch.rsqrt(x.square().mean(-1, keepdim=True) + c.eps)
                * (weight.float() + 1)
            )
            .to(torch.bfloat16)
            .float()
        )
        first, second = normalized[..., : c.rotary_dim].chunk(2, dim=-1)
        rotated = torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )
        return torch.cat((rotated, normalized[..., c.rotary_dim :]), dim=-1).to(
            torch.bfloat16
        )

    query = norm_rope(query, inputs.q_norm).reshape(t, c.q_size)
    key = norm_rope(key.view(t, c.num_kv_heads, c.head_dim), inputs.k_norm)
    valid = inputs.slots >= 0
    slots = inputs.slots[valid]
    block, offset = slots // inputs.cache.shape[2], slots % inputs.cache.shape[2]
    inputs.cache[block, :, offset, : c.head_dim] = key[valid]
    inputs.cache[block, :, offset, c.head_dim :] = value.view(
        t, c.num_kv_heads, c.head_dim
    )[valid]
    return Output(query, key.reshape(t, c.kv_size), value, gate)


def changed(x, **kw):
    return SimpleNamespace(**{**vars(x), **kw})


def implementation():
    # Direct module loading keeps this distributed adapter independent of vLLM
    # and unrelated optional operators. Public-export coverage lives in pytest.
    if "megapreattention_hy_test_impl" not in sys.modules:
        path = (
            Path(__file__).resolve().parents[1]
            / "src/flaggems_vllm/runtime/backend/_hygon/ops/megapreattention.py"
        )
        spec = importlib.util.spec_from_file_location(
            "megapreattention_hy_test_impl", path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules["megapreattention_hy_test_impl"]


def candidate(x):
    c = x.config
    return Output(
        *implementation().megapreattention(
            x.hidden,
            x.weight,
            x.positions,
            x.q_norm,
            x.k_norm,
            x.rope,
            x.cache,
            x.slots,
            x.group,
            num_heads=c.num_q_heads * x.group.size(),
            num_kv_heads=c.global_kv_heads,
            head_dim=c.head_dim,
            rotary_dim=c.rotary_dim,
            mrope_section=c.mrope_section,
            eps=c.eps,
        )
    )


def check_outputs(actual, expected, cache, reference_cache):
    errors = []
    for name, a, b in zip(
        ("query", "key", "value", "gate", "cache"),
        (*actual.tensors(), cache),
        (*expected.tensors(), reference_cache),
    ):
        if not torch.isfinite(a).all():
            raise AssertionError(f"{name}: nonfinite output")
        error = (a.float() - b.float()).abs()
        errors.append(
            {
                "name": name,
                "max_abs": error.max().item(),
                "relative_rms": (
                    error.square().mean().sqrt()
                    / b.float().square().mean().sqrt().clamp_min(1e-12)
                ).item(),
            }
        )
        torch.testing.assert_close(
            a, b, atol=0.03, rtol=0.016, msg=lambda msg: f"{name}: {msg}"
        )
    return errors


def edge_inputs(x):
    t = x.positions.shape[1]
    pos = torch.stack(
        (
            torch.arange(t, device=x.hidden.device),
            torch.arange(t, device=x.hidden.device) * 2,
            torch.arange(t, device=x.hidden.device) * 3,
        )
    )
    backing = torch.empty((3, 2 * t), dtype=torch.int64, device=x.hidden.device)
    backing[:, ::2] = pos
    slots = torch.empty(2 * t, device=x.hidden.device, dtype=torch.int64)
    slots[::2] = torch.arange(t - 1, -1, -1, device=x.hidden.device)
    slots[::34] = -1
    b, h, n, d = x.cache.shape
    cache = torch.full(
        (n, h, b, d), 0.375, device=x.hidden.device, dtype=torch.bfloat16
    ).permute(2, 1, 0, 3)
    weight_storage = torch.empty(
        x.weight.numel() + 1, device=x.hidden.device, dtype=x.weight.dtype
    )
    weight = weight_storage[1:].view(x.weight.shape)
    weight.copy_(x.weight)
    hidden_storage = torch.empty(
        x.hidden.numel() + 1, device=x.hidden.device, dtype=x.hidden.dtype
    )
    hidden = hidden_storage[1:].view(x.hidden.shape)
    hidden.copy_(x.hidden)
    return changed(
        x,
        positions=backing[:, ::2],
        slots=slots[::2],
        cache=cache,
        hidden=hidden,
        weight=weight,
    )


def check_contract(x):
    # These all fail before communicator creation/kernel launch.
    cases = {
        "fp32": changed(x, hidden=x.hidden.float()),
        "requires_grad": changed(x, weight=x.weight.clone().requires_grad_(True)),
        "strided_hidden": changed(
            x,
            hidden=torch.empty(
                (x.hidden.shape[0], x.hidden.shape[1] * 2),
                device=x.hidden.device,
                dtype=x.hidden.dtype,
            )[:, ::2],
        ),
        "empty": changed(
            x, hidden=x.hidden[:0], positions=x.positions[:, :0], slots=x.slots[:0]
        ),
        "overlap_cache": changed(
            x, cache=x.cache.as_strided(x.cache.shape, (0, *x.cache.stride()[1:]))
        ),
        "bad_weight": changed(x, weight=x.weight[:-1]),
    }
    for label, bad in cases.items():
        try:
            candidate(bad)
        except (ValueError, NotImplementedError):
            pass
        else:
            raise AssertionError(f"accepted unsupported input: {label}")
    return list(cases)


def distribution(samples):
    v = sorted(samples)

    def percentile(q):
        index = (len(v) - 1) * q
        i = int(index)
        return v[i] + (v[min(i + 1, len(v) - 1)] - v[i]) * (index - i)

    median = statistics.median(samples)
    span = (percentile(0.9) - percentile(0.1)) / median
    return dict(
        mean_ms=statistics.mean(samples),
        median_ms=median,
        span=span,
        stable=len(v) >= 10 and span <= 0.05,
    )


def measure(calls, iterations, mode):
    from torch._dynamo.utils import counters
    from triton.runtime.jit import JITFunction

    original = JITFunction._do_compile
    compiled = []

    def tracked(*args, **kwargs):
        compiled.append(True)
        return original(*args, **kwargs)

    before = dict(counters["stats"])
    JITFunction._do_compile = tracked
    samples = {name: [] for name in calls}
    try:
        names = list(calls)
        for i in range(iterations):
            for name in names[i % len(names) :] + names[: i % len(names)]:
                dist.barrier()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                begin = time.perf_counter_ns()
                start.record()
                calls[name]()
                end.record()
                end.synchronize()
                elapsed = (
                    (time.perf_counter_ns() - begin) / 1e6
                    if mode == "operator"
                    else start.elapsed_time(end)
                )
                samples[name].append(elapsed)
    finally:
        JITFunction._do_compile = original
    assert not compiled, "measurement triggered Triton JIT"
    assert before == dict(counters["stats"]), "measurement triggered Dynamo compilation"
    return samples


def _check_growth(x):
    impl = implementation()
    tp = x.group.size()
    small_t = 128 * tp
    small = changed(
        x,
        hidden=x.hidden[:128],
        positions=x.positions[:, :small_t],
        slots=x.slots[:small_t],
        cache=torch.full_like(x.cache, 0.375),
    )
    candidate(small)
    candidate(x)
    assert not impl._retired
    impl.clear_megapreattention_cache()
    candidate(small)
    dist.barrier()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = candidate(small)
    # Creating the larger generation must preserve the captured smaller one.
    candidate(x)
    dist.barrier()
    torch.cuda.synchronize()
    oracle_cache = torch.full_like(small.cache, 0.375)
    expected = independent_reference(changed(small, cache=oracle_cache))
    small.cache.fill_(0.375)
    graph.replay()
    torch.cuda.synchronize()
    check_outputs(out, expected, small.cache, oracle_cache)
    graph.reset()
    assert len(impl._retired) == 1
    irregular_t = 384 * tp
    irregular = changed(
        x,
        hidden=x.hidden[:384],
        positions=x.positions[:, :irregular_t],
        slots=x.slots[:irregular_t],
        cache=torch.full_like(x.cache, 0.375),
    )
    ref_cache = torch.full_like(x.cache, 0.375)
    expected = independent_reference(changed(irregular, cache=ref_cache))
    out = candidate(irregular)
    torch.cuda.synchronize()
    check_outputs(out, expected, irregular.cache, ref_cache)


def _worker(
    rank,
    tp,
    directory,
    root,
    case_ids,
    mode,
    warmup,
    iterations,
    check_only,
    edge,
    rope_parameters,
):
    import faulthandler

    torch.set_num_threads(4)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=tp,
        init_method=f"file://{directory}/rdzv",
        device_id=torch.device("cuda", rank),
        timeout=timedelta(seconds=180),
    )
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "tools/megapreattention_data")
    )
    from dataset import Dataset

    data = Dataset(root)
    reports = []
    impl = implementation()
    try:
        for case_id in case_ids:
            faulthandler.dump_traceback_later(600, exit=True)
            x = data.load_rank(case_id, rank, f"cuda:{rank}", dist.group.WORLD)
            print(rank, case_id, "loaded", flush=True)
            if rope_parameters:
                x = changed(x, config=replace(x.config, **rope_parameters))
            if edge:
                x = edge_inputs(x)
            oracle_cache = torch.full_like(x.cache, 0.375)
            expected = independent_reference(changed(x, cache=oracle_cache))
            print(rank, "oracle ready", flush=True)
            if check_only:
                with torch.cuda.device((rank + 1) % tp):
                    actual = candidate(x)
                    assert torch.cuda.current_device() == (rank + 1) % tp
            else:
                actual = candidate(x)
            torch.cuda.synchronize()
            print(rank, "candidate ready", flush=True)
            errors = check_outputs(actual, expected, x.cache, oracle_cache)
            functions = {"megapreattention": candidate}
            sources = {"megapreattention": x}
            baseline_meta = None
            if not check_only:
                from .megapreattention_native import NativeWindow

                native = NativeWindow(x)
                sources["native"] = changed(x, cache=torch.full_like(x.cache, 0.375))
                functions["native"] = native
                baseline_meta = native.evidence
                result = native(sources["native"])
                torch.cuda.synchronize()
                check_outputs(result, expected, sources["native"].cache, oracle_cache)
            dist.barrier()
            for _ in range(warmup):
                for name, f in functions.items():
                    f(sources[name])
            dist.barrier()
            torch.cuda.synchronize()
            print(rank, "warmup ready", flush=True)
            graphs, outputs, calls = {}, {}, {}
            if mode == "cudagraph":
                for name, f in functions.items():
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        outputs[name] = f(sources[name])
                    graphs[name] = graph
                    calls[name] = graph.replay
            else:
                calls = {
                    name: partial(functions[name], sources[name]) for name in functions
                }
            print(rank, "capture ready", flush=True)
            saved_hidden = x.hidden.clone()
            for factor in (-0.75, -1.25):
                x.hidden.mul_(factor)
                oracle_cache.fill_(0.375)
                expected = independent_reference(changed(x, cache=oracle_cache))
                for w in impl._windows.values():
                    w.buffer.fill_(float("nan"))
                for name in functions:
                    sources[name].cache.fill_(0.375)
                    if mode == "cudagraph":
                        calls[name]()
                        result = outputs[name]
                    else:
                        result = functions[name](sources[name])
                    torch.cuda.synchronize()
                    errors = check_outputs(
                        result, expected, sources[name].cache, oracle_cache
                    )
            x.hidden.copy_(saved_hidden)
            del saved_hidden
            print(rank, "changed input ready", flush=True)
            rejected = check_contract(x) if check_only else []
            samples = {} if check_only else measure(calls, iterations, mode)
            record = {
                "rank": rank,
                "errors": errors,
                "rejected": rejected,
                "samples_ms": samples,
                "resources": [w.resources for w in impl._windows.values()],
            }
            ranks = [None] * tp
            dist.all_gather_object(ranks, record)
            metrics = {}
            if not check_only:
                for name in functions:
                    means = [statistics.mean(r["samples_ms"][name]) for r in ranks]
                    metrics[name] = dict(
                        latency_ms=max(means),
                        rank_mean_ms=means,
                        rank_max_distribution=distribution(
                            [
                                max(r["samples_ms"][name][i] for r in ranks)
                                for i in range(iterations)
                            ]
                        ),
                    )
            reports.append(
                dict(
                    case=case_id,
                    correctness_passed=True,
                    measurement_jit_free=None if check_only else True,
                    mode=mode,
                    timing="wall_ms" if mode == "operator" else "device_ms",
                    edge=edge,
                    baseline=baseline_meta,
                    metrics=metrics,
                    ranks=ranks,
                )
            )
            for graph in graphs.values():
                graph.reset()
            outputs.clear()
            graphs.clear()
            calls.clear()
            torch.cuda.synchronize()
            dist.barrier()
            impl.clear_megapreattention_cache()
            if check_only and case_id.endswith("_t4096"):
                _check_growth(x)
                impl.clear_megapreattention_cache()
            if rank == 0:
                print(
                    json.dumps(
                        {"case": case_id, "correctness": "pass", "metrics": metrics}
                    ),
                    flush=True,
                )
            del x, sources, functions, expected, actual
        if rank == 0:
            (Path(directory) / "report.json").write_text(json.dumps(reports, indent=2))
    finally:
        faulthandler.cancel_dump_traceback_later()
        dist.destroy_process_group()


def run_distributed(
    tp,
    case_ids,
    *,
    root=None,
    mode="cudagraph",
    warmup=3,
    iterations=10,
    rounds=1,
    check_only=True,
    edge=False,
    rope_parameters=None,
):
    root = root or os.environ.get(
        "MEGAPREATTENTION_DATA",
        "/data/lixichen/megapreattention_data/layer3_nv_20261027_v1",
    )
    if edge and not check_only:
        raise ValueError(
            "edge layouts are accuracy-only; benchmark uses the manifest layout"
        )
    reports = []
    for _ in range(rounds):
        with tempfile.TemporaryDirectory(prefix="megapreattention-hy-") as d:
            context = mp.spawn(
                _worker,
                args=(
                    tp,
                    d,
                    root,
                    list(case_ids),
                    mode,
                    warmup,
                    iterations,
                    check_only,
                    edge,
                    rope_parameters,
                ),
                nprocs=tp,
                join=False,
            )
            deadline = time.monotonic() + max(600, 600 * len(case_ids))
            try:
                while not context.join(timeout=1):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("MegaPreAttention workers exceeded timeout")
                reports.append(json.loads((Path(d) / "report.json").read_text()))
            finally:
                for p in context.processes:
                    if p.is_alive():
                        p.terminate()
                for p in context.processes:
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()
                        p.join(timeout=5)
    summary = {}
    if not check_only:
        for i, case_id in enumerate(case_ids):
            values = {
                name: statistics.mean(
                    r[i]["metrics"][name]["latency_ms"] for r in reports
                )
                for name in ("native", "megapreattention")
            }
            values["speedup"] = values["native"] / values["megapreattention"]
            values["stable_all_rounds"] = all(
                r[i]["metrics"][n]["rank_max_distribution"]["stable"]
                for r in reports
                for n in ("native", "megapreattention")
            )
            summary[case_id] = values
    return dict(
        correctness_passed=True, tp=tp, rounds=rounds, summary=summary, reports=reports
    )
