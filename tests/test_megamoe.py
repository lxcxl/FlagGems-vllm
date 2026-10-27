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

"""Tensor-interface tests independent of frozen shapes and routing fixtures."""

import ctypes
import importlib
import inspect
import os
from contextlib import ExitStack
from dataclasses import FrozenInstanceError

import pytest
import torch

from benchmark.test_megamoe import create_workspace
from flaggems_vllm import megamoe
from flaggems_vllm.runtime import device

pytestmark = [
    pytest.mark.megamoe,
    pytest.mark.skipif(device.vendor_name != "metax", reason="MetaX EP kernel"),
]


@pytest.fixture(scope="module")
def shmem():
    # Even a local SDK device call requires an initialized SHMEM device state.
    torch.empty(1, device="cuda")
    library = ctypes.CDLL(os.environ["MEGAMOE_HOST_LIB"])
    library.mxshmem_init_wrapper()
    yield library
    torch.cuda.synchronize()
    library.mxshmem_finalize_wrapper()


@pytest.fixture
def workspace(shmem):
    with ExitStack() as stack:

        def build(*args, **kwargs):
            return stack.enter_context(create_workspace(*args, shmem=shmem, **kwargs))

        yield build


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


@pytest.mark.parametrize(
    "t,h,i,experts,top_k",
    [
        (1, 128, 64, 4, 2),
        (7, 80, 70, 5, 3),
        (33, 256, 128, 8, 4),
    ],
)
def test_tensor_contract(t, h, i, experts, top_k, workspace):
    x, xs, w1, w2, s1, s2, routes, weights = make_inputs(t, h, i, experts, top_k)
    ws = workspace(x, w1, w2, weights, routes, s1, s2, a1_scale=xs)
    y = megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs)
    expected = reference(x, xs, w1, w2, s1, s2, routes, weights).to(y.device)
    error = (y.float() - expected.float()).norm() / expected.float().norm().clamp_min(
        1e-10
    )
    assert error < 0.005, error


def test_dynamic_routes_and_token_counts(workspace):
    x, xs, w1, w2, s1, s2, routes, weights = make_inputs(65, 128, 64, 8, 2)
    ws = workspace(x, w1, w2, weights, routes, s1, s2, a1_scale=xs)
    for t, expert in [(65, 7), (3, 0), (33, 6), (0, 0), (65, 1)]:
        current = routes[:t].clone()
        current[:, 0] = expert
        current[:, 1] = -1
        y = megamoe(
            x[:t], w1, w2, weights[:t], current, s1, s2, workspace=ws, a1_scale=xs[:t]
        )
        expected = reference(x[:t], xs[:t], w1, w2, s1, s2, current, weights[:t]).to(
            y.device
        )
        torch.testing.assert_close(y, expected, rtol=0.02, atol=0.015)


def test_noncontiguous_inputs_and_expert_map(workspace):
    x, xs, w1, w2, s1, s2, routes, weights = make_inputs(9, 128, 64, 4, 2)
    x = x.T.contiguous().T
    routes = routes.T.contiguous().T
    weights = weights.T.contiguous().T
    permutation = torch.tensor([2, 0, 3, 1], device=x.device)
    mapping = torch.argsort(permutation).int()
    ws = workspace(
        x,
        w1[permutation],
        w2[permutation],
        weights,
        routes,
        s1[permutation],
        s2[permutation],
        a1_scale=xs,
        expert_map=mapping,
    )
    y = megamoe(
        x,
        w1[permutation],
        w2[permutation],
        weights,
        routes,
        s1[permutation],
        s2[permutation],
        workspace=ws,
        a1_scale=xs,
        expert_map=mapping,
    )
    expected = reference(x, xs, w1, w2, s1, s2, routes, weights).to(y.device)
    torch.testing.assert_close(y, expected, rtol=0.02, atol=0.015)


def test_single_launch_and_bf16_input(workspace):
    x, xs, w1, w2, s1, s2, routes, weights = make_inputs(7, 128, 64, 4, 2)
    hidden = (x.float() * xs[:, None]).bfloat16()
    ws = workspace(hidden, w1, w2, weights, routes, s1, s2)
    megamoe(hidden, w1, w2, weights, routes, s1, s2, workspace=ws)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        y = megamoe(hidden, w1, w2, weights, routes, s1, s2, workspace=ws)
        torch.cuda.synchronize()
    assert (
        len(
            [
                e
                for e in prof.events()
                if e.device_type == torch.autograd.DeviceType.CUDA
            ]
        )
        == 1
    )
    scale = hidden.float().abs().amax(1) / 127
    q = (
        (hidden.float() * (127 / hidden.float().abs().amax(1))[:, None])
        .round()
        .to(torch.int8)
    )
    expected = reference(q, scale, w1, w2, s1, s2, routes, weights).to(y.device)
    torch.testing.assert_close(y, expected, rtol=0.02, atol=0.015)


def test_graph_replay_with_changing_routes(workspace):
    x, xs, w1, w2, s1, s2, routes, weights = make_inputs(7, 128, 64, 4, 2)
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    ws = workspace(x, w1, w2, weights, routes, s1, s2, a1_scale=xs)
    megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out)
    for expert in (0, 3, 1, 2, 0):
        routes[:, 0] = expert
        routes[:, 1] = -1
        graph.replay()
        expected = reference(x, xs, w1, w2, s1, s2, routes, weights).to(x.device)
        torch.testing.assert_close(out, expected, rtol=0.02, atol=0.015)


def test_dense_workspace_occupancy_changes_and_graph(workspace, monkeypatch):
    x, xs, w1, w2, s1, s2, routes, weights = make_inputs(19, 80, 70, 3, 2)
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    harness = importlib.import_module("benchmark.test_megamoe")
    query = harness.resident_blocks
    strided_w1 = w1.transpose(1, 2).contiguous().transpose(1, 2)

    def forbidden_query(*args):
        raise AssertionError("occupancy queried during an operator call")

    for limit, tokens in ((1, 19), (2, 3), (1, 0), (3, 19)):
        monkeypatch.setattr(harness, "resident_blocks", query)
        current_w1 = w1 if limit == 1 else strided_w1
        ws = workspace(
            x[:tokens],
            current_w1,
            w2,
            weights[:tokens],
            routes[:tokens],
            s1,
            s2,
            a1_scale=xs[:tokens],
            max_tokens=sms * 64,
            ctas_per_sm=limit,
        )
        assert ws.workers <= limit * sms
        monkeypatch.setattr(harness, "resident_blocks", forbidden_query)
        y = megamoe(
            x[:tokens],
            current_w1,
            w2,
            weights[:tokens],
            routes[:tokens],
            s1,
            s2,
            workspace=ws,
            a1_scale=xs[:tokens],
        )
        expected = reference(
            x[:tokens], xs[:tokens], w1, w2, s1, s2, routes[:tokens], weights[:tokens]
        ).to(y.device)
        torch.testing.assert_close(y, expected, rtol=0.02, atol=0.015)
    monkeypatch.setattr(harness, "resident_blocks", query)
    ws = workspace(x, w1, w2, weights, routes, s1, s2, a1_scale=xs, max_tokens=sms * 64)
    monkeypatch.setattr(harness, "resident_blocks", forbidden_query)
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out)
        torch.cuda.synchronize()
    assert (
        len(
            [
                event
                for event in prof.events()
                if event.device_type == torch.autograd.DeviceType.CUDA
            ]
        )
        == 1
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out)
    for expert in (2, 0, 1):
        routes[:, 0] = expert
        routes[:, 1] = -1
        graph.replay()
        expected = reference(x, xs, w1, w2, s1, s2, routes, weights).to(x.device)
        torch.testing.assert_close(out, expected, rtol=0.02, atol=0.015)


def test_zero_rows_out_and_unsupported_input(workspace):
    x, xs, w1, w2, s1, s2, routes, weights = make_inputs(7, 128, 64, 4, 2)
    x.zero_()
    out = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
    ws = workspace(x, w1, w2, weights, routes, s1, s2, a1_scale=xs)
    result = megamoe(
        x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out
    )
    assert result is out
    assert torch.count_nonzero(out) == 0
    with pytest.raises(ValueError, match="a1_scale"):
        megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws)
    with pytest.raises(NotImplementedError):
        megamoe(x.float(), w1, w2, weights, routes, s1, s2, workspace=ws)


def test_public_backend_exports(workspace):
    import flaggems_vllm
    from flaggems_vllm.ops.megamoe import megamoe as generic_entry
    from flaggems_vllm.runtime.backend._metax.ops.megamoe import (
        megamoe as backend_entry,
    )

    assert flaggems_vllm.megamoe is backend_entry
    assert "workspace" in inspect.signature(backend_entry).parameters
    assert "workspace" in inspect.signature(generic_entry).parameters
    assert not hasattr(flaggems_vllm, "create_megamoe_workspace")
    x, xs, w1, w2, s1, s2, routes, weights = make_inputs(3, 128, 64, 4, 2)
    ws = workspace(x, w1, w2, weights, routes, s1, s2, a1_scale=xs)
    y = generic_entry(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs)
    expected = reference(x, xs, w1, w2, s1, s2, routes, weights).to(y.device)
    torch.testing.assert_close(y, expected, rtol=0.02, atol=0.015)


def test_batched_partials_and_empty_graph(workspace):
    # A small resident grid forces several ring generations on modest tensors.
    x, xs, w1, w2, s1, s2, routes, weights = make_inputs(2049, 80, 70, 3, 2)
    x = x.T.contiguous().T
    w1 = w1.transpose(1, 2).contiguous().transpose(1, 2)
    w2 = w2.transpose(1, 2).contiguous().transpose(1, 2)
    routes = routes.T.contiguous().T
    weights = weights.T.contiguous().T
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    ws = workspace(
        x,
        w1,
        w2,
        weights,
        routes,
        s1,
        s2,
        a1_scale=xs,
        max_tokens=max(2049, sms * 32),
        workers=8,
        local_output=True,
    )
    expected = reference(x, xs, w1, w2, s1, s2, routes, weights).to(x.device)
    out = torch.empty_like(x, dtype=torch.bfloat16)

    def run():
        megamoe(
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

    run()
    torch.testing.assert_close(out, expected, rtol=0.02, atol=0.015)
    assert bool((ws.buffers[0][24:28] > 0).all()), "pipeline was not exercised"
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    saved_routes = routes.clone()
    for empty in (True, False, True, False):
        routes.copy_(torch.full_like(routes, -1) if empty else saved_routes)
        graph.replay()
        target = torch.zeros_like(expected) if empty else expected
        torch.testing.assert_close(out, target, rtol=0.02, atol=0.015)


def test_workspace_reuse_without_caching_inputs(workspace):
    x, xs, w1, w2, s1, s2, routes, weights = make_inputs(7, 80, 70, 4, 2)
    ws = workspace(x, w1, w2, weights, routes, s1, s2, a1_scale=xs)
    pointers = tuple(buffer.data_ptr() for buffer in ws.buffers)
    y = megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs)
    changed_w1 = w1.neg()
    changed_routes = routes[:5].flip(1)
    actual = megamoe(
        x[:5],
        changed_w1,
        w2,
        weights[:5],
        changed_routes,
        s1,
        s2,
        workspace=ws,
        a1_scale=xs[:5],
    )
    expected = reference(
        x[:5], xs[:5], changed_w1, w2, s1, s2, changed_routes, weights[:5]
    ).to(x.device)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.015)
    assert pointers == tuple(buffer.data_ptr() for buffer in ws.buffers)
    assert y.shape == x.shape


def test_workspace_is_explicit_and_caller_owned(workspace):
    x, xs, w1, w2, s1, s2, routes, weights = make_inputs(3, 128, 64, 4, 2)
    first = workspace(x, w1, w2, weights, routes, s1, s2, a1_scale=xs)
    second = workspace(x, w1, w2, weights, routes, s1, s2, a1_scale=xs)
    assert first.buffers[0].data_ptr() != second.buffers[0].data_ptr()
    with pytest.raises(FrozenInstanceError):
        first.workers = 1
    with pytest.raises(TypeError, match="workspace"):
        megamoe(x, w1, w2, weights, routes, s1, s2, a1_scale=xs)
    for ws in (first, second, first):
        actual = megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs)
        expected = reference(x, xs, w1, w2, s1, s2, routes, weights).to(x.device)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.015)


def test_workspace_capacity_and_fixed_launch(workspace, monkeypatch):
    x, xs, w1, w2, s1, s2, routes, weights = make_inputs(8, 80, 70, 4, 2)
    ws = workspace(
        x[:7],
        w1,
        w2,
        weights[:7],
        routes[:7],
        s1,
        s2,
        a1_scale=xs[:7],
        max_tokens=7,
        workers=3,
        local_output=True,
    )
    harness = importlib.import_module("benchmark.test_megamoe")

    def forbidden_query(*args):
        raise AssertionError("occupancy queried during an operator call")

    monkeypatch.setattr(harness, "resident_blocks", forbidden_query)
    actual = megamoe(
        x[:7],
        w1,
        w2,
        weights[:7],
        routes[:7],
        s1,
        s2,
        workspace=ws,
        a1_scale=xs[:7],
    )
    expected = reference(x[:7], xs[:7], w1, w2, s1, s2, routes[:7], weights[:7]).to(
        x.device
    )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.015)
    assert ws.workers == 3
    with pytest.raises(ValueError, match="capacity"):
        megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs)
