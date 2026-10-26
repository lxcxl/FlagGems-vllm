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

"""Test the public MegaMoE API and seeded INT8 numerical contract."""

import importlib

import pytest
import torch

import flaggems_vllm

pytestmark = pytest.mark.megamoe


def _ppu_api():
    if flaggems_vllm.vendor_name != "thead":
        pytest.skip("requires the PPU backend")
    return importlib.import_module("flaggems_vllm.runtime.backend._thead.ops.megamoe")


def _require_ppu_device():
    api = _ppu_api()
    if not torch.cuda.is_available():
        pytest.skip("requires a PPU device")
    return api


def _reference_int8_moe(x, scales, w1, s1, w2, s2, router, ids):
    """Independent CPU reference, including the two BF16 rounding boundaries."""
    t, h = x.shape
    partial = torch.zeros((ids.shape[1], t, h), dtype=torch.bfloat16)
    for expert in range(w1.shape[0]):
        tokens, slots = torch.where(ids == expert)
        if not len(tokens):
            continue
        first = x[tokens].float() @ w1[expert].float().T
        first = (first * scales[tokens] * s1[expert].T).bfloat16().float()
        gate, up = first.chunk(2, dim=-1)
        silu = (gate / (1 + torch.exp(-gate))).bfloat16().float()
        activated = (silu * up).bfloat16().float()
        amax = activated.abs().amax(-1, keepdim=True).clamp_min(1e-10)
        scale = amax / 127
        scaled = activated * (127 / amax)
        quantized = (scaled.sign() * (scaled.abs() + 0.5).floor()).to(torch.int8)
        second = quantized.float() @ w2[expert].float().T
        values = second * scale * s2[expert].T * router[tokens, slots, None]
        partial[slots, tokens] = values.bfloat16()
    return partial.float().sum(0).bfloat16()


def test_megamoe_is_exported_on_ppu():
    api = _ppu_api()
    assert callable(api.megamoe)
    assert flaggems_vllm.megamoe is api.megamoe
    package = importlib.import_module(
        "flaggems_vllm.runtime.backend._thead.mega.megamoe"
    )
    assert package.megamoe is api.megamoe
    assert package.pack_gate_up_rows is api.pack_gate_up_rows
    assert package.peer_buffer_sizes is api.peer_buffer_sizes


def test_megamoe_fp8_is_exported_on_hopper():
    if flaggems_vllm.vendor_name != "nvidia":
        pytest.skip("requires the NVIDIA backend")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires Hopper")
    from flaggems_vllm.utils.triton_version_utils import has_triton_tle

    if not has_triton_tle(3, 6, 0):
        pytest.skip("requires FlagTree TLE")
    api = pytest.importorskip(
        "flaggems_vllm.runtime.backend._nvidia.hopper.ops.megamoe"
    )
    assert flaggems_vllm.megamoe_fp8 is api.megamoe_fp8


@pytest.mark.parametrize("dtype", [torch.int8, torch.float32])
@pytest.mark.parametrize("shape", [(1, 32, 1), (2, 64, 3)])
def test_megamoe_w1_layout(dtype, shape):
    api = _require_ppu_device()
    source = (
        torch.arange(shape[0] * shape[1] * shape[2], dtype=torch.int32)
        .reshape(shape)
        .to(dtype)
    )
    half = shape[1] // 2
    expected = torch.stack(
        (
            source[:, :half].reshape(shape[0], half // 8, 8, shape[2]),
            source[:, half:].reshape(shape[0], half // 8, 8, shape[2]),
        ),
        dim=2,
    ).reshape(shape)
    actual = api.pack_gate_up_rows(source.cuda())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape,error",
    [
        ((4, 8), "rank 3"),
        ((1, 15, 2), "even row count"),
        ((1, 20, 2), "divisible by 8"),
    ],
)
def test_megamoe_w1_layout_rejects_invalid_shape(shape, error):
    api = _ppu_api()
    with pytest.raises(ValueError, match=error):
        api.pack_gate_up_rows(torch.empty(shape, dtype=torch.int8))


@pytest.mark.parametrize(
    "values,error_type",
    [
        ((0, 256, 128, 8, 2, 2), ValueError),
        ((True, 256, 128, 8, 2, 2), ValueError),
        ((17, 192, 128, 8, 2, 2), NotImplementedError),
        ((17, 256, 192, 8, 2, 2), NotImplementedError),
        ((17, 256, 128, 7, 2, 2), ValueError),
        ((17, 256, 128, 8, 9, 2), ValueError),
    ],
)
def test_megamoe_rejects_unsupported_contract(values, error_type):
    api = _ppu_api()
    with pytest.raises(error_type):
        api.peer_buffer_sizes(*values)


def test_megamoe_peer_capacity_covers_routing_skew():
    sizes = _ppu_api().peer_buffer_sizes(17, 256, 384, 10, 3, 2)
    assert sizes["tok_tab"] == 17 * 256
    assert sizes["cb_tab"] == 17 * 3 * 256 * 2
    assert sizes["meta_tab"] >= 2 * 17 * 3 * 12
    assert set(sizes) == {
        "tok_tab",
        "sf_tab",
        "w_tab",
        "route_tab",
        "cb_tab",
        "sig_tab",
        "meta_tab",
    }


@pytest.mark.parametrize("shape", [(1, 128, 128, 4, 1), (17, 256, 384, 5, 3)])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("routing", ["normal", "skew", "dropped"])
def test_megamoe_int8_correctness(shape, ids_dtype, routing):
    api = _require_ppu_device()
    t, h, i, e, k = shape
    rng = torch.Generator().manual_seed(918)
    x = torch.randint(-8, 9, (t, h), dtype=torch.int8, generator=rng)
    scales = torch.rand(t, 1, generator=rng) * 0.02 + 0.01
    w1 = torch.randint(-8, 9, (e, 2 * i, h), dtype=torch.int8, generator=rng)
    s1 = torch.rand(e, 2 * i, 1, generator=rng) * 0.02 + 0.01
    w2 = torch.randint(-8, 9, (e, h, i), dtype=torch.int8, generator=rng)
    s2 = torch.rand(e, h, 1, generator=rng) * 0.02 + 0.01
    router = torch.rand(t, k, generator=rng)
    ids = torch.randint(e, (t, k), dtype=ids_dtype, generator=rng)
    if routing == "skew":
        ids.fill_(e - 1)
    elif routing == "dropped":
        ids.fill_(-1)
    expected = _reference_int8_moe(x, scales, w1, s1, w2, s2, router, ids)
    actual = api.megamoe(
        x.cuda(),
        scales.cuda(),
        api.pack_gate_up_rows(w1.cuda()),
        api.pack_gate_up_rows(s1.cuda()),
        w2.cuda(),
        s2.cuda(),
        router.cuda(),
        ids.cuda(),
    )
    assert actual.dtype == torch.bfloat16 and actual.shape == (t, h)
    actual = actual.cpu()
    if routing == "dropped":
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        from benchmark.test_megamoe import _assert_relative_l2

        _assert_relative_l2(actual, expected)


@pytest.mark.parametrize("num_ranks", [2, 4, 8])
@pytest.mark.parametrize("model_shape", [False, True], ids=["odd", "model"])
def test_megamoe_expert_parallel(num_ranks, model_shape):
    _require_ppu_device()
    if torch.cuda.device_count() < num_ranks:
        pytest.skip(f"requires {num_ranks} visible PPU devices")
    from benchmark.test_megamoe import _BenchmarkConfig, _launch_megamoe

    config = _BenchmarkConfig(
        num_ranks=num_ranks,
        tokens=32 if model_shape else 17,
        hidden=2048 if model_shape else 256,
        intermediate=512 if model_shape else 384,
        num_experts=256 if model_shape else num_ranks * 5,
        topk=8 if model_shape else 3,
        stages=4,
        warmup=0,
        iterations=1,
        timeout=120,
        ppu=True,
        validate=True,
    )
    assert _launch_megamoe(config) > 0


def test_megamoe_launcher_parses_every_rank():
    from benchmark.test_megamoe import _parse_rank_latencies

    stdout = "\n".join(
        [
            "MEGAMOE_RESULT scope=kernel rank=1 world=2 latency_us=1500.000",
            "MEGAMOE_RESULT scope=kernel rank=0 world=2 latency_us=1250.000",
        ]
    )
    assert _parse_rank_latencies(stdout, 2) == (1.25, 1.5)
    with pytest.raises(RuntimeError, match="every rank"):
        _parse_rank_latencies(stdout.splitlines()[0], 2)
    with pytest.raises(RuntimeError, match="every rank"):
        _parse_rank_latencies("\n".join([stdout.splitlines()[0]] * 2), 2)


def test_megamoe_worker_cli_preserves_test_config(monkeypatch):
    from benchmark.test_megamoe import (
        _BenchmarkConfig,
        _parse_worker_config,
        _worker_arguments,
    )

    config = _BenchmarkConfig(
        num_ranks=2,
        tokens=17,
        hidden=256,
        intermediate=384,
        num_experts=10,
        topk=3,
        stages=4,
        warmup=2,
        iterations=3,
        timeout=90,
        ppu=True,
        validate=True,
    )
    monkeypatch.setattr("sys.argv", ["worker", *_worker_arguments(config)])
    assert _parse_worker_config() == config
