# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Hygon true-layer3, Graph and unsupported-input coverage."""
import os
from pathlib import Path

import pytest
import torch

from . import conftest as cfg
from .megapreattention_utils import run_distributed

pytestmark = pytest.mark.megapreattention


def require_hygon(tp):
    if not torch.version.hip or torch.cuda.device_count() < tp:
        pytest.skip(f"requires {tp} Hygon GPUs")
    root = os.environ.get(
        "MEGAPREATTENTION_DATA",
        "/data/lixichen/megapreattention_data/layer3_nv_20261027_v1",
    )
    if not Path(root, "manifest.json").exists():
        pytest.skip("layer3 dataset unavailable")
    return root


def test_megapreattention_exports():
    import flaggems_vllm
    from flaggems_vllm.ops import clear_megapreattention_cache, megapreattention

    assert callable(flaggems_vllm.megapreattention)
    assert callable(flaggems_vllm.clear_megapreattention_cache)
    assert callable(megapreattention) and callable(clear_megapreattention_cache)


@pytest.mark.parametrize("tp", [2] if cfg.QUICK_MODE else [2, 4])
@pytest.mark.parametrize("edge", [False, True])
def test_megapreattention(tp, edge):
    root = require_hygon(tp)
    tokens = (4096,) if cfg.QUICK_MODE or edge else (4096, 8192, 16384, 32768)
    cases = [
        f"{m}_tp{tp}_t{t}" for m in ("Qwen3.6-35B-A3B", "Qwen3.6-27B") for t in tokens
    ]
    result = run_distributed(
        tp,
        cases,
        root=root,
        edge=edge,
        mode="cudagraph",
        check_only=True,
        rounds=1,
        warmup=3,
    )
    assert result["correctness_passed"]


@pytest.mark.parametrize("sections,eps", [((16, 8, 8), 1e-5), ((8, 12, 12), 1e-4)])
def test_megapreattention_rope_parameters(sections, eps):
    root = require_hygon(2)
    result = run_distributed(
        2,
        ["Qwen3.6-35B-A3B_tp2_t4096"],
        root=root,
        edge=True,
        check_only=True,
        rounds=1,
        warmup=2,
        rope_parameters=dict(mrope_section=sections, eps=eps),
    )
    assert result["correctness_passed"]
