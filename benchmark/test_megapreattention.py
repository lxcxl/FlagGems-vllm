# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Hygon paired full-window benchmark.

Baseline uses RCCL AllGather + rocBLAS matmul + torch.compile postprocessing.
It is explicitly a Hygon local baseline, not NV's overlapping SymmMem/vLLM
baseline, which this environment cannot execute. No NV speedups are inferred.
"""
import os
from pathlib import Path

import pytest
import torch

from tests.megapreattention_utils import run_distributed

from . import base
from .conftest import Config


class MegaPreAttentionBenchmark(base.Benchmark):
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))

    def __init__(self):
        super().__init__("megapreattention", self.native, gems_op=self.candidate)
        self.paired_results = {}

    @staticmethod
    def native():
        """RCCL/rocBLAS/compiled-post baseline label."""

    @staticmethod
    def candidate():
        """Hygon candidate label."""

    def get_input_iter(self, dtype):
        yield from self.shapes

    def get_latency(
        self, op, tokens, tp, hidden, q_heads, kv_heads, head_dim, rotary_dim
    ):
        if not torch.version.hip or torch.cuda.device_count() < tp:
            pytest.skip(f"requires {tp} Hygon GPUs")
        root = os.environ.get(
            "MEGAPREATTENTION_DATA",
            "/data/lixichen/megapreattention_data/layer3_nv_20261027_v1",
        )
        if not Path(root, "manifest.json").exists():
            pytest.skip("layer3 dataset unavailable")
        dims = (hidden, q_heads, kv_heads)
        models = {(2048, 16, 2): "Qwen3.6-35B-A3B", (5120, 24, 4): "Qwen3.6-27B"}
        if dims not in models or (head_dim, rotary_dim) != (256, 64):
            raise NotImplementedError(
                "benchmark dataset covers only the two D256/R64 models"
            )
        case_id = f"{models[dims]}_tp{tp}_t{tokens}"
        mode = Config.mode.value
        if mode not in ("kernel", "operator", "cudagraph"):
            raise ValueError("use kernel, operator or cudagraph mode")
        key = (case_id, mode, Config.warm_up, Config.repetition)
        if key not in self.paired_results:
            self.paired_results[key] = run_distributed(
                tp,
                [case_id],
                root=root,
                mode=mode,
                check_only=False,
                warmup=Config.warm_up,
                iterations=Config.repetition,
                rounds=5,
            )
        name = "native" if op == self.native else "megapreattention"
        return self.paired_results[key]["summary"][case_id][name]


@pytest.mark.megapreattention
def test_megapreattention():
    MegaPreAttentionBenchmark().run()
