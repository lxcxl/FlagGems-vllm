#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run Hygon same-byte layer3 accuracy and paired Graph benchmarks."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.megapreattention_utils import run_distributed  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root", default="/data/lixichen/megapreattention_data/layer3_nv_20261027_v1"
    )
    p.add_argument("--tp", type=int, choices=(2, 4), default=2)
    p.add_argument("--case", action="append")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--edge", action="store_true")
    p.add_argument("--mode", choices=("operator", "cudagraph"), default="cudagraph")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iter", type=int, default=100)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.edge and not a.check_only:
        p.error("--edge requires --check-only")
    manifest = json.loads((Path(a.root) / "manifest.json").read_text())
    cases = a.case or [c["id"] for c in manifest["cases"] if c["tp"] == a.tp]
    known = {c["id"]: c for c in manifest["cases"]}
    if not cases or any(c not in known or known[c]["tp"] != a.tp for c in cases):
        p.error("all case IDs must exist in the manifest and match --tp")
    if min(a.rounds, a.iter) <= 0 or a.warmup < 0:
        p.error("rounds/iter must be positive and warmup must be nonnegative")
    if not torch.version.hip or torch.cuda.device_count() < a.tp:
        p.error(f"requires at least {a.tp} visible Hygon HIP devices")
    result = run_distributed(
        a.tp,
        cases,
        root=a.root,
        mode=a.mode,
        warmup=a.warmup,
        iterations=a.iter,
        rounds=a.rounds,
        check_only=a.check_only,
        edge=a.edge,
    )
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
