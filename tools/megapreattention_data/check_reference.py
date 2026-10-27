#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Check fixture compatibility with the unmodified NVIDIA loader function."""

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from dataset import Dataset, file_sha


def check(root, reference):
    # Execute only the original loader function; avoid importing vLLM, test
    # conftest, NVIDIA kernels or process-group machinery during data checks.
    tree = ast.parse(reference.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_fixture"
    )
    namespace = {"torch": torch, "hashlib": hashlib}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(reference), "exec"),
        namespace,
    )
    data = Dataset(root)
    checked = 0
    for model, info in data.manifest["models"].items():
        fixture = namespace["load_fixture"](
            str(root / info["fixture"]), model_dims=tuple(info["model_dims"])
        )
        weights = fixture["weights"]
        hidden, q_heads, kv_heads = info["model_dims"]
        for tp in (2, 4):
            packed = []
            for rank in range(tp):
                # Independent layout-based check: q_proj consists of one
                # [Q,gate] pair per head; split whole heads across TP ranks.
                q = weights["q_proj.weight"].view(q_heads, 2, 256, hidden)
                q = q.chunk(tp, dim=0)[rank].reshape(-1, hidden)
                kv_indices = (
                    list(range(rank * (kv_heads // tp), (rank + 1) * (kv_heads // tp)))
                    if kv_heads >= tp
                    else [rank // (tp // kv_heads)]
                )
                parts = [q]
                for name in ("k_proj.weight", "v_proj.weight"):
                    parts.append(
                        weights[name]
                        .view(kv_heads, 256, hidden)[kv_indices]
                        .reshape(-1, hidden)
                    )
                packed.append(torch.cat(parts))
                case_id = f"{model}_tp{tp}_t4096"
                replay = data.load_rank(case_id, rank)
                assert torch.equal(replay.weight, packed[-1])
                del replay
                checked += 1
            if kv_heads < tp:
                # TP4/35B: adjacent rank pairs share the K/V projection rows.
                qwidth = (q_heads // tp) * 2 * 256
                assert torch.equal(packed[0][qwidth:], packed[1][qwidth:])
                assert torch.equal(packed[2][qwidth:], packed[3][qwidth:])
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "nvidia_fixture_loader_models": 2,
        "independent_head_layout_packed_ranks": checked,
        "reference_helper": str(reference),
        "reference_helper_sha256": file_sha(reference),
        "manifest_sha256": file_sha(root / "manifest.json"),
        "scope": "Fixture and TP layout only; vLLM and operator execution not tested",
    }
    (root / "reference_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    check(args.root, args.reference)
