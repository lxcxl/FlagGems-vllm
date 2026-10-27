#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare and replay the 16 layer3 MegaPreAttention benchmark cases.

Test/benchmark utility only. Does not import or implement a production kernel.
"""

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

REFERENCE_COMMIT = "f060b50e6a9471af0f296bb44d5ac60c2750bcd7"
TOKENS = (4096, 8192, 16384, 32768)
SEED = 20261010


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def tensor_info(tensor):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "sha256": tensor_sha(tensor),
    }


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def save_tensors(root, relative, tensors, files, metadata=None):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    torch.save({"tensors": tensors, "metadata": metadata or {}}, target)
    files[relative] = {
        "bytes": target.stat().st_size,
        "sha256": file_sha(target),
        "tensors": {key: tensor_info(value) for key, value in tensors.items()},
    }
    return relative


def packed_weight(weights, dims, tp, rank):
    _, q_heads, kv_heads = dims
    qwidth = q_heads * 256 * 2 // tp
    kvwidth = max(1, kv_heads // tp) * 256
    kv_rank = rank // max(1, tp // kv_heads)
    kv_start = kv_rank * kvwidth
    return torch.cat(
        [
            weights["q_proj.weight"][rank * qwidth : (rank + 1) * qwidth],
            weights["k_proj.weight"][kv_start : kv_start + kvwidth],
            weights["v_proj.weight"][kv_start : kv_start + kvwidth],
        ]
    )


def prepare(root, device):
    if (root / "manifest.json").exists():
        raise FileExistsError("Dataset already prepared; use verify instead")
    extraction = json.loads((root / "extraction.json").read_text())
    if (root / "cases").exists():
        raise FileExistsError("Partial dataset exists; inspect before regenerating")
    files, models, cases = {}, {}, []
    for model, record in extraction["models"].items():
        weights = {}
        for name, info in record["tensors"].items():
            raw = bytearray((root / info["raw_file"]).read_bytes())
            assert hashlib.sha256(raw).hexdigest() == info["sha256"]
            tensor = (
                torch.frombuffer(raw, dtype=torch.bfloat16)
                .reshape(info["shape"])
                .clone()
            )
            assert torch.isfinite(tensor).all(), name
            weights[name] = tensor
        weight_hashes = {name: tensor_sha(value) for name, value in weights.items()}
        assert weight_hashes == {
            name: x["sha256"] for name, x in record["tensors"].items()
        }
        relative = f"{model}/layer3.pt"
        metadata = {
            "input_origin": "real checkpoint layer3 weights; persisted seeded synthetic hidden",
            "weight_sha256": weight_hashes,
            "source_directory": record["source_directory"],
            "config_sha256": record["config_sha256"],
            "index_sha256": record["index_sha256"],
            "model_dims": record["model_dims"],
            "layer": 3,
        }
        torch.save({"weights": weights, "metadata": metadata}, root / relative)
        files[relative] = {
            "bytes": (root / relative).stat().st_size,
            "sha256": file_sha(root / relative),
            "weights": {name: tensor_info(tensor) for name, tensor in weights.items()},
        }
        models[model] = {"fixture": relative, "model_dims": record["model_dims"]}
        hidden_size, q_heads, kv_heads = record["model_dims"]
        for tp in (2, 4):
            rank_weights = [
                tensor_info(packed_weight(weights, record["model_dims"], tp, rank))
                for rank in range(tp)
            ]
            for tokens in TOKENS:
                case_id = f"{model}_tp{tp}_t{tokens}"
                local_kv = max(1, kv_heads // tp)
                shape = [tokens // tp, hidden_size]
                case = {
                    "id": case_id,
                    "model": model,
                    "tp": tp,
                    "global_tokens": tokens,
                    "local_q_heads": q_heads // tp,
                    "local_kv_heads": local_kv,
                    "hidden_shape": shape,
                    "cache_shape": [(tokens + 255) // 128, local_kv, 128, 512],
                    "cache_initial_value": 0.375,
                    "shared_inputs": f"shared/t{tokens}.pt",
                    "ranks": [],
                    "benchmark_rounds": (
                        5
                        if model == "Qwen3.6-27B"
                        or tokens == 32768
                        or (tp == 4 and tokens == 4096)
                        else 1
                    ),
                }
                for rank in range(tp):
                    seed = SEED + rank + tokens
                    generator = torch.Generator(device=device).manual_seed(seed)
                    hidden = (
                        torch.randn(
                            shape,
                            dtype=torch.bfloat16,
                            device=device,
                            generator=generator,
                        )
                        * 0.05
                    ).cpu()
                    relative = f"cases/{case_id}/rank{rank}.pt"
                    save_tensors(
                        root,
                        relative,
                        {"hidden": hidden},
                        files,
                        {"seed": seed, "rank": rank, "case": case_id},
                    )
                    case["ranks"].append(
                        {
                            "rank": rank,
                            "seed": seed,
                            "inputs": relative,
                            "packed_weight": rank_weights[rank],
                        }
                    )
                cases.append(case)
                print("Prepared", case_id, flush=True)
    # Non-edge text positions and slot mapping, shared by all ranks/models.
    for tokens in TOKENS:
        base = torch.arange(tokens, dtype=torch.int64)
        save_tensors(
            root,
            f"shared/t{tokens}.pt",
            {"positions": base.repeat(3, 1), "slots": base.clone()},
            files,
        )
    # Plain, unscaled vLLM MRotaryEmbedding cache formula. Its historical
    # default enlarges constructor max_position_embeddings by a factor of four.
    # Only rows 0..32767 are accessed by these cases. Generate once on CPU so
    # both backends replay identical BF16 cos/sin rather than recomputing them.
    rotary_dim, rows = 64, 262144 * 4
    inv_freq = 1.0 / (
        1.0e7 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    angles = torch.outer(torch.arange(rows, dtype=torch.float32), inv_freq)
    rope = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    save_tensors(root, "shared/rope.pt", {"rope": rope}, files)
    manifest = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_commit": REFERENCE_COMMIT,
        "extraction_sha256": file_sha(root / "extraction.json"),
        "generation": {
            "torch": str(torch.__version__),
            "hip": torch.version.hip,
            "hidden_device": device,
            "hidden_device_name": (
                torch.cuda.get_device_name(device)
                if device.startswith("cuda")
                else "CPU"
            ),
            "seed_base": SEED,
            "seed_formula": "20261010 + rank + global_tokens",
            "hidden_formula": (
                "torch.randn(local_tokens, hidden_size, "
                "dtype=bfloat16, generator=device_generator) * 0.05"
            ),
            "replay_policy": (
                "Load persisted tensors on both backends; "
                "same seed does not guarantee CUDA/HIP bit identity"
            ),
        },
        "config": {
            "head_dim": 256,
            "rotary_dim": 64,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
            "is_neox_style": True,
            "eps": 1e-6,
            "cache_block_size": 128,
            "rope_base": 1e7,
            "rope_rows": rows,
            "rope_file": "shared/rope.pt",
            "edge": False,
        },
        "benchmark": {
            "dtype": "bfloat16",
            "cuda_graph": True,
            "warmup": 20,
            "samples_per_method": 100,
            "alternating": True,
            "rank_aggregation": "maximum of per-rank mean",
            "stability": "(P90-P10)/median <= 0.05",
        },
        "limitations": [
            "NVIDIA report did not publish checkpoint revision/hash or exact seed; "
            "original-input byte identity is unverified",
            "Report K=1024 is undefined; model hidden dimensions remain 5120/2048",
            "No kernel correctness or performance validation performed during data preparation",
            "vLLM is absent in this container; replay cache uses plain unscaled RoPE formula, "
            "with provenance URL recorded",
        ],
        "rope_source": (
            "https://github.com/vllm-project/vllm/blob/main/"
            "vllm/model_executor/layers/rotary_embedding/base.py"
        ),
        "models": models,
        "cases": cases,
        "files": files,
    }
    save_json(root / "manifest.json", manifest)
    with (root / "cases.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case",
                "tp",
                "global_tokens",
                "local_tokens",
                "hidden_size",
                "local_q_heads",
                "local_kv_heads",
                "packed_weight_rows",
                "rounds",
            ]
        )
        for case in cases:
            writer.writerow(
                [
                    case["id"],
                    case["tp"],
                    case["global_tokens"],
                    *case["hidden_shape"],
                    case["local_q_heads"],
                    case["local_kv_heads"],
                    case["ranks"][0]["packed_weight"]["shape"][0],
                    case["benchmark_rounds"],
                ]
            )


@dataclass(frozen=True)
class Config:
    num_q_heads: int
    num_kv_heads: int
    global_kv_heads: int
    head_dim: int = 256
    rotary_dim: int = 64
    mrope_section: tuple = (11, 11, 10)
    eps: float = 1e-6

    @property
    def q_size(self):
        return self.num_q_heads * self.head_dim

    @property
    def kv_size(self):
        return self.num_kv_heads * self.head_dim

    @property
    def packed_size(self):
        return 2 * (self.q_size + self.kv_size)


class Dataset:
    """Portable replay loader. Load/allocate before warmup or graph capture."""

    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if (
            file_sha(self.root / "extraction.json")
            != self.manifest["extraction_sha256"]
        ):
            raise ValueError("extraction.json hash mismatch")
        self.cases = {case["id"]: case for case in self.manifest["cases"]}
        self._checked = set()
        self._fixtures = {}
        self._shared = {}

    def _load(self, relative):
        path = self.root / relative
        info = self.manifest["files"][relative]
        if relative not in self._checked:
            if path.stat().st_size != info["bytes"] or file_sha(path) != info["sha256"]:
                raise ValueError(f"File hash mismatch: {relative}")
        value = torch.load(path, map_location="cpu", weights_only=True)
        if relative not in self._checked:
            key = "weights" if "weights" in value else "tensors"
            for name, tensor in value[key].items():
                if tensor_info(tensor) != info[key][name]:
                    raise ValueError(f"Tensor mismatch: {relative}:{name}")
            self._checked.add(relative)
        return value

    def load_rank(self, case_id, rank, device="cpu", group=None):
        case = self.cases[case_id]
        if not 0 <= rank < case["tp"]:
            raise ValueError("Rank outside case TP group")
        model = case["model"]
        model_info = self.manifest["models"][model]
        if model not in self._fixtures:
            self._fixtures[model] = self._load(model_info["fixture"])
        weights = self._fixtures[model]["weights"]
        rank_info = case["ranks"][rank]
        hidden = self._load(rank_info["inputs"])["tensors"]["hidden"]
        weight = packed_weight(weights, model_info["model_dims"], case["tp"], rank)
        if tensor_info(weight) != rank_info["packed_weight"]:
            raise ValueError("Packed weight mismatch")
        for name in (case["shared_inputs"], self.manifest["config"]["rope_file"]):
            if name not in self._shared:
                self._shared[name] = self._load(name)["tensors"]
        shared = self._shared[case["shared_inputs"]]
        config = Config(
            case["local_q_heads"], case["local_kv_heads"], model_info["model_dims"][2]
        )
        return SimpleNamespace(
            hidden=hidden.to(device),
            weight=weight.to(device),
            positions=shared["positions"].to(device),
            slots=shared["slots"].to(device),
            q_norm=weights["q_norm.weight"].to(device),
            k_norm=weights["k_norm.weight"].to(device),
            rope=self._shared[self.manifest["config"]["rope_file"]]["rope"].to(device),
            cache=torch.full(
                case["cache_shape"],
                case["cache_initial_value"],
                dtype=torch.bfloat16,
                device=device,
            ),
            scale=torch.ones(1, device=device),
            group=group,
            config=config,
        )


def verify(root):
    data = Dataset(root)
    assert len(data.cases) == 16
    assert (
        len({(c["model"], c["tp"], c["global_tokens"]) for c in data.cases.values()})
        == 16
    )
    checked = []
    for case in data.cases.values():
        for rank in range(case["tp"]):
            inputs = data.load_rank(case["id"], rank)
            assert list(inputs.hidden.shape) == case["hidden_shape"]
            assert (
                inputs.hidden.dtype == torch.bfloat16 and inputs.hidden.is_contiguous()
            )
            assert (
                inputs.weight.is_contiguous() and inputs.weight.dtype == torch.bfloat16
            )
            assert inputs.weight.shape[0] == inputs.config.packed_size
            assert (
                torch.isfinite(inputs.hidden).all()
                and torch.isfinite(inputs.weight).all()
            )
            expected = torch.arange(case["global_tokens"], dtype=torch.int64)
            assert torch.equal(inputs.positions, expected.repeat(3, 1))
            assert torch.equal(inputs.slots, expected)
            assert list(inputs.cache.shape) == case["cache_shape"]
            assert inputs.cache[0, 0, 0, 0].item() == 0.375
            assert inputs.rope.shape == (1048576, 64)
            del inputs
        checked.append(case["id"])
        print("Verified", case["id"], flush=True)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": file_sha(Path(root) / "manifest.json"),
        "cases": len(checked),
        "rank_inputs": 48,
        "files_checked": len(data._checked),
        "status": "pass",
        "checks": [
            "file and tensor SHA256",
            "real weights match extracted BF16 bytes",
            "all ranks shape/dtype/finite",
            "TP Q/gate slicing and KV replication packed hashes",
            "positions/slots/cache contract",
            "weights_only torch.load replay",
        ],
        "operator_execution": False,
    }
    save_json(Path(root) / "validation.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Generator device; replay always uses persisted tensors",
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.action == "prepare":
        prepare(args.root, args.device)
    else:
        verify(args.root)
