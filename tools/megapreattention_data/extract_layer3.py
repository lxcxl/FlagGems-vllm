#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Host-side, stdlib-only extraction; never loads/copies a whole model shard."""

import argparse
import hashlib
import json
import math
import struct
from datetime import datetime, timezone
from pathlib import Path

MODELS = {
    "Qwen3.6-27B": (5120, 24, 4),
    "Qwen3.6-35B-A3B": (2048, 16, 2),
}
PREFIX = "model.language_model.layers.3.self_attn."


def sha(data):
    return hashlib.sha256(data).hexdigest()


def extract(model_root, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "models": {}}
    for model, (hidden, q_heads, kv_heads) in MODELS.items():
        source = model_root / model
        index_bytes = (source / "model.safetensors.index.json").read_bytes()
        config_bytes = (source / "config.json").read_bytes()
        index = json.loads(index_bytes)["weight_map"]
        config = json.loads(config_bytes)
        text_config = config.get("text_config", config)
        assert text_config["hidden_size"] == hidden
        assert text_config["num_attention_heads"] == q_heads
        assert text_config["num_key_value_heads"] == kv_heads
        assert text_config["head_dim"] == 256
        target = output / model
        target.mkdir()
        (target / "config.json").write_bytes(config_bytes)
        (target / "model.safetensors.index.json").write_bytes(index_bytes)
        record = {
            "source_directory": str(source),
            "resolved_directory": str(source.resolve()),
            "config_sha256": sha(config_bytes),
            "index_sha256": sha(index_bytes),
            "layer": 3,
            "model_dims": [hidden, q_heads, kv_heads],
            "text_config": text_config,
            "tensors": {},
        }
        shapes = {
            "q_proj.weight": [2 * q_heads * 256, hidden],
            "k_proj.weight": [kv_heads * 256, hidden],
            "v_proj.weight": [kv_heads * 256, hidden],
            "q_norm.weight": [256],
            "k_norm.weight": [256],
        }
        for name, shape in shapes.items():
            key = PREFIX + name
            shard = source / index[key]
            stat_before = shard.stat()
            with shard.open("rb") as stream:
                header_length = struct.unpack("<Q", stream.read(8))[0]
                assert 0 < header_length < 100_000_000
                header_bytes = stream.read(header_length)
                header = json.loads(header_bytes)
                entry = header[key]
                assert entry["dtype"] == "BF16" and entry["shape"] == shape
                start, end = entry["data_offsets"]
                assert 0 <= start < end
                assert end - start == math.prod(shape) * 2
                assert 8 + header_length + end <= stat_before.st_size
                stream.seek(8 + header_length + start)
                payload = stream.read(end - start)
                assert len(payload) == end - start
            stat_after = shard.stat()
            assert (stat_before.st_size, stat_before.st_mtime_ns) == (
                stat_after.st_size,
                stat_after.st_mtime_ns,
            ), "Source changed during extraction"
            filename = name + ".bf16.bin"
            (target / filename).write_bytes(payload)
            record["tensors"][name] = {
                "checkpoint_key": key,
                "shape": shape,
                "dtype": "bfloat16",
                "sha256": sha(payload),
                "raw_file": str(Path(model) / filename),
                "source_shard": str(shard),
                "source_shard_bytes": stat_before.st_size,
                "source_shard_mtime_ns": stat_before.st_mtime_ns,
                "source_header_sha256": sha(header_bytes),
                "source_payload_offset": 8 + header_length + start,
                "nbytes": len(payload),
            }
            print(model, name, shape, sha(payload), flush=True)
        # Recheck config/index to guard against a concurrent checkpoint update.
        assert (source / "config.json").read_bytes() == config_bytes
        assert (source / "model.safetensors.index.json").read_bytes() == index_bytes
        manifest["models"][model] = record
    (output / "extraction.json").write_text(json.dumps(manifest, indent=2) + "\n")


def verify_source(output):
    manifest = json.loads((output / "extraction.json").read_text())
    checked = 0
    for model, record in manifest["models"].items():
        source = Path(record["source_directory"])
        assert sha((source / "config.json").read_bytes()) == record["config_sha256"]
        index = source / "model.safetensors.index.json"
        assert sha(index.read_bytes()) == record["index_sha256"]
        for name, info in record["tensors"].items():
            with Path(info["source_shard"]).open("rb") as stream:
                stream.seek(info["source_payload_offset"])
                payload = stream.read(info["nbytes"])
            assert sha(payload) == info["sha256"], (model, name)
            assert sha((output / info["raw_file"]).read_bytes()) == info["sha256"]
            checked += 1
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "selected_checkpoint_tensors_rechecked": checked,
        "scope": "Selected layer3 payloads and config/index, not full checkpoint shards",
    }
    (output / "source_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("/public-flash/models"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-source", action="store_true")
    args = parser.parse_args()
    if args.verify_source:
        verify_source(args.output)
    else:
        extract(args.model_root, args.output)
