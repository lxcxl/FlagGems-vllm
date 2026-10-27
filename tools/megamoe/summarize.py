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

"""Summarize this acceptance run without modifying any raw measurements."""

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import re
import statistics
from pathlib import Path

TOKENS = [1, 2, 4, 8, 16, *range(24, 257, 8), *range(272, 513, 16), 2048, 16384]
EPS = (2, 4, 8)
CANONICAL_TOKENS = tuple(TOKENS)
GRAPH_PROTOCOL = "pr774-full-ep-both-cudagraph-5-mean-v2"
REPEATS = 5
TIMING_PROTOCOL = "pr774-isolated-events-5-mean-v1"
BAD_FIELDS = (
    "combine_bad",
    "reduction_bound_bad",
    "dispatch_metadata_bad",
    "dispatch_route_set_bad",
    "dispatch_expert_bad",
    "dispatch_token_bad",
    "dispatch_scale_bad",
    "dispatch_weight_bad",
    "native_q_bad",
    "native_sf_bad",
    "native_metadata_bad",
    "native_partial_bad",
)
NONFINITE_FIELDS = (
    "output_nonfinite",
    "native_sf_nonfinite",
    "native_partial_nonfinite",
)
BENCH = re.compile(
    r"BENCH_REPEAT\s+rank=(\d+)\s+tokens=(\d+)\s+repeat=(\d+)\s+mean_us=([0-9.eE+-]+)"
)


def number(value, positive=False):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (not positive or value > 0)
    )


def read_source(path, sources):
    if not path.is_file():
        sources[path.name] = {"exists": False}
        return None
    data = path.read_bytes()
    sources[path.name] = {
        "exists": True,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    return data.decode("utf-8", errors="replace")


def json_objects(text):
    """Accept MPI prefixes and multiple adjacent JSON objects on one line."""
    decoder = json.JSONDecoder()
    for line in text.splitlines():
        offset = 0
        while True:
            start = line.find("{", offset)
            if start < 0:
                break
            try:
                value, end = decoder.raw_decode(line, start)
            except ValueError:
                offset = start + 1
                continue
            offset = end
            if isinstance(value, dict):
                yield value


def validate_correctness(record):
    errors = []
    for key in (
        "input_hashes_match",
        "native_checked",
        "native_exact",
        "output_ok",
        "dispatch_ok",
    ):
        if record.get(key) is not True:
            errors.append(f"{key} is not true")
    if record.get("status") != "PASS":
        errors.append("status is not PASS")
    error = record.get("rel_l2")
    if not number(error) or not 0 <= error <= 0.005:
        errors.append("rel_l2 is missing, nonfinite, or outside [0,0.005]")
    for key in (*BAD_FIELDS, *NONFINITE_FIELDS):
        if not number(record.get(key)) or record[key] != 0:
            errors.append(f"{key} is missing or nonzero")
    for key, value in record.items():
        if key.endswith("_bad") and key not in BAD_FIELDS:
            if key == "input_hash_bad":
                if value != []:
                    errors.append("input_hash_bad is not empty")
            elif not number(value) or value != 0:
                errors.append(f"{key} is nonzero or nonnumeric")
        if key.endswith("_error") and value:
            errors.append(f"{key}: {value}")
    if record.get("dispatch_expected_routes") != record.get("dispatch_actual_routes"):
        errors.append("dispatch route counts differ")
    return errors


def validate_shape(record):
    errors = []
    if record.get("status") != "PASS":
        errors.append("status is not PASS")
    for key in ("input_hashes_match", "native_checked", "pre_passed", "post_passed"):
        if record.get(key) is not True:
            errors.append(f"{key} is not true")
    if record.get("input_hash_bad") != []:
        errors.append("input_hash_bad is missing or not empty")
    for key, value in record.items():
        if key.endswith("_bad") and key != "input_hash_bad":
            if not number(value) or value != 0:
                errors.append(f"{key} is nonzero or nonnumeric")
        if key.endswith("_error") and value:
            errors.append(f"{key}: {value}")
    return errors


def summarize_ep(directory, ep):
    sources, errors = {}, []
    baseline_by_token = collections.defaultdict(list)
    measurements = collections.defaultdict(list)
    records = collections.defaultdict(list)
    exits = {}
    for kind in ("baseline", "candidate"):
        raw = read_source(directory / f"{kind}-np{ep}.exit", sources)
        try:
            exits[kind] = int(raw.strip()) if raw is not None else None
        except ValueError:
            exits[kind] = None
            errors.append(f"{kind} exit file is not an integer")
        if exits[kind] not in (None, 0):
            errors.append(f"{kind} exited with {exits[kind]}")

    raw = read_source(directory / f"baseline-np{ep}.json", sources)
    baseline = []
    if raw is not None:
        try:
            baseline = json.loads(raw)
            if not isinstance(baseline, list):
                raise ValueError("expected a list of baseline records")
        except ValueError as error:
            errors.append(f"baseline JSON: {error}")
            baseline = []
    for record in baseline:
        if not isinstance(record, dict):
            errors.append("baseline contains a non-object record")
            continue
        token = record.get("tokens_per_rank")
        if token not in TOKENS or record.get("world_size") != ep:
            errors.append(
                f"baseline EP/shape mismatch: world={record.get('world_size')}, T={token}"
            )
            continue
        baseline_by_token[token].append(record)

    raw = read_source(directory / f"candidate-np{ep}.log", sources) or ""
    for match in BENCH.finditer(raw):
        rank, token, repeat = map(int, match.groups()[:3])
        if rank not in range(ep) or token not in TOKENS or repeat not in range(REPEATS):
            errors.append(
                f"unexpected BENCH_REPEAT rank={rank}, T={token}, repeat={repeat}"
            )
            continue
        try:
            mean = float(match[4]) / 1000.0
        except ValueError:
            mean = None
        measurements[token, rank, repeat].append(mean)
    for record in json_objects(raw):
        kind = record.get("kind")
        if kind not in ("correctness", "shape_summary", "benchmark"):
            continue
        token, rank = record.get("tokens"), record.get("rank")
        if token not in TOKENS or rank not in range(ep) or record.get("world") != ep:
            errors.append(
                f"candidate EP/shape/rank mismatch: {kind}, world={record.get('world')}, T={token}, rank={rank}"
            )
            continue
        phase = record.get("correctness_phase") if kind == "correctness" else kind
        if phase not in ("pre", "post", "shape_summary", "benchmark"):
            errors.append(f"unexpected correctness phase: {phase}")
            continue
        records[token, rank, phase].append(record)

    rows = []
    for token in TOKENS:
        row_errors, baseline_errors, timing_errors = [], [], []
        base_records = baseline_by_token[token]
        base_means = {}
        for record in base_records:
            repeat = record.get("repeat")
            if repeat not in range(REPEATS) or repeat in base_means:
                baseline_errors.append(f"invalid or duplicate repeat {repeat}")
                continue
            performance = record.get("performance") or {}
            if performance.get("timing_protocol") != TIMING_PROTOCOL:
                baseline_errors.append(
                    f"repeat {repeat}: baseline timing protocol mismatch"
                )
            mean = performance.get("primary_latency_ms")
            base_means[repeat] = mean if number(mean, positive=True) else None
            if not number(mean, positive=True):
                baseline_errors.append(
                    f"repeat {repeat}: latency is not positive finite"
                )
            correctness = record.get("correctness") or {}
            if correctness.get("pass") is not True:
                baseline_errors.append(f"repeat {repeat}: correctness.pass is not true")
            rel_l2 = correctness.get("relative_l2_error")
            if not number(rel_l2) or not 0 <= rel_l2 <= 0.005:
                baseline_errors.append(
                    f"repeat {repeat}: baseline rel_l2 outside [0,0.005]"
                )
            shape = json.loads((directory / "manifest.json").read_text())["shape"]
            dimensions = {
                "hidden_size": shape["hidden_size"],
                "intermediate_size": shape["moe_intermediate_size"],
                "num_experts": shape["num_experts"],
                "local_experts": shape["num_experts"] // ep,
                "top_k": shape["num_experts_per_tok"],
            }
            for key, expected in dimensions.items():
                if record.get(key) != expected:
                    baseline_errors.append(f"repeat {repeat}: {key} != {expected}")
            if performance.get("warmup") != 20 or performance.get("iterations") != 100:
                baseline_errors.append(
                    f"repeat {repeat}: expected warmup=20, iterations=100"
                )
            if record.get("first_input_quantization_in_timed_region") is not False:
                baseline_errors.append(
                    f"repeat {repeat}: initial quantization timing differs"
                )
            means = performance.get("per_rank_mean_ms")
            if (
                not isinstance(means, list)
                or len(means) != ep
                or not all(number(x, positive=True) for x in means)
            ):
                baseline_errors.append(
                    f"repeat {repeat}: per-rank means missing or invalid"
                )
            elif not number(mean) or not math.isclose(
                mean, max(means), rel_tol=1e-9, abs_tol=1e-12
            ):
                baseline_errors.append(
                    f"repeat {repeat}: primary latency != max rank mean"
                )
        if len(base_records) != REPEATS or set(base_means) != set(range(REPEATS)):
            baseline_errors.append(
                f"expected 5 unique repeats, found {len(base_records)} records"
            )

        rank_means = {}
        for rank in range(ep):
            for phase in ("pre", "post", "shape_summary"):
                items = records[token, rank, phase]
                if len(items) != 1:
                    row_errors.append(
                        f"rank {rank}: expected one {phase}, found {len(items)}"
                    )
                for record in items:
                    validator = (
                        validate_shape
                        if phase == "shape_summary"
                        else validate_correctness
                    )
                    row_errors.extend(
                        f"rank {rank} {phase}: {error}" for error in validator(record)
                    )
            for repeat in range(REPEATS):
                values = measurements[token, rank, repeat]
                if len(values) != 1 or not number(values[0], positive=True):
                    timing_errors.append(
                        f"rank {rank} repeat {repeat}: expected one positive finite measurement, found {values}"
                    )
                else:
                    rank_means[rank, repeat] = values[0]
            items = records[token, rank, "benchmark"]
            if len(items) != 1:
                timing_errors.append(
                    f"rank {rank}: expected one benchmark record, found {len(items)}"
                )
            for record in items:
                if record.get("timing_protocol") != TIMING_PROTOCOL:
                    timing_errors.append(
                        f"rank {rank}: candidate timing protocol mismatch"
                    )
                if (
                    record.get("warmup") != 20
                    or record.get("iters") != 100
                    or record.get("repeats") != REPEATS
                ):
                    timing_errors.append(
                        f"rank {rank}: benchmark warmup/iters/repeats differs"
                    )
                if record.get("correctness") != "PASS":
                    timing_errors.append(
                        f"rank {rank}: benchmark correctness is not PASS"
                    )
                reported = record.get("mean_us")
                if not isinstance(reported, list) or len(reported) != REPEATS:
                    timing_errors.append(f"rank {rank}: benchmark means missing")
                else:
                    for repeat, value in enumerate(reported):
                        measured = rank_means.get((rank, repeat))
                        if (
                            not number(value, positive=True)
                            or measured is None
                            or not math.isclose(
                                value / 1000.0, measured, rel_tol=1e-8, abs_tol=1e-10
                            )
                        ):
                            timing_errors.append(
                                f"rank {rank} repeat {repeat}: JSON/line means differ"
                            )
        base_values = [base_means.get(k) for k in range(REPEATS)]
        maxima = (
            [max(rank_means[r, k] for r in range(ep)) for k in range(REPEATS)]
            if not timing_errors
            else []
        )
        base = statistics.mean(base_values) if not baseline_errors else None
        candidate = statistics.mean(maxima) if maxima else None
        ratio = base / candidate if base is not None and candidate is not None else None
        rows.append(
            {
                "tokens_per_rank": token,
                "vllm_ms": base,
                "megamoe_ms": candidate,
                "speedup": ratio,
                "correctness": not row_errors,
                "baseline_valid": not baseline_errors,
                "timing_valid": not timing_errors,
                "vllm_repeat_max_ms": base_values,
                "megamoe_repeat_max_ms": maxima,
                "errors": {
                    "candidate_correctness": row_errors,
                    "candidate_timing": timing_errors,
                    "baseline": baseline_errors,
                },
            }
        )
    ratios = [row["speedup"] for row in rows if row["speedup"] is not None]
    if TIMING_PROTOCOL == GRAPH_PROTOCOL:
        errors.extend(validate_graph(directory, ep))
    all_rows_valid = all(
        row["correctness"] and row["baseline_valid"] and row["timing_valid"]
        for row in rows
    )
    complete = (
        all_rows_valid and not errors and all(value == 0 for value in exits.values())
    )
    mean = statistics.mean(ratios) if len(ratios) == len(TOKENS) else None
    target_pass = bool(complete and mean >= 0.95)
    finished = all(value is not None for value in exits.values())
    return {
        "ranks": ep,
        "expected_shapes": len(TOKENS),
        "correctness_passed_shapes": sum(row["correctness"] for row in rows),
        "baseline_valid_shapes": sum(row["baseline_valid"] for row in rows),
        "timing_valid_shapes": sum(row["timing_valid"] for row in rows),
        "paired_timing_shapes": len(ratios),
        "complete": complete,
        "target_pass": target_pass,
        "status": "PASS" if target_pass else "FAIL" if finished else "INCOMPLETE",
        "arithmetic_mean_speedup": mean,
        "observed_arithmetic_mean_speedup": statistics.mean(ratios) if ratios else None,
        "geometric_mean_speedup": (
            math.exp(statistics.mean(map(math.log, ratios)))
            if len(ratios) == len(TOKENS)
            else None
        ),
        "min_speedup": min(ratios) if len(ratios) == len(TOKENS) else None,
        "num_below_0_95": (
            sum(ratio < 0.95 for ratio in ratios)
            if len(ratios) == len(TOKENS)
            else None
        ),
        "total_latency_ratio": (
            sum(row["vllm_ms"] for row in rows) / sum(row["megamoe_ms"] for row in rows)
            if len(ratios) == len(TOKENS)
            else None
        ),
        "acceptance": (
            "Each EP: all 53 shapes correct; mean(vLLM_ms / MegaMoE_ms) >= 0.95. "
            "Latency: mean of 5 repeats of max-rank mean, 20 warmup, 100 iterations."
        ),
        "exit_codes": exits,
        "errors": errors,
        "sources": sources,
        "rows": rows,
    }


def validate_graph(directory, ep):
    """Require complete native communication and dynamic replay checks on all ranks."""
    errors = []
    phases = {
        "original_pre",
        "remote_input_changed",
        "remote_routes_changed",
        "original_restored_pre",
        "post_measure",
    }
    for kind in ("baseline", "candidate"):
        expected = [
            t for t in TOKENS for _ in range(REPEATS if kind == "baseline" else 1)
        ]
        for rank in range(ep):
            path = directory / f"audit-{kind}/np{ep}-rank{rank}.jsonl"
            try:
                records = [json.loads(line) for line in path.read_text().splitlines()]
                assert [r["tokens"] for r in records] == expected
                for r in records:
                    assert (
                        r["rank"] == rank and r["world"] == ep and r["capture_complete"]
                    )
                    assert r["timing_protocol"] == GRAPH_PROTOCOL
                    assert len(r["checks"]) == len(phases)
                    assert {c["phase"] for c in r["checks"]} == phases
                    assert all(
                        c["finite"] and c["bit_mismatches"] == 0 for c in r["checks"]
                    )
                    if kind == "baseline":
                        assert r["captured_collectives"] == {
                            "all_gather": 2,
                            "reduce_scatter": 1,
                        }
                    if rank == 0:
                        assert all(
                            c["changed_from_original"] > 0
                            for c in r["checks"]
                            if c["phase"].startswith("remote_")
                        )
            except (OSError, ValueError, KeyError, AssertionError) as error:
                errors.append(
                    f"Graph audit {kind} rank {rank}: {type(error).__name__}: {error}"
                )
    return errors


def write_text(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_report(directory, result):
    ep = result["ranks"]
    write_text(
        directory / f"summary-np{ep}.json",
        json.dumps(result, indent=2, allow_nan=False) + "\n",
    )
    text = io.StringIO()
    fields = [
        "tokens_per_rank",
        "vllm_ms",
        "megamoe_ms",
        "speedup",
        "correctness",
        "baseline_valid",
        "timing_valid",
    ]
    writer = csv.DictWriter(text, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: row[key] for key in fields} for row in result["rows"])
    write_text(directory / f"summary-np{ep}.csv", text.getvalue())


def main():
    global TOKENS, TIMING_PROTOCOL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--eps", type=int, nargs="+", choices=EPS)
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--tokens", default=",".join(map(str, CANONICAL_TOKENS)))
    args = parser.parse_args()
    TOKENS = list(map(int, args.tokens.split(",")))
    TIMING_PROTOCOL = (
        GRAPH_PROTOCOL if args.mode == "graph" else "pr774-isolated-events-5-mean-v1"
    )
    if not args.directory.is_dir():
        parser.error("result directory does not exist")
    eps = sorted(
        set(
            args.eps
            or [
                ep
                for ep in EPS
                if any(
                    (args.directory / name).exists()
                    for name in (
                        f"baseline-np{ep}.json",
                        f"candidate-np{ep}.log",
                        f"baseline-np{ep}.exit",
                        f"candidate-np{ep}.exit",
                    )
                )
            ]
        )
    )
    results = {}
    for ep in eps:
        result = summarize_ep(args.directory, ep)
        write_report(args.directory, result)
        results[str(ep)] = {
            key: value for key, value in result.items() if key != "rows"
        }
        mean = result["arithmetic_mean_speedup"]
        print(
            f"EP={ep} {result['status']} "
            f"correctness={result['correctness_passed_shapes']}/{len(TOKENS)} "
            f"timing={result['timing_valid_shapes']}/{len(TOKENS)} "
            f"baseline={result['baseline_valid_shapes']}/{len(TOKENS)} "
            f"mean={mean if mean is not None else 'pending'} "
            f"target_pass={result['target_pass']}"
        )
    matrix_complete = bool(eps) and all(item["complete"] for item in results.values())
    summary = {
        "required_eps": list(EPS),
        "mode": args.mode,
        "tokens": TOKENS,
        "full_53_shape_ep248": set(eps) == set(EPS)
        and tuple(TOKENS) == CANONICAL_TOKENS,
        "observed_eps": eps,
        "complete": matrix_complete,
        "target_pass": bool(
            matrix_complete and all(item["target_pass"] for item in results.values())
        ),
        "results": results,
    }
    write_text(
        args.directory / "summary.json",
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
    )
    print(
        f"MATRIX complete={summary['complete']} target_pass={summary['target_pass']} observed_eps={eps}"
    )


if __name__ == "__main__":
    main()
