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

"""Run reproducible full-EP MetaX vLLM 0.24 versus MegaMoE.

Prerequisites: matching MACA/FlagTree, torch, vllm-metax 0.24 and mcoplib 0.4.9
in the selected Python environment; MPI and the caller-owned host/device bridge
(python -m tools.megamoe.prepare build). Source its generated env.sh first.

The external canonical dataset contains manifest.json plus shared/expert tensor
files. Shared keys: hidden_states_bf16, input_int8, input_scales, topk_idx,
topk_weights. Expert keys: l1_weight_int8, l2_weight_int8, l1_weight_scale,
l2_weight_scale. The manifest supplies shape, file names and SHA256 digests.
The separate model config directory is consumed by vLLM EngineArgs.

Example:
  python -m tools.megamoe --data DATA --model MODEL --host-lib HOST.so
      --devices 0,1,2,3,4,5,6,7 --eps 2 4 8 --mode both --output NEW_DIR

Default: 53 real token shapes, 20 warmups, 100 completed CUDA-event samples,
5 repeats, max-rank mean then mean of 5. Caller setup, JIT, Graph capture and
initial quantization are outside timing. Eager events include the full native
EP call and its internal allocation/workspace requests; Graph events cover the
full replay, with allocations performed during capture. Graph mode
checks remote input/route mutations and measured output against eager.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from tools.megamoe import summarize

ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, *, name, output, env, timeout):
    (output / f"{name}.command.json").write_text(
        json.dumps(
            dict(
                command=command,
                devices=env["CUDA_VISIBLE_DEVICES"],
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"START {output.name}/{name}", flush=True)
    with (output / f"{name}.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            rc = 124
    (output / f"{name}.exit").write_text(f"{rc}\n")
    if rc:
        raise RuntimeError(f"{name} exited {rc}; see {output / (name + '.log')}")
    print(f"PASS {output.name}/{name}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--host-lib", type=Path, default=os.environ.get("MEGAMOE_HOST_LIB")
    )
    parser.add_argument(
        "--scale-lut", type=Path, help="Generate natively in output dir if omitted"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--devices", required=True, help="Physical GPU indices, in rank order"
    )
    parser.add_argument(
        "--eps", type=int, nargs="+", choices=(2, 4, 8), default=[2, 4, 8]
    )
    parser.add_argument(
        "--tokens", default=",".join(map(str, summarize.CANONICAL_TOKENS))
    )
    parser.add_argument("--mode", choices=("eager", "graph", "both"), default="both")
    parser.add_argument(
        "--device-visibility",
        choices=("per-rank", "all"),
        default="per-rank",
        help="vLLM device visibility; physical rank assignment is unchanged",
    )
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--graph-profile", action="store_true")
    parser.add_argument(
        "--require-target",
        action="store_true",
        help="Exit nonzero if any selected EP/mode mean speedup is below 0.95",
    )
    args = parser.parse_args()
    if args.host_lib is None:
        parser.error("--host-lib or MEGAMOE_HOST_LIB is required")
    devices, eps = args.devices.split(","), sorted(set(args.eps))
    tokens = list(map(int, args.tokens.split(",")))
    if len(devices) < max(eps) or len(set(devices)) != len(devices):
        parser.error("--devices must provide enough distinct GPU indices")
    if (
        not tokens
        or len(set(tokens)) != len(tokens)
        or not set(tokens) <= set(summarize.CANONICAL_TOKENS)
    ):
        parser.error(
            "--tokens must be a nonempty distinct subset of the canonical 53 shapes"
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "tmp").mkdir()
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join(
            [str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH", "")]
        ),
        TMPDIR=str(output / "tmp"),
        TRITON_CACHE_DIR=str(output / "triton-cache"),
        DNN_VENDOR="metax",
        MXSHMEM_BOOTSTRAP="MPI",
        MXSHMEM_SYMMETRIC_SIZE="2G",
        MEGAMOE_DATA_VERIFY_SHA256="1",
        USE_PRECOMPILED_KERNEL="1",
        USE_VLLM_TRITON_EXPERT="0",
        MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE="0",
        MACA_VLLM_ENABLE_MCTLASS_PYTHON_API="0",
        VLLM_PLUGINS="metax,metax_enhanced_customized,metax_enhanced_model",
    )
    env.setdefault("OMP_NUM_THREADS", "4")
    lut = args.scale_lut
    if lut is None:
        lut = output / "native_quant_scale_lut.bin"
        run(
            [
                sys.executable,
                "-m",
                "tools.megamoe.prepare",
                "lut",
                "--output",
                str(lut),
            ],
            name="prepare-lut",
            output=output,
            env=dict(env, CUDA_VISIBLE_DEVICES=devices[0]),
            timeout=args.timeout,
        )
    lut = lut.resolve()
    data, model = args.data.resolve(), args.model.resolve()
    dataset = json.loads((data / "manifest.json").read_text())
    sources = [
        *sorted(LOCAL.glob("*.py")),
        ROOT / "benchmark/test_megamoe.py",
        ROOT / "tests/test_megamoe.py",
        ROOT / "benchmark/megamoe_host.cpp",
        ROOT / "src/flaggems_vllm/ops/megamoe.py",
        ROOT / "src/flaggems_vllm/runtime/backend/_metax/ops/megamoe.py",
        *sorted((ROOT / "src/flaggems_vllm/runtime/backend/_metax/mega").rglob("*.py")),
        args.host_lib.resolve(),
        lut,
        data / "manifest.json",
        Path(env["MEGAMOE_DEVICE_LIB"]),
        Path(env["MXSHMEM_DEVICE_LIB"]),
    ]
    frozen = {str(p): digest(p) for p in sources}
    modes = ["eager", "graph"] if args.mode == "both" else [args.mode]
    manifest = dict(
        source_sha256=frozen,
        eps=eps,
        tokens=tokens,
        devices=devices,
        modes=modes,
        shape=dataset["shape"],
        warmup=20,
        iterations=100,
        repeats=5,
        device_visibility=args.device_visibility,
        full_53_shape_ep248=(
            eps == [2, 4, 8] and tokens == list(summarize.CANONICAL_TOKENS)
        ),
        baseline="MetaX vLLM0.24 / mcoplib0.4.9 native W8A8 + full EP",
        initial_quantization_in_timing=False,
        graph_capture_in_timing=False,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = {mode: {} for mode in modes}
    for mode in modes:
        directory = output / mode
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for ep in eps:
        group_env = dict(env, CUDA_VISIBLE_DEVICES=",".join(devices[:ep]))
        for mode in modes:
            directory = output / mode
            references = directory / f"references-np{ep}"
            native_script = LOCAL / (
                "graph_baseline.py" if mode == "graph" else "baseline.py"
            )
            baseline = [
                sys.executable,
                "-u",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node",
                str(ep),
            ]
            if args.device_visibility == "per-rank":
                baseline.append(str(LOCAL / "rank_launcher.py"))
            baseline += [
                str(native_script),
                "--model",
                str(model),
                "--shared-data-dir",
                str(data),
                "--tokens-list",
                args.tokens,
                "--warmup",
                "20",
                "--iters",
                "100",
                "--repeats",
                "5",
                "--diff-tol",
                "0.005",
                "--capture-dir",
                str(references),
                "--json-output",
                str(directory / f"baseline-np{ep}.json"),
            ]
            maca = Path(env["MACA_PATH"])
            candidate = [
                str(maca / "ompi/bin/mpirun"),
                "--allow-run-as-root",
                "--mca",
                "btl",
                "self,tcp",
                "--mca",
                "plm",
                "isolated",
                "--bind-to",
                "none",
                "-np",
                str(ep),
                sys.executable,
                "-u",
            ]
            candidate += (
                [str(LOCAL / "graph_candidate.py")]
                if mode == "graph"
                else ["-m", "benchmark.test_megamoe"]
            )
            candidate += [
                "--host-lib",
                str(args.host_lib.resolve()),
                "--data",
                str(data),
                "--references",
                str(references),
                "--scale-lut",
                str(lut),
                "--tokens",
                args.tokens,
                "--check-native",
                "--benchmark",
                "--warmup",
                "20",
                "--iters",
                "100",
                "--repeats",
                "5",
            ]
            for kind, command in (("baseline", baseline), ("candidate", candidate)):
                if mode == "graph":
                    command += ["--graph-audit-dir", str(directory / f"audit-{kind}")]
                    if args.graph_profile:
                        command.append("--graph-profile")
                run(
                    command,
                    name=f"{kind}-np{ep}",
                    output=directory,
                    env=group_env,
                    timeout=args.timeout,
                )
            if any(digest(Path(p)) != h for p, h in frozen.items()):
                raise RuntimeError("Measured source/library/input manifest changed")
            summarize.TOKENS = tokens
            summarize.TIMING_PROTOCOL = (
                summarize.GRAPH_PROTOCOL
                if mode == "graph"
                else "pr774-isolated-events-5-mean-v1"
            )
            result = summarize.summarize_ep(directory, ep)
            summarize.write_report(directory, result)
            if not result["complete"]:
                raise RuntimeError(
                    f"{mode} EP{ep} failed strict validation; see summary-np{ep}.json"
                )
            results[mode][str(ep)] = result
            report = dict(
                complete=all(len(v) == len(eps) for v in results.values()),
                full_53_shape_ep248=manifest["full_53_shape_ep248"],
                target_pass=all(
                    r["target_pass"]
                    for group in results.values()
                    for r in group.values()
                ),
                results=results,
            )
            (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"{mode} EP{ep} mean={result['arithmetic_mean_speedup']:.6f}",
                flush=True,
            )
    if args.require_target and not report["target_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
