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

"""Build caller-owned SDK libraries or generate native BF16 scale LUT.

Examples:
  python -m tools.megamoe.prepare build --sdk SDK --arch xcore1000 --output BUILD
  python -m tools.megamoe.prepare lut --output BUILD/native_quant_scale_lut.bin
Generated files stay outside Git and outside every timed operator call.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def generate_lut(output, device=0):
    import torch
    from vllm import _custom_ops as ops

    torch.cuda.set_device(device)
    # Every nonnegative finite BF16 magnitude. Each row has exactly that amax.
    bits = torch.arange(0x7F80, dtype=torch.int32, device="cuda")
    values = (bits << 16).view(torch.float32).bfloat16()
    rows = values[:, None].expand(-1, 128).contiguous()
    _, scales, _ = ops.scaled_int8_quant(rows)
    raw = scales.flatten().view(torch.int32).cpu().numpy().tobytes()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    record = dict(
        format="metax-mcoplib-bf16-int8-scale-v1",
        entries=0x7F80,
        sha256=hashlib.sha256(raw).hexdigest(),
        source="vllm._custom_ops.scaled_int8_quant",
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "vllm", "vllm-metax", "mcoplib")
        },
    )
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return output


def build(args):
    sdk, output = args.sdk.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "benchmark/megamoe_host.cpp"
    host, device = output / "megamoe_host.so", output / "megamoe_sync.bc"
    commands = [
        [
            args.compiler,
            f"-offload-arch={args.arch}",
            "-x",
            "maca",
            "-shared",
            "-fPIC",
            "-fgpu-rdc",
            "-std=c++17",
            f"-I{sdk / 'include'}",
            f"-I{sdk / 'include/mxshmem'}",
            f"-I{sdk / 'ompi/include'}",
            f"-L{sdk / 'lib'}",
            f"-L{sdk / 'ompi/lib'}",
            str(source),
            "-lmxshmem_host",
            "-lmxshmem",
            "-lmpi",
            "-o",
            str(host),
        ],
        [
            args.compiler,
            "-x",
            "maca",
            "-maca-device-only",
            "-emit-llvm",
            "-c",
            "-O2",
            "-fgpu-rdc",
            "-DMEGAMOE_DEVICE_LIBRARY",
            str(source),
            "-o",
            str(device),
        ],
    ]
    for command in commands:
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True, timeout=600)
    variables = {
        "MACA_PATH": str(sdk),
        "MEGAMOE_HOST_LIB": str(host),
        "MEGAMOE_DEVICE_LIB": str(device),
        "MXSHMEM_DEVICE_LIB": str(sdk / "lib/libmxshmem_device.bc"),
        "DNN_VENDOR": "metax",
        "MXSHMEM_BOOTSTRAP": "MPI",
        "MXSHMEM_SYMMETRIC_SIZE": "2G",
        "OMP_NUM_THREADS": "4",
    }
    text = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in variables.items())
    text += f'\nexport PYTHONPATH={shlex.quote(str(ROOT) + os.pathsep + str(ROOT / "src"))}:"${{PYTHONPATH:-}}"\n'
    (output / "env.sh").write_text(text)
    (output / "build.json").write_text(
        json.dumps(
            dict(
                commands=commands,
                arch=args.arch,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"source {shlex.quote(str(output / 'env.sh'))}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    compile_parser = sub.add_parser(
        "build", help="Ahead-of-time mxSHMEM host and device adapters"
    )
    compile_parser.add_argument("--sdk", type=Path, default=os.environ.get("MACA_PATH"))
    compile_parser.add_argument(
        "--arch", required=True, help="Target offload architecture, e.g. xcore1000"
    )
    compile_parser.add_argument("--compiler", default="mxcc")
    compile_parser.add_argument("--output", type=Path, required=True)
    lut_parser = sub.add_parser(
        "lut", help="Generate all native BF16 scale bit patterns"
    )
    lut_parser.add_argument("--output", type=Path, required=True)
    lut_parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.command == "build":
        if args.sdk is None:
            parser.error("--sdk or MACA_PATH is required")
        build(args)
    else:
        generate_lut(args.output, args.device)


if __name__ == "__main__":
    main()
