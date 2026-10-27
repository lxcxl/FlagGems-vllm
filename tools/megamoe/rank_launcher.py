# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Select one assigned physical GPU before Torch/vLLM creates any context."""

import os
import runpy
import sys

local_rank = int(os.environ["LOCAL_RANK"])
physical = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[local_rank]
os.environ["CUDA_VISIBLE_DEVICES"] = physical
os.environ["LOCAL_RANK"] = "0"
os.environ["MEGAMOE_PHYSICAL_GPU"] = physical
print(
    f"RANK_DEVICE rank={os.environ['RANK']} physical_gpu={physical} local_device=0",
    flush=True,
)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
