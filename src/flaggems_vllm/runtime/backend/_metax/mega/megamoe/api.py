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

"""Public data contract for MetaX W8A8 MegaMoE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class MegaMoEContext:
    """Caller-owned communication buffers, scratch and fixed launch configuration.

    The caller initializes the transport, collectively allocates and zeroes the
    symmetric buffers, and builds contiguous device int64 peer-pointer tables.
    With C=capacity, K=top_k, H=hidden_size and R=world_size, each rank needs
    layout[0]*R input bytes, K*C*H*2 partial bytes and 3*R*4 signal bytes.
    layout is (rank_stride, input_offset, scale_offset, route_offset, weight_offset)
    in bytes; each input region starts with a token-count header.

    buffers contains state/epoch, route histogram/prefix, expert rank counts/
    totals, block tasks, row metadata, gathered INT8 input/scales/route weights,
    BF16 activations, INT8 activations and activation scales, in that order.
    For B=ceil(C*K*R/bm)+E and L=B*bm, these contiguous tensors have shapes:
    (32,), (1,), (R*ceil(C*K/64)*E,) twice, (E*R,), (E,), (B,2),
    (L,3), (L,H), (L,), (L,), (L,I), (L,I), (L,).
    State and epoch start at zero. Counts/metadata are int32, gathered/quantized
    data int8, activations BF16, and all scales/route weights FP32.
    local_output uses BF16 expert_partials with at least min(B,4*workers)*bm*H
    elements; otherwise expert_partials aliases the partial peer-pointer table.

    Every CTA must be concurrently resident: the caller validates workers against
    the compiled specialization's occupancy before launching or capturing a graph.
    Preserve the prepared dtype/strides/configuration when reusing this context;
    route values and token counts up to capacity may change. All ranks call in the
    same order. Keep allocations alive until all serial calls/graph replays finish.
    Allocation, transport setup, device queries and grid selection belong to the
    caller; this data contract performs none of them.

    extern_libs contains the caller-prepared fence adapter and matching mxSHMEM
    device library. Before any launch/capture, the caller loads the specialization
    and registers its module with mxshmemx_cumodule_init, after transport setup.
    Keep the registration alive through all calls/graph replays, then synchronize
    and unregister with mxshmemx_cumodule_finalize before ending the transport.
    """

    rank: int
    world_size: int
    capacity: int
    top_k: int
    model_shape: Tuple[int, int, int]  # E_local, H, I
    layout: Tuple[int, int, int, int, int]
    workers: int
    bm: int
    bn: int
    bk: int
    stages: int
    local_output: bool
    inputs: torch.Tensor
    partials: torch.Tensor
    signals: torch.Tensor
    buffers: Tuple[torch.Tensor, ...]
    expert_partials: torch.Tensor
    extern_libs: Tuple[Tuple[str, str], ...]
    lut: Optional[torch.Tensor] = None


def megamoe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    workspace: MegaMoEContext,
    a1_scale: Optional[torch.Tensor] = None,
    expert_map: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Launch one asynchronous distributed SwiGLU MoE step.

    hidden_states is INT8 [T,H] with per-token FP32 a1_scale, or BF16 quantized
    inside the kernel. INT8 weights use [E_local,2I,H] / [E_local,H,I] and
    per-channel FP32 scales. Routes/weights are [T,top_k]; -1 drops a route.
    expert_map optionally maps global expert IDs to local IDs, or -1.
    Output is BF16 [T,H]. Empty ranks still participate. The caller supplies and
    owns workspace; see MegaMoEContext for capacity, residency and lifetime rules.
    """
    from .kernel import launch_megamoe

    return launch_megamoe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        workspace=workspace,
        a1_scale=a1_scale,
        expert_map=expert_map,
        out=out,
    )


__all__ = ["MegaMoEContext", "megamoe"]
