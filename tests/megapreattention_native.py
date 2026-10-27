# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Hygon paired full-window benchmark.

Baseline uses RCCL AllGather + rocBLAS matmul + torch.compile postprocessing.
It is explicitly a Hygon local baseline, not NV's overlapping SymmMem/vLLM
baseline, which this environment cannot execute. No NV speedups are inferred.
"""
import torch
import torch.distributed as dist

from .megapreattention_utils import Output


class NativeWindow:
    def __init__(self, inputs):
        self.config = inputs.config
        self.post = torch.compile(self._post, fullgraph=True, dynamic=False)
        self.evidence = {
            "communication": "RCCL all_gather_into_tensor",
            "projection": "torch.matmul / rocBLAS",
            "post": "torch.compile fullgraph Gemma RMSNorm / NeoX interleaved MRoPE / gate / cache",
            "overlaps_communication_gemm": False,
            "comparable_to_nv_native_baseline": False,
            "torch_version": torch.__version__,
        }

    def _post(self, qkv, qnorm, knorm, positions, rope, cache, slots):
        c = self.config
        t = positions.shape[1]
        qgate, key, value = qkv.split((2 * c.q_size, c.kv_size, c.kv_size), dim=-1)
        query, gate = qgate.view(t, c.num_q_heads, 512).chunk(2, dim=-1)
        freq = torch.arange(32, device=qkv.device)
        axis = torch.where(
            (freq % 3 == 1) & (freq < 3 * c.mrope_section[1]),
            1,
            torch.where((freq % 3 == 2) & (freq < 3 * c.mrope_section[2]), 2, 0),
        )
        pos = positions[axis].t()
        co, si = rope[pos, freq].float().unsqueeze(1), rope[
            pos, freq + 32
        ].float().unsqueeze(1)

        def norm_rope(x, w):
            x = x.float()
            x = (
                (
                    x
                    * torch.rsqrt(x.square().mean(-1, keepdim=True) + c.eps)
                    * (w.float() + 1)
                )
                .bfloat16()
                .float()
            )
            a, b = x[..., :64].chunk(2, dim=-1)
            return torch.cat(
                (a * co - b * si, b * co + a * si, x[..., 64:]), dim=-1
            ).bfloat16()

        query = norm_rope(query, qnorm).reshape(t, c.q_size)
        key = norm_rope(key.view(t, c.num_kv_heads, 256), knorm)
        # Benchmark active set has unique nonnegative slots. The independent
        # oracle separately validates candidate negative-slot/strided handling.
        block, offset = slots // cache.shape[2], slots % cache.shape[2]
        cache[block, :, offset, :256] = key
        cache[block, :, offset, 256:] = value.view(t, c.num_kv_heads, 256)
        return query, key.reshape(t, c.kv_size), value, gate

    def __call__(self, x):
        h = torch.empty(
            (x.positions.shape[1], x.hidden.shape[1]),
            device=x.hidden.device,
            dtype=x.hidden.dtype,
        )
        dist.all_gather_into_tensor(h, x.hidden, group=x.group)
        qkv = h @ x.weight.t()
        return Output(
            *self.post(qkv, x.q_norm, x.k_norm, x.positions, x.rope, x.cache, x.slots),
            workspace=(h, qkv),
        )
