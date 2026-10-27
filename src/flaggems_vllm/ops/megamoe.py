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

"""Distributed W8A8 MoE; backend dispatch with caller-owned workspace."""


def _implementation():
    from flaggems_vllm.runtime import device

    if device.vendor_name != "metax":
        raise NotImplementedError(
            "Distributed MegaMoE is currently implemented for MetaX"
        )
    from importlib import import_module

    return import_module("flaggems_vllm.runtime.backend._metax.ops.megamoe")


def megamoe(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    w1_scale,
    w2_scale,
    *,
    workspace,
    a1_scale=None,
    expert_map=None,
    out=None,
):
    return _implementation().megamoe(
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
