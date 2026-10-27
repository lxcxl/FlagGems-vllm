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

"""MetaX device synchronization and reciprocal intrinsics."""

import triton.language as tl
from triton.language import core


@core.extern
def fence_sys(_semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [],
        {(): ("flaggems_mxshmem_fence", tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def native_rcp(x, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x],
        {(tl.float32,): ("llvm.mxc.rcp.f32", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )
