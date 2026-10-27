# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Backend-dispatched MegaPreAttention API (inference, no aten equivalent)."""


def megapreattention(*args, **kwargs):
    from flaggems_vllm.runtime import device

    if device.vendor_name != "hygon":
        raise NotImplementedError(
            "MegaPreAttention is currently exposed for Hygon only"
        )
    from flaggems_vllm.runtime.backend._hygon.ops.megapreattention import (
        megapreattention as implementation,
    )

    return implementation(*args, **kwargs)


def clear_megapreattention_cache():
    from flaggems_vllm.runtime import device

    if device.vendor_name != "hygon":
        raise NotImplementedError(
            "MegaPreAttention is currently exposed for Hygon only"
        )
    from flaggems_vllm.runtime.backend._hygon.ops.megapreattention import (
        clear_megapreattention_cache as cleanup,
    )

    cleanup()
