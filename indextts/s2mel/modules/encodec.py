# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convolutional layers wrappers and utilities."""

from typing import override

from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from indextts.util import patch_call


class SConv1d(nn.Module):
    """
    Conv1d layer with built-in handling of asymmetric padding and normalization.
    """

    kernel_size: int
    conv: nn.Conv1d

    @staticmethod
    def _remap_weights(_module: object, state_dict: dict[str, object], *_args: object) -> None:
        for k in list(state_dict.keys()):
            new_k = k.replace("conv.conv", "conv")
            state_dict[new_k] = state_dict.pop(k)

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.register_load_state_dict_pre_hook(self._remap_weights)

        self.kernel_size = kernel_size
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size))

    @override
    def forward(self, x: Float[Tensor, "b c t"]) -> Tensor:
        if self.kernel_size > 1:
            x = F.pad(x, (2, 2), "reflect")
        return self.conv(x)

    @patch_call(forward)
    def __call__(self) -> None: ...
