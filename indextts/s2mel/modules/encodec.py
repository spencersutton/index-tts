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


class _NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    conv: nn.Conv1d
    norm: nn.Identity

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()

        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size))
        self.norm = nn.Identity()

    @override
    def forward(self, x: Float[Tensor, "b c t"]) -> Tensor:
        return self.norm(self.conv(x))

    @patch_call(forward)
    def __call__(self) -> None: ...


class SConv1d(nn.Module):
    """
    Conv1d layer with built-in handling of asymmetric padding and normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = _NormConv1d(in_channels, out_channels, kernel_size)

    @override
    def forward(self, x: Float[Tensor, "b c t"]) -> Tensor:
        if self.kernel_size > 1:
            x = F.pad(x, [2, 2], "reflect")
        return self.conv(x)

    @patch_call(forward)
    def __call__(self) -> None: ...
