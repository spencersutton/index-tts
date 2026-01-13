# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convolutional layers wrappers and utilities."""

from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from indextts.util import patch_call


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.conv: nn.Conv1d = weight_norm(nn.Conv1d(*args, **kwargs))
        self.norm = nn.Identity()

    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        return self.norm(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class SConv1d(nn.Module):
    """
    Conv1d layer with built-in handling of asymmetric padding and normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()

        self.conv = NormConv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] > 1:
            x = F.pad(x, (2, 2), "reflect")
        return self.conv(x)

    @patch_call(forward)
    def __call__(self) -> None: ...
