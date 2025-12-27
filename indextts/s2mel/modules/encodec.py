# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Convolutional layers wrappers and utilities."""

from typing import override

from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from indextts.util import patch_call


class Conv1dWrapper(nn.Module):
    """Wrapper around Conv1d"""

    conv: nn.Conv1d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
            )
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class SConv1d(nn.Module):
    """Conv1d with some builtin handling of asymmetric or causal padding and normalization."""

    conv: Conv1dWrapper

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = Conv1dWrapper(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        conv = self.conv.conv
        kernel_size = (conv.kernel_size[0] - 1) * conv.dilation[0] + 1  # effective kernel
        stride = conv.stride[0]

        padding_total = kernel_size - stride
        length = x.shape[-1]

        extra_padding = (-length) % stride  # same as ceil(length/stride)*stride - length
        padding_right_base = padding_total // 2
        padding_left = padding_total - padding_right_base
        padding_right = padding_right_base + extra_padding

        assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)

        max_pad = max(padding_left, padding_right)
        extra_pad = max(0, max_pad - length + 1)
        if extra_pad:
            x = F.pad(x, (0, extra_pad))

        x = F.pad(x, (padding_left, padding_right), "reflect")
        if extra_pad:
            x = x[..., :-extra_pad]

        return self.conv(x)

    @patch_call(forward)
    def __call__(self) -> None: ...
