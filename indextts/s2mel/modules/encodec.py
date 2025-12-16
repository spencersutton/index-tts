# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# pyright: reportPrivateUsage=false

"""Convolutional layers wrappers and utilities."""

import math
from collections.abc import Sequence
from typing import override

from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from indextts.util import patch_call


def pad1d(x: Tensor, paddings: Sequence[int], mode: str = "zero", value: float = 0.0) -> Tensor:
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (
        padding_left,
        padding_right,
    )
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

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
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
        self.pad_mode = "reflect"

    @override
    def forward(self, x: Tensor) -> Tensor:
        # B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]

        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride

        length = x.shape[-1]
        extra_padding = math.ceil(length / stride) * stride - length
        # Asymmetric padding required for odd strides
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right

        x = pad1d(
            x,
            (padding_left, padding_right + extra_padding),
            mode=self.pad_mode,
        )
        return self.conv(x)

    @patch_call(forward)
    def __call__(self) -> None: ...
