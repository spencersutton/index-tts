# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convolutional layers wrappers and utilities."""

import warnings

from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from indextts.util import patch_call


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.conv = weight_norm(nn.Conv1d(*args, **kwargs))
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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: str = "reflect",
        **kwargs,
    ) -> None:
        super().__init__()
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn(
                "SConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        self.conv = NormConv1d(
            in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias
        )
        self.pad_mode = pad_mode

    def forward(self, x) -> torch.Tensor:
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        # Asymmetric padding required for odd strides
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)

    @patch_call(forward)
    def __call__(self) -> None: ...
