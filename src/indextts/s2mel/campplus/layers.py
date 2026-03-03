# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from collections import OrderedDict
from typing import ClassVar, Final, override

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float
from torch import Tensor, nn

from indextts.util import patch_call

# Number of time-steps used by _CAMLayer's segment-pooling branch.
# Each segment is avg-pooled with this kernel/stride, then repeated to
# match the original time dimension before being used as a context signal.
_CAM_SEG_POOL_LEN: Final = 100


def get_nonlinear(channels: int) -> nn.Sequential:
    modules: OrderedDict[str, nn.Module] = OrderedDict({
        "batchnorm": nn.BatchNorm1d(channels),
        "relu": nn.ReLU(inplace=True),
    })
    return nn.Sequential(modules)


class StatsPool(nn.Module):
    """Temporal statistics pooling: concatenates mean and standard deviation over the time axis.

    Reduces a ``(batch, channels, time)`` feature map to ``(batch, channels*2)`` by
    computing per-channel mean and unbiased std and concatenating them.  Used as the
    final temporal aggregation step before speaker-embedding projection.
    """

    @override
    @beartype
    def forward(self, x: Float[Tensor, "batch channels time"]) -> Float[Tensor, "batch channels_2x"]:
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=True)
        return torch.cat([mean, std], dim=-1)

    @patch_call(forward)
    def __call__(self) -> None: ...


class TDNNLayer(nn.Module):
    """Single TDNN (Time-Delay Neural Network) layer with stride-2 downsampling.

    Applies a 1-D convolution with kernel size 5 and stride 2 followed by
    batch-norm + ReLU.  Acts as the first feature-extraction stage that halves
    the time resolution of the input sequence.
    """

    linear: nn.Conv1d
    nonlinear: nn.Sequential

    def __init__(self, in_channels: int, out_channels: int = 128) -> None:
        super().__init__()

        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=False)
        self.nonlinear = get_nonlinear(out_channels)

    @override
    @beartype
    def forward(self, x: Float[Tensor, "batch in_channels time"]) -> Float[Tensor, "batch out_channels time"]:
        x = self.linear(x)
        return self.nonlinear(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _CAMLayer(nn.Module):
    """Context-Aware Module (CAM) layer with dilated convolution and global context gating.

    Combines a local dilated convolution branch with a global context branch (segment
    pooling + channel-wise attention) to produce a context-gated output.  The gating
    signal is a sigmoid-activated combination of the channel mean and segment-pooled
    context, modulating the local convolution output element-wise.
    """

    linear_local: nn.Conv1d
    linear1: nn.Conv1d
    linear2: nn.Conv1d
    relu: nn.ReLU
    sigmoid: nn.Sigmoid

    def __init__(self, dilation: int, in_channels: int = 128, out_channels: int = 32) -> None:
        super().__init__()
        self.linear_local = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False
        )
        self.linear1 = nn.Conv1d(in_channels, in_channels // 2, 1)
        self.linear2 = nn.Conv1d(in_channels // 2, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    @override
    @beartype
    def forward(self, x: Float[Tensor, "batch channels time"]) -> Float[Tensor, "batch out_channels time"]:
        seg_pooled = F.avg_pool1d(x, kernel_size=_CAM_SEG_POOL_LEN, ceil_mode=True)
        seg_pooled = seg_pooled.repeat_interleave(_CAM_SEG_POOL_LEN, dim=-1)[..., : x.shape[-1]]
        context = x.mean(-1, keepdim=True) + seg_pooled
        context = self.relu(self.linear1(context))
        y = self.linear_local(x)
        m = self.sigmoid(self.linear2(context))
        return y * m

    @patch_call(forward)
    def __call__(self) -> None: ...


class _CAMDenseTDNNLayer(nn.Module):
    """A single densely-connected CAM-TDNN layer used inside :class:`CAMDenseTDNNBlock`.

    Applies a bottleneck projection (``bn_function``) followed by batch-norm and a
    CAM layer.  The output is intended to be *concatenated* with the input by the
    enclosing block (dense connectivity pattern), growing the channel dimension by
    ``_CAMLayer``'s ``out_channels`` at each layer.
    """

    cam_layer: _CAMLayer
    linear1: nn.Conv1d
    nonlinear1: nn.Sequential
    nonlinear2: nn.Sequential

    def __init__(self, in_channels: int, dilation: int, bn_channels: int = 128) -> None:
        super().__init__()

        self.cam_layer = _CAMLayer(dilation=dilation)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear1 = get_nonlinear(in_channels)
        self.nonlinear2 = get_nonlinear(bn_channels)

    @beartype
    def bn_function(self, x: Float[Tensor, "batch in_channels time"]) -> Float[Tensor, "batch bn_channels time"]:
        return self.linear1(self.nonlinear1(x))

    @override
    @beartype
    def forward(self, x: Float[Tensor, "batch in_channels time"]) -> Float[Tensor, "batch cam_out_channels time"]:
        x = self.bn_function(x)
        return self.cam_layer(self.nonlinear2(x))

    @patch_call(forward)
    def __call__(self) -> None: ...


class CAMDenseTDNNBlock(nn.ModuleList):
    """Densely-connected block of :class:`_CAMDenseTDNNLayer` layers (DenseNet-style).

    Each layer receives the concatenation of all previous feature maps (dense
    connectivity).  The fixed ``growth_rate`` of 32 means every additional layer
    contributes 32 output channels, so the channel count after *k* layers is
    ``in_channels + k * growth_rate``.
    """

    growth_rate: ClassVar[int] = 32

    def __init__(self, num_layers: int, in_channels: int, dilation: int) -> None:
        super().__init__()

        for i in range(num_layers):
            layer = _CAMDenseTDNNLayer(in_channels=in_channels + i * self.growth_rate, dilation=dilation)
            self.add_module(f"tdnnd{i + 1}", layer)

    @override
    @beartype
    def forward(self, x: Float[Tensor, "batch channels time"]) -> Float[Tensor, "batch channels_out time"]:
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x

    @patch_call(forward)
    def __call__(self) -> None: ...


class TransitLayer(nn.Module):
    """Transition layer: batch-norm + ReLU followed by a 1×1 point-wise convolution.

    Used between dense blocks to project the accumulated channel dimension down to
    a fixed ``out_channels`` size before the next block.
    """

    linear: nn.Conv1d
    nonlinear: nn.Sequential

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(in_channels)

    @override
    @beartype
    def forward(self, x: Float[Tensor, "batch in_channels time"]) -> Float[Tensor, "batch out_channels time"]:
        x = self.nonlinear(x)
        return self.linear(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class DenseLayer(nn.Module):
    """Final dense projection layer (affine-free batch-norm + 1×1 conv).

    Applies a learnable 1×1 convolution without bias followed by a
    non-affine ``BatchNorm1d``.  Accepts either a 2-D ``(batch, channels)``
    or a 3-D ``(batch, channels, 1)`` input tensor.
    """

    linear: nn.Conv1d
    nonlinear: nn.Sequential

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False) -> None:
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

        modules = OrderedDict({"batchnorm": nn.BatchNorm1d(out_channels, affine=False)})
        self.nonlinear = nn.Sequential(modules)

    @override
    @beartype
    def forward(self, x: Float[Tensor, "batch in_channels"]) -> Float[Tensor, "batch out_channels"]:
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        return self.nonlinear(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class BasicResBlock(nn.Module):
    """2-D basic residual block for frequency-time feature extraction.

    Standard two-layer residual block operating on ``(batch, planes, freq, time)``
    feature maps.  Applies strided convolution along the frequency axis only
    (``stride=(stride, 1)``) while keeping the time dimension unchanged.  A
    1×1 shortcut projection is added when ``stride != 1`` to match spatial dims.
    """

    bn1: nn.BatchNorm2d
    bn2: nn.BatchNorm2d
    conv1: nn.Conv2d
    conv2: nn.Conv2d
    shortcut: nn.Sequential

    def __init__(self, stride: int, planes: int = 32) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(planes, planes, kernel_size=1, stride=(stride, 1), bias=False), nn.BatchNorm2d(planes)
            )

    @override
    @beartype
    def forward(self, x: Float[Tensor, "batch planes freq time"]) -> Float[Tensor, "batch planes freq_out time"]:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

    @patch_call(forward)
    def __call__(self) -> None: ...
