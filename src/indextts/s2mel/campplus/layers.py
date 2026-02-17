# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from collections import OrderedDict
from typing import Final, Literal, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from indextts.util import patch_call


def get_nonlinear(channels: int) -> nn.Sequential:
    modules: OrderedDict[str, nn.Module] = OrderedDict({
        "batchnorm": nn.BatchNorm1d(channels),
        "relu": nn.ReLU(inplace=True),
    })
    return nn.Sequential(modules)


class StatsPool(nn.Module):
    @override
    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=True)
        return torch.cat([mean, std], dim=-1)

    @patch_call(forward)
    def __call__(self) -> None: ...


class TDNNLayer(nn.Module):
    linear: nn.Conv1d
    nonlinear: nn.Sequential

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.linear = nn.Conv1d(in_channels, 128, 5, stride=2, padding=2, bias=False)
        self.nonlinear = get_nonlinear(128)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        return self.nonlinear(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _CAMLayer(nn.Module):
    linear_local: nn.Conv1d
    linear1: nn.Conv1d
    relu: nn.ReLU
    linear2: nn.Conv1d
    sigmoid: nn.Sigmoid

    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.linear_local = nn.Conv1d(128, 32, 3, padding=dilation, dilation=dilation, bias=False)
        self.linear1 = nn.Conv1d(128, 64, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(64, 32, 1)
        self.sigmoid = nn.Sigmoid()

    @override
    def forward(self, x: Tensor) -> Tensor:
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x: Tensor) -> Tensor:
        seg_len: Final = 100
        seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        return seg[..., : x.shape[-1]]

    @patch_call(forward)
    def __call__(self) -> None: ...


class _CAMDenseTDNNLayer(nn.Module):
    nonlinear1: nn.Sequential
    linear1: nn.Conv1d
    nonlinear2: nn.Sequential
    cam_layer: _CAMLayer

    def __init__(self, in_channels: int, dilation: int) -> None:
        super().__init__()
        self.nonlinear1 = get_nonlinear(in_channels)
        self.linear1 = nn.Conv1d(in_channels, 128, 1, bias=False)
        self.nonlinear2 = get_nonlinear(128)
        self.cam_layer = _CAMLayer(dilation=dilation)

    def bn_function(self, x: Tensor) -> Tensor:
        return self.linear1(self.nonlinear1(x))

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.bn_function(x)
        return self.cam_layer(self.nonlinear2(x))

    @patch_call(forward)
    def __call__(self) -> None: ...


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(self, num_layers: int, in_channels: int, dilation: int) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _CAMDenseTDNNLayer(in_channels=in_channels + i * 32, dilation=dilation)
            self.add_module(f"tdnnd{i + 1}", layer)

    @override
    def forward(self, x: Tensor) -> Tensor:
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x

    @patch_call(forward)
    def __call__(self) -> None: ...


class TransitLayer(nn.Module):
    nonlinear: nn.Sequential
    linear: nn.Conv1d

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        super().__init__()
        self.nonlinear = get_nonlinear(in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.nonlinear(x)
        return self.linear(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class DenseLayer(nn.Module):
    linear: nn.Conv1d
    nonlinear: nn.Sequential

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False) -> None:
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

        modules: OrderedDict[str, nn.Module] = OrderedDict({"batchnorm": nn.BatchNorm1d(out_channels, affine=False)})
        self.nonlinear = nn.Sequential(modules)

    @override
    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        return self.nonlinear(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class BasicResBlock(nn.Module):
    conv1: nn.Conv2d
    bn1: nn.BatchNorm2d
    conv2: nn.Conv2d
    bn2: nn.BatchNorm2d
    shortcut: nn.Sequential

    def __init__(self, stride: Literal[1, 2] = 1, m_channels: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(m_channels, m_channels, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(m_channels),
            )

    @override
    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

    @patch_call(forward)
    def __call__(self) -> None: ...
