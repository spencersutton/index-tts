# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

# Copied from: https://github.com/modelscope/3D-Speaker/blob/main/speakerlab/models/campplus/layers.py
from __future__ import annotations

from collections import OrderedDict
from typing import override

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor, nn

from indextts.util import patch_call


def get_nonlinear(channels: int = 128) -> nn.Sequential[nn.Module]:
    return nn.Sequential(
        OrderedDict({
            "batchnorm": nn.BatchNorm1d(channels),
            "relu": nn.ReLU(inplace=True),
        })
    )


class StatsPool(nn.Module):
    @override
    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=True)
        return torch.cat([mean, std], dim=-1)

    @patch_call(forward)
    def __call__(self) -> None: ...


class TDNNLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Conv1d(320, 128, 5, stride=2, padding=2, bias=False)
        self.nonlinear = get_nonlinear()

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        return self.nonlinear(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class CAMLayer(nn.Module):
    def __init__(self, padding: int, dilation: int) -> None:
        super().__init__()
        self.linear_local = nn.Conv1d(
            128,
            32,
            3,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
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

    @patch_call(forward)
    def __call__(self) -> None: ...

    def seg_pooling(self, x: Tensor, seg_len: int = 100, stype: str = "avg") -> Tensor:
        if stype == "avg":
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            msg = "Wrong segment pooling type."
            raise ValueError(msg)
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        return seg[..., : x.shape[-1]]


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self, in_channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.nonlinear1 = get_nonlinear(in_channels)
        self.linear1 = nn.Conv1d(in_channels, 128, 1, bias=False)
        self.nonlinear2 = get_nonlinear()
        self.cam_layer = CAMLayer(padding=dilation, dilation=dilation)

    def bn_function(self, x: Tensor) -> Tensor:
        return self.linear1(self.nonlinear1(x))

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        return self.cam_layer(self.nonlinear2(x))

    @patch_call(forward)
    def __call__(self) -> None: ...


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(in_channels=in_channels + i * 32, dilation=dilation)
            self.add_module(f"tdnnd{i + 1}", layer)

    @override
    def forward(self, x: Tensor) -> Tensor:
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x

    @patch_call(forward)
    def __call__(self) -> None: ...


class TransitLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.nonlinear = get_nonlinear(in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.nonlinear(x)
        return self.linear(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class DenseLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Conv1d(1024, 192, 1, bias=False)
        self.nonlinear = nn.Sequential(OrderedDict([("batchnorm", nn.BatchNorm1d(192, affine=False))]))

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1) if len(x.shape) == 2 else self.linear(x)
        return self.nonlinear(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


PLANES = 32


class BasicResBlock(nn.Module):
    def __init__(self, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            PLANES,
            PLANES,
            kernel_size=3,
            stride=(stride, 1),
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(PLANES)
        self.conv2 = nn.Conv2d(PLANES, PLANES, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(PLANES)

        self.shortcut: nn.Sequential[nn.Conv2d | nn.BatchNorm2d] = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    PLANES,
                    PLANES,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(PLANES),
            )

    @override
    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

    @patch_call(forward)
    def __call__(self) -> None: ...
