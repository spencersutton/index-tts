# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from collections import OrderedDict
from typing import Literal, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from indextts.util import patch_call


def _get_nonlinear(channels: int) -> nn.Sequential:
    return nn.Sequential(OrderedDict({"batchnorm": nn.BatchNorm1d(channels), "relu": nn.ReLU(inplace=True)}))


class _StatsPool(nn.Module):
    @override
    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=True)
        return torch.cat([mean, std], dim=-1)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _TDNNLayer(nn.Module):
    linear: nn.Conv1d
    nonlinear: nn.Sequential

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Conv1d(320, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.nonlinear = _get_nonlinear(128)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        return self.nonlinear(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _CAMLayer(nn.Module):
    linear_local: nn.Conv1d
    linear1: nn.Conv1d
    linear2: nn.Conv1d
    relu: nn.ReLU
    sigmoid: nn.Sigmoid

    def __init__(self, dilation: int) -> None:
        super().__init__()

        self.linear_local = nn.Conv1d(128, 32, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.linear1 = nn.Conv1d(128, 64, 1)
        self.linear2 = nn.Conv1d(64, 32, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    @override
    def forward(self, x: Tensor) -> Tensor:
        y = self.linear_local(x)
        pool = F.avg_pool1d(x, kernel_size=x.shape[-1], ceil_mode=True).repeat_interleave(x.shape[-1], dim=-1)[
            ..., : x.shape[-1]
        ]

        context = x.mean(-1, keepdim=True) + pool
        context = self.linear1(context)
        context = self.relu(context)
        context = self.linear2(context)
        m = self.sigmoid(context)
        return y * m

    @patch_call(forward)
    def __call__(self) -> None: ...


class _CAMDenseTDNNLayer(nn.Module):
    cam_layer: _CAMLayer
    linear1: nn.Conv1d
    nonlinear1: nn.Sequential
    nonlinear2: nn.Sequential

    def __init__(self, in_channels: int, dilation: int) -> None:
        super().__init__()

        self.cam_layer = _CAMLayer(dilation=dilation)
        self.linear1 = nn.Conv1d(in_channels, 128, 1, bias=False)
        self.nonlinear1 = _get_nonlinear(in_channels)
        self.nonlinear2 = _get_nonlinear(128)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.nonlinear1(x)
        x = self.linear1(x)
        x = self.nonlinear2(x)
        return self.cam_layer.__call__(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _CAMDenseTDNNBlock(nn.ModuleList):
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


class _TransitLayer(nn.Module):
    nonlinear: nn.Sequential
    linear: nn.Conv1d

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.nonlinear = _get_nonlinear(in_channels)
        self.linear = nn.Conv1d(in_channels, in_channels // 2, 1, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.nonlinear(x)
        return self.linear(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _DenseLayer(nn.Module):
    linear: nn.Conv1d
    nonlinear: nn.Sequential

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Conv1d(1024, 192, 1, bias=False)

        modules = OrderedDict({"batchnorm": nn.BatchNorm1d(192, affine=False)})
        self.nonlinear = nn.Sequential(modules)

    @override
    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        return self.nonlinear(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _BasicResBlock(nn.Module):
    bn1: nn.BatchNorm2d
    bn2: nn.BatchNorm2d
    conv1: nn.Conv2d
    conv2: nn.Conv2d
    shortcut: nn.Sequential

    def __init__(self, stride: Literal[1, 2] = 1) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=(stride, 1), padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=1, stride=(stride, 1), bias=False), nn.BatchNorm2d(32)
            )

    @override
    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _FCM(nn.Module):
    bn1: nn.BatchNorm2d
    bn2: nn.BatchNorm2d
    conv1: nn.Conv2d
    conv2: nn.Conv2d
    layer1: nn.Sequential
    layer2: nn.Sequential

    def __init__(self) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.layer1 = nn.Sequential(*[_BasicResBlock(x) for x in (2, 1)])
        self.layer2 = nn.Sequential(*[_BasicResBlock(x) for x in (2, 1)])

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        return out.reshape(shape[0], shape[1] * shape[2], shape[3])

    @patch_call(forward)
    def __call__(self) -> None: ...


class CAMPPlus(nn.Module):
    head: _FCM
    xvector: nn.Sequential

    def __init__(self) -> None:
        super().__init__()

        self.head = _FCM()
        self.xvector = nn.Sequential(
            OrderedDict({
                "tdnn": _TDNNLayer(),
                "block1": _CAMDenseTDNNBlock(num_layers=12, in_channels=128, dilation=1),
                "transit1": _TransitLayer(512),
                "block2": _CAMDenseTDNNBlock(num_layers=24, in_channels=256, dilation=2),
                "transit2": _TransitLayer(1024),
                "block3": _CAMDenseTDNNBlock(num_layers=16, in_channels=512, dilation=2),
                "transit3": _TransitLayer(1024),
                "out_nonlinear": _get_nonlinear(512),
                "stats": _StatsPool(),
                "dense": _DenseLayer(),
            })
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        return self.xvector(x)

    @patch_call(forward)
    def __call__(self) -> None: ...
