# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from collections import OrderedDict
from typing import override

import torch.nn.functional as F
from torch import Tensor, nn

import indextts.s2mel.campplus.layers as layers
from indextts.util import patch_call


class _FCM(nn.Module):
    bn1: nn.BatchNorm2d
    bn2: nn.BatchNorm2d
    conv1: nn.Conv2d
    conv2: nn.Conv2d
    in_planes: int
    layer1: nn.Sequential
    layer2: nn.Sequential
    out_channels: int

    def __init__(self, m_channels: int = 32) -> None:
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = nn.Sequential(*[layers.BasicResBlock(x) for x in (2, 1)])
        self.layer2 = nn.Sequential(*[layers.BasicResBlock(x) for x in (2, 1)])

        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * 10

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        shape = out.shape
        return out.reshape(shape[0], shape[1] * shape[2], shape[3])

    @patch_call(forward)
    def __call__(self) -> None: ...


class CAMPPlus(nn.Module):
    head: _FCM
    xvector: nn.Sequential

    def __init__(self, style_dim: int = 192) -> None:
        super().__init__()

        self.head = _FCM()
        channels = self.head.out_channels

        modules = OrderedDict({"tdnn": layers.TDNNLayer(channels)})
        self.xvector = nn.Sequential(modules)
        channels = 128
        for i, (num_layers, dilation) in enumerate(zip((12, 24, 16), (1, 2, 2))):
            block = layers.CAMDenseTDNNBlock(num_layers=num_layers, in_channels=channels, dilation=dilation)
            self.xvector.add_module(f"block{i + 1}", block)
            channels += num_layers * layers.CAMDenseTDNNBlock.growth_rate
            self.xvector.add_module(f"transit{i + 1}", layers.TransitLayer(channels, channels // 2, bias=False))
            channels //= 2

        self.xvector.add_module("out_nonlinear", layers.get_nonlinear(channels))

        self.xvector.add_module("stats", layers.StatsPool())
        self.xvector.add_module("dense", layers.DenseLayer(channels * 2, style_dim))

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
