# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from collections import OrderedDict

import torch.nn.functional as F
from torch import nn

from indextts.s2mel.modules.campplus.layers import (
    BasicResBlock,
    CAMDenseTDNNBlock,
    DenseLayer,
    StatsPool,
    TDNNLayer,
    TransitLayer,
    get_nonlinear,
)
from indextts.util import patch_call


class FCM(nn.Module):
    def __init__(self, block=BasicResBlock, num_blocks=[2, 2], m_channels=32, feat_dim=80) -> None:
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=2)

        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        return out.reshape(shape[0], shape[1] * shape[2], shape[3])

    @patch_call(forward)
    def __call__(self): ...


class CAMPPlus(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.head = FCM(feat_dim=80)
        channels = self.head.out_channels

        modules: OrderedDict[str, nn.Module] = OrderedDict({"tdnn": TDNNLayer(channels)})
        self.xvector = nn.Sequential(modules)
        channels = 128
        for i, (num_layers, dilation) in enumerate(zip((12, 24, 16), (1, 2, 2))):
            block = CAMDenseTDNNBlock(num_layers=num_layers, in_channels=channels, dilation=dilation)
            self.xvector.add_module(f"block{i + 1}", block)
            channels = channels + num_layers * 32
            self.xvector.add_module(f"transit{i + 1}", TransitLayer(channels, channels // 2, bias=False))
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(channels))

        self.xvector.add_module("stats", StatsPool())
        self.xvector.add_module("dense", DenseLayer(channels * 2, 192))

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        return self.xvector(x)

    @patch_call(forward)
    def __call__(self): ...
