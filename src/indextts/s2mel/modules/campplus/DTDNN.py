# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from collections import OrderedDict

import torch
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
from indextts.s2mel.modules.constants import M_CHANNELS
from indextts.util import patch_call


class FCM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_planes = M_CHANNELS
        self.conv1 = nn.Conv2d(1, M_CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(M_CHANNELS)

        self.layer1 = nn.Sequential(*[BasicResBlock(x) for x in (2, 1)])
        self.layer2 = nn.Sequential(*[BasicResBlock(x) for x in (2, 1)])

        self.conv2 = nn.Conv2d(M_CHANNELS, M_CHANNELS, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(M_CHANNELS)
        self.out_channels = M_CHANNELS * 10

    def forward(self, x) -> torch.Tensor:
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
    def __init__(self) -> None:
        super().__init__()

        self.head = FCM()
        channels = self.head.out_channels

        modules: OrderedDict[str, nn.Module] = OrderedDict({"tdnn": TDNNLayer(channels)})
        self.xvector = nn.Sequential(modules)
        channels = 128
        for i, (num_layers, dilation) in enumerate(zip((12, 24, 16), (1, 2, 2))):
            block = CAMDenseTDNNBlock(num_layers=num_layers, in_channels=channels, dilation=dilation)
            self.xvector.add_module(f"block{i + 1}", block)
            channels += num_layers * 32
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

    def forward(self, x) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        return self.xvector(x)

    @patch_call(forward)
    def __call__(self) -> None: ...
