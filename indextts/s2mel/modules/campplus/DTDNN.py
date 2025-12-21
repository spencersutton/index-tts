# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

# Copied from: https://github.com/modelscope/3D-Speaker/blob/main/speakerlab/models/campplus/DTDNN.py
from __future__ import annotations

from collections import OrderedDict
from typing import override
from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor, nn

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

M_CHANNELS = 32


class FCM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_planes = M_CHANNELS
        self.conv1 = nn.Conv2d(1, M_CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(M_CHANNELS)

        self.layer1 = nn.Sequential(*(BasicResBlock(x) for x in (2, 1)))
        self.layer2 = self.layer1

        self.conv2 = nn.Conv2d(M_CHANNELS, M_CHANNELS, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(M_CHANNELS)
        self.out_channels = M_CHANNELS * 10

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
    def __init__(self) -> None:
        super().__init__()

        self.head = FCM()

        self.xvector = nn.Sequential(
            OrderedDict({
                "tdnn": TDNNLayer(),
                "block1": CAMDenseTDNNBlock(12, 128, 1),
                "transit1": TransitLayer(512),
                "block2": CAMDenseTDNNBlock(24, 256, 2),
                "transit2": TransitLayer(1024),
                "block3": CAMDenseTDNNBlock(16, 512, 2),
                "transit3": TransitLayer(1024),
                "out_nonlinear": get_nonlinear(512),
                "stats": StatsPool(),
                "dense": DenseLayer(),
            })
        )

        for m in self.modules():
            if not isinstance(m, (nn.Conv1d, nn.Linear)):
                continue

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
