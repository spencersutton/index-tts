# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

# Copied from: https://github.com/modelscope/3D-Speaker/blob/main/speakerlab/models/campplus/DTDNN.py
from __future__ import annotations

from collections import OrderedDict
from typing import override

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
    def __init__(self, feat_dim: int = 80) -> None:
        super().__init__()
        self.in_planes = M_CHANNELS
        self.conv1 = nn.Conv2d(1, M_CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(M_CHANNELS)

        self.layer1 = self._make_layer()
        self.layer2 = self._make_layer()

        self.conv2 = nn.Conv2d(
            M_CHANNELS,
            M_CHANNELS,
            kernel_size=3,
            stride=(2, 1),
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(M_CHANNELS)
        self.out_channels = M_CHANNELS * 10

    def _make_layer(self) -> nn.Sequential[BasicResBlock]:
        layers = [BasicResBlock(stride) for stride in (2, 1)]
        return nn.Sequential(*layers)

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
    def __init__(
        self,
        feat_dim: int = 80,
        embedding_size: int = 512,
        growth_rate: int = 32,
        bn_size: int = 4,
        init_channels: int = 128,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = True,
    ) -> None:
        super().__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels

        layer = TDNNLayer(
            channels,
            init_channels,
            5,
            stride=2,
            dilation=1,
            padding=-1,
            config_str=config_str,
        )
        self.xvector = nn.Sequential(OrderedDict([("tdnn", layer)]))
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module(f"block{i + 1}", block)
            channels += num_layers * growth_rate
            self.xvector.add_module(
                f"transit{i + 1}",
                TransitLayer(channels, channels // 2, bias=False, config_str=config_str),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

        self.xvector.add_module("stats", StatsPool())
        self.xvector.add_module(
            "dense",
            DenseLayer(channels * 2, embedding_size, config_str="batchnorm_"),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    assert isinstance(m.bias, Tensor)
                    nn.init.zeros_(m.bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        return self.xvector(x)

    @patch_call(forward)
    def __call__(self) -> None: ...
