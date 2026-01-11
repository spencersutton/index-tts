# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Sequence
from typing import cast

import torch
from torch import nn


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(self):
        super().__init__()
        self.dwconv = nn.Conv1d(384, 384, kernel_size=7, padding=3, groups=384)  # depthwise conv
        self.norm = nn.LayerNorm(384, eps=1e-6)
        self.pwconv1 = nn.Linear(384, 2048)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2048, 384)
        self.gamma = nn.Parameter(1 / 12 * torch.ones(384))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.mT  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.mT  # (B, T, C) -> (B, C, T)

        return residual + x


class VocosBackbone(nn.Module):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization
    """

    def __init__(self):
        super().__init__()
        self.embed = nn.Conv1d(1024, 384, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(384, eps=1e-6)
        self.convnext = cast(Sequence[ConvNeXtBlock], nn.ModuleList([ConvNeXtBlock() for _ in range(12)]))
        self.final_layer_norm = nn.LayerNorm(384, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            assert m.bias is not None
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.norm(x.mT)
        x = x.mT
        for conv_block in self.convnext:
            x = conv_block(x)
        return self.final_layer_norm(x.mT)
