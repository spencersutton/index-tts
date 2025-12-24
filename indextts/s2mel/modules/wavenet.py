from __future__ import annotations

from typing import cast, override

import torch
from torch import Tensor, nn

from indextts.s2mel.modules.commons import fused_add_tanh_sigmoid_multiply
from indextts.s2mel.modules.encodec import SConv1d
from indextts.util import patch_call


class WaveNet(nn.Module):
    res_skip_layers: nn.ModuleList[SConv1d]
    in_layers: nn.ModuleList[SConv1d]
    cond_layer: SConv1d
    drop: nn.Dropout
    hidden_channels: int
    n_layers: int

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0,
        causal: bool = False,
    ) -> None:
        super().__init__()

        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            self.cond_layer = SConv1d(
                gin_channels,
                2 * hidden_channels * n_layers,
                1,
            )

        for i in range(n_layers):
            dilation = cast(int, dilation_rate**i)
            in_layer = SConv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
            )
            self.in_layers.append(in_layer)

            # last one is not necessary
            res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels

            res_skip_layer = SConv1d(hidden_channels, res_skip_channels, 1)
            self.res_skip_layers.append(res_skip_layer)

    @override
    def forward(self, x: Tensor, x_mask: Tensor, g: Tensor | None = None) -> Tensor:
        output = torch.zeros_like(x)

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output += res_skip_acts[:, self.hidden_channels :, :]
            else:
                output += res_skip_acts
        return output * x_mask

    @patch_call(forward)
    def __call__(self) -> None: ...
