from typing import override

import torch
from torch import Tensor, nn

from indextts.s2mel.modules.encodec import SConv1d
from indextts.util import patch_call


class WaveNet(nn.Module):
    cond_layer: SConv1d
    in_layers: nn.ModuleList[SConv1d]
    res_skip_layers: nn.ModuleList[SConv1d]
    n_layers: int
    dim: int

    def __init__(self, dim: int, n_layers: int = 8, kernel_size: int = 5) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim

        self.cond_layer = SConv1d(dim, 2 * dim * n_layers, 1)
        layers = [SConv1d(dim, 2 * dim, kernel_size) for _ in range(n_layers)]
        self.in_layers = nn.ModuleList(layers)

        layers = [SConv1d(dim, 2 * dim, 1) for _ in range(n_layers - 1)]
        layers.append(SConv1d(dim, dim, 1))
        self.res_skip_layers = nn.ModuleList(layers)

    @override
    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        output = torch.zeros_like(x)

        g = self.cond_layer(g)

        for i in range(self.n_layers):
            offset = i * 2 * self.dim
            g_l = g[:, offset : offset + 2 * self.dim, :]

            x_in = self.in_layers[i].__call__(x)
            t_act_part, s_act_part = (x_in + g_l).split(self.dim, dim=1)
            acts = t_act_part.tanh() * s_act_part.sigmoid()

            res_skip_acts = self.res_skip_layers[i].__call__(acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.dim, :]
                x = x + res_acts
                output += res_skip_acts[:, self.dim :, :]
            else:
                output += res_skip_acts
        return output

    @patch_call(forward)
    def __call__(self) -> None: ...
