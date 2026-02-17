from typing import Final, override

import torch
from torch import Tensor, nn

from indextts.s2mel.modules.encodec import SConv1d
from indextts.util import patch_call

DIM: Final = 512
N_LAYERS: Final = 8
KERNEL_SIZE: Final = 5
DIM_2: Final = 2 * DIM


class WaveNet(nn.Module):
    cond_layer: SConv1d
    in_layers: nn.ModuleList[SConv1d]
    res_skip_layers: nn.ModuleList[SConv1d]

    def __init__(self) -> None:
        super().__init__()

        self.cond_layer = SConv1d(DIM_2 * N_LAYERS, 1)
        layers = [SConv1d(DIM_2, KERNEL_SIZE) for _ in range(N_LAYERS)]
        self.in_layers = nn.ModuleList(layers)

        layers = [SConv1d(DIM_2, 1) for _ in range(N_LAYERS - 1)]
        layers.append(SConv1d(DIM, 1))
        self.res_skip_layers = nn.ModuleList(layers)

    @override
    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        output = torch.zeros_like(x)

        g = self.cond_layer(g)

        for i in range(N_LAYERS):
            offset = i * DIM_2
            g_l = g[:, offset : offset + DIM_2, :]

            x_in = self.in_layers[i].__call__(x)
            t_act_part, s_act_part = (x_in + g_l).split(DIM, dim=1)
            acts = t_act_part.tanh() * s_act_part.sigmoid()

            res_skip_acts = self.res_skip_layers[i].__call__(acts)
            if i < N_LAYERS - 1:
                res_acts = res_skip_acts[:, :DIM, :]
                x = x + res_acts
                output += res_skip_acts[:, DIM:, :]
            else:
                output += res_skip_acts
        return output

    @patch_call(forward)
    def __call__(self) -> None: ...
