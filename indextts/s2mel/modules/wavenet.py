from collections.abc import Sequence
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

from indextts.constants import DIM
from indextts.s2mel.modules.encodec import SConv1d
from indextts.util import patch_call


class WaveNet(nn.Module):
    cond_layer: SConv1d
    in_layers: Sequence[SConv1d]
    res_skip_layers: Sequence[SConv1d]
    n_layers: int

    def __init__(self) -> None:
        super().__init__()
        self.n_layers = 8

        self.cond_layer = SConv1d(8192, 1)
        layers = [SConv1d(1024, 5) for _ in range(8)]
        self.in_layers = nn.ModuleList(layers)  # pyright: ignore[reportAttributeAccessIssue]

        layers = [SConv1d(1024, 1) for _ in range(7)]
        layers.append(SConv1d(DIM, 1))
        self.res_skip_layers = nn.ModuleList(layers)  # pyright: ignore[reportAttributeAccessIssue]

    @override
    def forward(self, x: Float[Tensor, "b c t"], g: Float[Tensor, "b c t"]) -> Tensor:
        output = torch.zeros_like(x)

        g = self.cond_layer(g)

        for i in range(8):
            offset = i * 1024
            g_l = g[:, offset : offset + 1024, :]

            x_in = self.in_layers[i].__call__(x)
            t_act_part, s_act_part = (x_in + g_l).split(DIM, dim=1)
            acts = t_act_part.tanh() * s_act_part.sigmoid()

            res_skip_acts = self.res_skip_layers[i].__call__(acts)
            if i < 7:
                res_acts = res_skip_acts[:, :DIM, :]
                x = x + res_acts
                output += res_skip_acts[:, DIM:, :]
            else:
                output += res_skip_acts
        return output

    @patch_call(forward)
    def __call__(self) -> None: ...
