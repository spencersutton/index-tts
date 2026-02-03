from collections.abc import Sequence
from typing import Final, override

import torch
from jaxtyping import Float
from torch import Tensor, nn

from indextts.constants import S2MEL_MODEL_DIM
from indextts.s2mel.modules.encodec import SConv1d
from indextts.util import patch_call


class WaveNet(nn.Module):
    cond_layer: SConv1d
    in_layers: Sequence[SConv1d]
    res_skip_layers: Sequence[SConv1d]
    n_layers: int
    dim: Final = S2MEL_MODEL_DIM

    def __init__(self) -> None:
        super().__init__()
        self.n_layers = 8

        # Per-layer conditioning chunk is 2*dim (split into tanh/sigmoid parts).
        cond_chunk = S2MEL_MODEL_DIM * 2
        self.cond_layer = SConv1d(self.n_layers * cond_chunk, 1)

        layers = [SConv1d(cond_chunk, 5) for _ in range(self.n_layers)]
        self.in_layers = nn.ModuleList(layers)  # pyright: ignore[reportAttributeAccessIssue]

        layers = [SConv1d(cond_chunk, 1) for _ in range(self.n_layers - 1)]
        layers.append(SConv1d(self.dim, 1))
        self.res_skip_layers = nn.ModuleList(layers)  # pyright: ignore[reportAttributeAccessIssue]

    @override
    def forward(self, x: Float[Tensor, "b c t"], g: Float[Tensor, "b c t"]) -> Tensor:
        output = torch.zeros_like(x)

        g = self.cond_layer(g)

        cond_chunk = S2MEL_MODEL_DIM * 2

        for i in range(self.n_layers):
            offset = i * cond_chunk
            g_l = g[:, offset : offset + cond_chunk, :]

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
