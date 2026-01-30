from collections.abc import Sequence
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

from indextts.s2mel.modules.constants import DIM, NUM_LAYERS, P_DROPOUT
from indextts.s2mel.modules.encodec import SConv1d
from indextts.util import patch_call

KERNEL_SIZE = 5


@torch.compile
def fused_add_tanh_sigmoid_multiply(input_a: Float[Tensor, "b c t"], input_b: Float[Tensor, "b c t"]) -> Tensor:
    in_act = input_a + input_b
    # use torch.split to avoid dynamic slicing
    t_act_part, s_act_part = torch.split(in_act, DIM, dim=1)
    t_act = torch.tanh(t_act_part)
    s_act = torch.sigmoid(s_act_part)
    return t_act * s_act


class WaveNet(nn.Module):
    cond_layer: SConv1d
    drop: nn.Dropout
    in_layers: Sequence[SConv1d]
    res_skip_layers: Sequence[SConv1d]

    def __init__(self) -> None:
        super().__init__()

        self.drop = nn.Dropout(P_DROPOUT)
        self.cond_layer = SConv1d(DIM, 2 * DIM * NUM_LAYERS, 1)

        layers = [SConv1d(DIM, 2 * DIM, KERNEL_SIZE) for _ in range(NUM_LAYERS)]
        self.in_layers = nn.ModuleList(layers)  # pyright: ignore[reportAttributeAccessIssue]

        layers = [SConv1d(DIM, 2 * DIM, 1) for _ in range(NUM_LAYERS - 1)]
        layers.append(SConv1d(DIM, DIM, 1))
        self.res_skip_layers = nn.ModuleList(layers)  # pyright: ignore[reportAttributeAccessIssue]

    @override
    def forward(self, x: Float[Tensor, "b c t"], x_mask: Float[Tensor, "b 1 t"], g: Float[Tensor, "b c t"]) -> Tensor:
        output = torch.zeros_like(x)

        g = self.cond_layer(g)

        for i in range(NUM_LAYERS):
            offset = i * 2 * DIM
            g_l = g[:, offset : offset + 2 * DIM, :]

            x_in = self.in_layers[i].__call__(x)
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i].__call__(acts)
            if i < NUM_LAYERS - 1:
                res_acts = res_skip_acts[:, :DIM, :]
                x = (x + res_acts) * x_mask
                output += res_skip_acts[:, DIM:, :]
            else:
                output += res_skip_acts
        return output * x_mask

    @patch_call(forward)
    def __call__(self) -> None: ...
