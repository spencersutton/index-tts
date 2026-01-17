import torch
from torch import Tensor, nn

from indextts.s2mel.modules.commons import fused_add_tanh_sigmoid_multiply
from indextts.s2mel.modules.constants import HIDDEN_DIM, NUM_LAYERS, P_DROPOUT
from indextts.s2mel.modules.encodec import SConv1d
from indextts.util import patch_call

KERNEL_SIZE = 5


class WaveNet(nn.Module):
    in_layers: nn.ModuleList
    res_skip_layers: nn.ModuleList
    drop: nn.Dropout
    cond_layer: SConv1d

    def __init__(self) -> None:
        super().__init__()

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(P_DROPOUT)
        self.cond_layer = SConv1d(HIDDEN_DIM, 2 * HIDDEN_DIM * NUM_LAYERS)

        for i in range(NUM_LAYERS):
            in_layer = SConv1d(HIDDEN_DIM, 2 * HIDDEN_DIM, KERNEL_SIZE)
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < NUM_LAYERS - 1:
                res_skip_channels = 2 * HIDDEN_DIM
            else:
                res_skip_channels = HIDDEN_DIM

            res_skip_layer = SConv1d(HIDDEN_DIM, res_skip_channels)
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x: Tensor, x_mask: Tensor, g: Tensor) -> Tensor:
        output = torch.zeros_like(x)

        g = self.cond_layer(g)

        for i in range(NUM_LAYERS):
            cond_offset = i * 2 * HIDDEN_DIM
            g_l = g[:, cond_offset : cond_offset + 2 * HIDDEN_DIM, :]

            x_in = self.in_layers[i](x)
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < NUM_LAYERS - 1:
                res_acts = res_skip_acts[:, :HIDDEN_DIM, :]
                x = (x + res_acts) * x_mask
                output += res_skip_acts[:, HIDDEN_DIM:, :]
            else:
                output += res_skip_acts
        return output * x_mask

    @patch_call(forward)
    def __call__(self) -> None: ...
