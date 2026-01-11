import torch
from torch import nn
from torch.nn import functional as F

from indextts.s2mel.modules import commons
from indextts.s2mel.modules.constants import HIDDEN_DIM, NUM_LAYERS, P_DROPOUT
from indextts.s2mel.modules.encodec import SConv1d
from indextts.util import patch_call

KERNEL_SIZE = 5


class LayerNorm(nn.Module):
    def __init__(self, channels, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)

    @patch_call(forward)
    def __call__(self) -> None: ...


class WN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(P_DROPOUT)
        self.cond_layer = SConv1d(HIDDEN_DIM, 2 * HIDDEN_DIM * NUM_LAYERS, 1)

        for i in range(NUM_LAYERS):
            in_layer = SConv1d(HIDDEN_DIM, 2 * HIDDEN_DIM, KERNEL_SIZE, dilation=1, padding=2)
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < NUM_LAYERS - 1:
                res_skip_channels = 2 * HIDDEN_DIM
            else:
                res_skip_channels = HIDDEN_DIM

            res_skip_layer = SConv1d(HIDDEN_DIM, res_skip_channels, 1)
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(x)
        n_channels_tensor = torch.tensor([HIDDEN_DIM])

        g = self.cond_layer(g)

        for i in range(NUM_LAYERS):
            x_in = self.in_layers[i](x)
            cond_offset = i * 2 * HIDDEN_DIM
            g_l = g[:, cond_offset : cond_offset + 2 * HIDDEN_DIM, :]

            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
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
