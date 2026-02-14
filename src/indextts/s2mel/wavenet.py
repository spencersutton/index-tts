from typing import Final, override

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from indextts.util import patch_call

DIM: Final = 512
DIM_2: Final = 2 * DIM
KERNEL_SIZE: Final = 5
N_LAYERS: Final = 8


class _SConv1d(nn.Module):
    """
    Conv1d layer with built-in handling of asymmetric padding and normalization.
    """

    conv: nn.Conv1d
    kernel_size: int

    @staticmethod
    def _remap_weights(_module: object, state_dict: dict[str, object], *_args: object) -> None:
        for k in list(state_dict.keys()):
            new_k = k.replace("conv.conv", "conv")
            state_dict[new_k] = state_dict.pop(k)

    def __init__(self, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.register_load_state_dict_pre_hook(self._remap_weights)

        self.kernel_size = kernel_size
        self.conv = weight_norm(nn.Conv1d(DIM, out_channels, kernel_size))

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.kernel_size > 1:
            x = F.pad(x, (2, 2), "reflect")
        return self.conv(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class WaveNet(nn.Module):
    cond_layer: _SConv1d
    in_layers: nn.ModuleList[_SConv1d]
    res_skip_layers: nn.ModuleList[_SConv1d]

    def __init__(self) -> None:
        super().__init__()

        self.cond_layer = _SConv1d(DIM_2 * N_LAYERS, 1)
        layers = [_SConv1d(DIM_2, KERNEL_SIZE) for _ in range(N_LAYERS)]
        self.in_layers = nn.ModuleList(layers)

        layers = [_SConv1d(DIM_2, 1) for _ in range(N_LAYERS - 1)]
        layers.append(_SConv1d(DIM, 1))
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
