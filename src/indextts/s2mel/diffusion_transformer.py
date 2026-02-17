import math
from typing import Final, override

import torch
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from indextts.s2mel.modules.gpt_fast.model import Transformer
from indextts.s2mel.modules.wavenet import WaveNet
from indextts.util import patch_call

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

BLOCK_SIZE: Final = 16384
CFG_RATE: Final = 0.7
CHANNELS: Final = 80
DIFFUSION_STEPS: Final = 25
DIM: Final = 512
STYLE_DIM: Final = 192


class _TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    freqs: Tensor
    mlp: nn.Sequential

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(DIM // 2, DIM), nn.SiLU(), nn.Linear(DIM, DIM))

        half = DIM // 4
        self.freqs = nn.Buffer((-math.log(10000) * torch.arange(half).float() / half).exp())

    @override
    def forward(self, t: Tensor) -> Tensor:
        args = 1000 * t[:, None] * self.freqs[None]
        return self.mlp(torch.cat([args.cos(), args.sin()], dim=-1))

    @patch_call(forward)
    def __call__(self) -> None: ...


class _FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    adaLN_modulation: nn.Sequential
    linear: nn.Linear
    norm_final: nn.LayerNorm

    def __init__(self) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(DIM, elementwise_affine=False, eps=1e-6)
        self.linear = weight_norm(nn.Linear(DIM, DIM))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(DIM, 2 * DIM))

    @override
    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))

    @patch_call(forward)
    def __call__(self) -> None: ...


class DiT(nn.Module):
    cond_projection: nn.Linear
    cond_x_merge_linear: nn.Linear
    conv1: nn.Linear
    conv2: nn.Conv1d
    final_layer: _FinalLayer
    input_pos: Tensor
    res_projection: nn.Linear
    skip_linear: nn.Linear
    t_embedder: _TimestepEmbedder
    t_embedder2: _TimestepEmbedder
    transformer: Transformer
    wavenet: WaveNet

    def __init__(self) -> None:
        super().__init__()
        self.transformer = Transformer()
        self.cond_projection = nn.Linear(DIM, DIM)  # continuous content
        self.t_embedder = _TimestepEmbedder()
        self.input_pos = nn.Buffer(torch.arange(BLOCK_SIZE))
        self.t_embedder2 = _TimestepEmbedder()
        self.conv1 = nn.Linear(DIM, DIM)
        self.conv2 = nn.Conv1d(DIM, CHANNELS, kernel_size=1)
        self.wavenet = WaveNet()
        self.final_layer = _FinalLayer()
        self.res_projection = nn.Linear(DIM, DIM)
        self.skip_linear = nn.Linear(DIM + CHANNELS, DIM)
        self.cond_x_merge_linear = nn.Linear(DIM + CHANNELS * 2 + STYLE_DIM, DIM)

    @override
    def forward(self, x: Tensor, prompt_x: Tensor, t: Tensor, style: Tensor, cond: Tensor) -> Tensor:
        T = x.size(2)

        t1 = self.t_embedder.__call__(t)
        cond = self.cond_projection(cond)

        x = x.mT
        prompt_x = prompt_x.mT

        x_in = torch.cat([x, prompt_x, cond], dim=-1)
        x_in = torch.cat([x_in, style[:, None, :].repeat(1, T, 1)], dim=-1)
        x_in = self.cond_x_merge_linear(x_in)

        x_res = self.input_pos[: x_in.size(1)]
        x_res = self.transformer.__call__(x_in, t1.unsqueeze(1), x_res)
        x_res = self.skip_linear(torch.cat([x_res, x], dim=-1))

        t2 = self.t_embedder2.__call__(t).unsqueeze(2)
        x = self.conv1.__call__(x_res).mT
        x = self.wavenet.__call__(x, g=t2).mT + self.res_projection(x_res)
        x = self.final_layer.__call__(x, t1).mT
        return self.conv2(x)

    @patch_call(forward)
    def __call__(self) -> None: ...
