import math
from typing import override

import torch
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from indextts.s2mel.gpt_fast.model import Transformer
from indextts.s2mel.wavenet import WaveNet
from indextts.util import patch_call


class _TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    freqs: Tensor
    mlp: nn.Sequential

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim // 2, dim), nn.SiLU(), nn.Linear(dim, dim))

        half = dim // 4
        self.freqs = nn.Buffer((-math.log(10000) * torch.arange(half) / half).exp())

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

    def __init__(self, dim: int = 512) -> None:
        super().__init__()

        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = weight_norm(nn.Linear(dim, dim))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

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

    def __init__(self, dim: int = 512, channels: int = 80, style_dim: int = 192, block_size: int = 2**14) -> None:
        super().__init__()

        self.cond_projection = nn.Linear(dim, dim)  # continuous content
        self.cond_x_merge_linear = nn.Linear(dim + channels * 2 + style_dim, dim)
        self.conv1 = nn.Linear(dim, dim)
        self.conv2 = nn.Conv1d(dim, channels, kernel_size=1)
        self.final_layer = _FinalLayer(dim=dim)
        self.input_pos = nn.Buffer(torch.arange(block_size))
        self.res_projection = nn.Linear(dim, dim)
        self.skip_linear = nn.Linear(dim + channels, dim)
        self.t_embedder = _TimestepEmbedder(dim=dim)
        self.t_embedder2 = _TimestepEmbedder(dim=dim)
        self.transformer = Transformer(dim=dim)
        self.wavenet = WaveNet(dim=dim)

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
