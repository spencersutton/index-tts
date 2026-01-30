import math
from typing import TYPE_CHECKING, override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from indextts.s2mel.modules.constants import BLOCK_SIZE, DIM
from indextts.s2mel.modules.gpt_fast.model import Transformer
from indextts.s2mel.modules.wavenet import WaveNet
from indextts.util import patch_call

STYLE_ENCODER_DIM = 192


def modulate(x: Float[Tensor, "b t d"], shift: Float[Tensor, "b d"], scale: Float[Tensor, "b d"]) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    if TYPE_CHECKING:
        freqs: Tensor = torch.empty(0)

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(DIM // 2, DIM), nn.SiLU(), nn.Linear(DIM, DIM))

        half = DIM // 4
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def timestep_embedding(self, t: Float[Tensor, "b"]) -> Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        args = 1000 * t[:, None].float() * self.freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    @override
    def forward(self, t: Float[Tensor, "b"]) -> Tensor:
        t_freq = self.timestep_embedding(t)
        return self.mlp(t_freq)

    @patch_call(forward)
    def __call__(self) -> None: ...


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(DIM, elementwise_affine=False, eps=1e-6)
        self.linear = weight_norm(nn.Linear(DIM, DIM))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(DIM, 2 * DIM))

    @override
    def forward(self, x: Float[Tensor, "b t d"], c: Float[Tensor, "b d"]) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class DiT(nn.Module):
    if TYPE_CHECKING:
        input_pos: Tensor = torch.empty(0)
    transformer: Transformer
    x_embedder: nn.Linear
    cond_projection: nn.Linear
    t_embedder: TimestepEmbedder
    t_embedder2: TimestepEmbedder
    conv1: nn.Linear
    conv2: nn.Conv1d
    wavenet: WaveNet
    final_layer: FinalLayer
    res_projection: nn.Linear
    skip_linear: nn.Linear
    cond_x_merge_linear: nn.Linear

    def __init__(self, dim: int, in_channels: int) -> None:
        super().__init__()
        self.transformer = Transformer()

        self.x_embedder = weight_norm(nn.Linear(in_channels, dim))
        self.cond_projection = nn.Linear(dim, dim)  # continuous content

        self.t_embedder = TimestepEmbedder()

        input_pos = torch.arange(BLOCK_SIZE)
        self.register_buffer("input_pos", input_pos)

        self.t_embedder2 = TimestepEmbedder()
        self.conv1 = nn.Linear(dim, dim)
        self.conv2 = nn.Conv1d(dim, in_channels, kernel_size=1)
        self.wavenet = WaveNet()
        self.final_layer = FinalLayer()
        # residual connection from tranformer output to final output
        self.res_projection = nn.Linear(dim, dim)

        self.skip_linear = nn.Linear(dim + in_channels, dim)

        self.cond_x_merge_linear = nn.Linear(dim + in_channels * 2 + STYLE_ENCODER_DIM, dim)

    @override
    def forward(
        self,
        x: Float[Tensor, "b c t"],
        prompt_x: Float[Tensor, "b c t"],
        x_lens: Int[Tensor, "b"],
        t: Float[Tensor, "b"],
        style: Float[Tensor, "b c"],
        cond: Float[Tensor, "b t c"],
    ) -> Tensor:
        T = x.size(2)

        t1 = self.t_embedder.__call__(t)
        cond = self.cond_projection(cond)

        x = x.mT
        prompt_x = prompt_x.mT

        x_in = torch.cat([x, prompt_x, cond], dim=-1)
        x_in = torch.cat([x_in, style[:, None, :].repeat(1, T, 1)], dim=-1)

        x_in = self.cond_x_merge_linear(x_in)

        input_pos = self.input_pos[: x_in.size(1)]
        x_res = self.transformer.__call__(x_in, t1.unsqueeze(1), input_pos)

        x_res = self.skip_linear(torch.cat([x_res, x], dim=-1))
        x = self.conv1.__call__(x_res)
        x = x.mT
        t2 = self.t_embedder2(t)

        x = self.wavenet.__call__(x, g=t2.unsqueeze(2)).mT + self.res_projection(x_res)
        x = self.final_layer.__call__(x, t1).mT

        return self.conv2(x)

    @patch_call(forward)
    def __call__(self) -> None: ...
