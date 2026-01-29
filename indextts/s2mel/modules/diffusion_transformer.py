import math
from typing import TYPE_CHECKING, override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from indextts.s2mel.modules.constants import BLOCK_SIZE, DIM, IN_CHANNELS
from indextts.s2mel.modules.gpt_fast.model import Transformer
from indextts.s2mel.modules.wavenet import WaveNet
from indextts.util import patch_call

STYLE_ENCODER_DIM = 192


def sequence_mask(length: Int[Tensor, "b"] | int, max_length: int | None = None) -> Tensor:
    length = torch.as_tensor(length)
    if max_length is None:
        max_length = int(length.max())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


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

    def __init__(self) -> None:
        super().__init__()
        self.transformer = Transformer()

        self.x_embedder = weight_norm(nn.Linear(IN_CHANNELS, DIM))

        self.cond_projection = nn.Linear(DIM, DIM)  # continuous content

        self.t_embedder = TimestepEmbedder()

        input_pos = torch.arange(BLOCK_SIZE)
        self.register_buffer("input_pos", input_pos)

        self.t_embedder2 = TimestepEmbedder()
        self.conv1 = nn.Linear(DIM, DIM)
        self.conv2 = nn.Conv1d(DIM, IN_CHANNELS, kernel_size=1)
        self.wavenet = WaveNet()
        self.final_layer = FinalLayer()
        # residual connection from tranformer output to final output
        self.res_projection = nn.Linear(DIM, DIM)

        self.skip_linear = nn.Linear(DIM + IN_CHANNELS, DIM)

        self.cond_x_merge_linear = nn.Linear(DIM + IN_CHANNELS * 2 + STYLE_ENCODER_DIM, DIM)

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
        """
        x (Tensor): random noise
        prompt_x (Tensor): reference mel + zero mel
            shape: (batch_size, 80, 795+1068)
        x_lens (Tensor): mel frames output
            shape: (batch_size, mel_timesteps)
        t (Tensor): radshape:
            shape: (batch_size)
        style (Tensor): reference global style
            shape: (batch_size, 192)
        cond (Tensor): semantic info of reference audio and altered audio
            shape: (batch_size, mel_timesteps(795+1069), 512)

        """
        T = x.size(2)

        t1 = self.t_embedder.__call__(t)  # (N, D) # t1 [2, 512]
        cond = self.cond_projection.__call__(cond)  # cond [2,1863,512]->[2,1863,512]

        x = x.mT  # [2,1863,80]
        prompt_x = prompt_x.mT  # [2,1863,80]

        x_in = torch.cat([x, prompt_x, cond], dim=-1)  # 80+80+512=672 [2, 1863, 672]
        x_in = torch.cat([x_in, style[:, None, :].repeat(1, T, 1)], dim=-1)  # [2, 1863, 864]

        x_in = self.cond_x_merge_linear.__call__(x_in)  # (N, T, D) [2, 1863, 512]

        x_mask = (
            sequence_mask(x_lens, max_length=x_in.size(1)).to(x.device).unsqueeze(1)
        )  # torch.Size([1, 1, 1863])True
        input_pos = self.input_pos[: x_in.size(1)]  # (T,) range（0，1863）
        x_mask_expanded = x_mask[:, None, :].repeat(1, 1, x_in.size(1), 1)  # torch.Size([1, 1, 1863, 1863]
        x_res = self.transformer.__call__(x_in, t1.unsqueeze(1), input_pos)  # [2, 1863, 512]

        x_res = self.skip_linear.__call__(torch.cat([x_res, x], dim=-1))
        x = self.conv1.__call__(x_res)
        x = x.mT
        t2 = self.t_embedder2(t)
        # long residual connection
        x = self.wavenet.__call__(x, g=t2.unsqueeze(2)).mT + self.res_projection(x_res)
        x = self.final_layer.__call__(x, t1).mT
        # x [2,80,1863]
        return self.conv2.__call__(x)

    @patch_call(forward)
    def __call__(self) -> None: ...
