import math

import torch
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from indextts.s2mel.modules.commons import sequence_mask
from indextts.s2mel.modules.constants import BLOCK_SIZE, HIDDEN_DIM, IN_CHANNELS
from indextts.s2mel.modules.gpt_fast.model import Transformer
from indextts.s2mel.modules.wavenet import WN
from indextts.util import patch_call

CONTENT_DIM = 512
DILATION_RATE = 1
FREQUENCY_EMBEDDING_SIZE = 256
KERNEL_SIZE = 5
STYLE_ENCODER_DIM = 192


def modulate(x: Tensor, shift: Tensor, scale: Tensor):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    freqs: Tensor

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(FREQUENCY_EMBEDDING_SIZE, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.max_period = 10000
        self.scale = 1000

        half = FREQUENCY_EMBEDDING_SIZE // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def timestep_embedding(self, t: Tensor) -> Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        args = self.scale * t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if FREQUENCY_EMBEDDING_SIZE % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t)
        return self.mlp(t_freq)

    @patch_call(forward)
    def __call__(self) -> None: ...


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = weight_norm(nn.Linear(hidden_size, patch_size * patch_size * out_channels))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

    @patch_call(forward)
    def __call__(self) -> None: ...


class DiT(nn.Module):
    input_pos: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.transformer = Transformer()

        self.x_embedder = weight_norm(nn.Linear(IN_CHANNELS, HIDDEN_DIM))

        self.cond_projection = nn.Linear(CONTENT_DIM, HIDDEN_DIM)  # continuous content

        self.t_embedder = TimestepEmbedder(HIDDEN_DIM)

        input_pos = torch.arange(BLOCK_SIZE)
        self.register_buffer("input_pos", input_pos)

        self.t_embedder2 = TimestepEmbedder(HIDDEN_DIM)
        self.conv1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.conv2 = nn.Conv1d(HIDDEN_DIM, IN_CHANNELS, 1)
        self.wavenet = WN()
        self.final_layer = FinalLayer(HIDDEN_DIM, 1, HIDDEN_DIM)
        # residual connection from tranformer output to final output
        self.res_projection = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

        self.skip_linear = nn.Linear(HIDDEN_DIM + IN_CHANNELS, HIDDEN_DIM)

        self.cond_x_merge_linear = nn.Linear(HIDDEN_DIM + IN_CHANNELS * 2 + STYLE_ENCODER_DIM, HIDDEN_DIM)

    def setup_caches(self, max_batch_size: int, max_seq_length: int) -> None:
        self.transformer.setup_caches(max_batch_size, max_seq_length)

    def forward(self, x: Tensor, prompt_x: Tensor, x_lens: Tensor, t: Tensor, style: Tensor, cond: Tensor) -> Tensor:
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
        _, _, T = x.size()

        t1 = self.t_embedder(t)  # (N, D) # t1 [2, 512]
        cond = self.cond_projection(cond)  # cond [2,1863,512]->[2,1863,512]

        x = x.mT  # [2,1863,80]
        prompt_x = prompt_x.mT  # [2,1863,80]

        x_in = torch.cat([x, prompt_x, cond], dim=-1)  # 80+80+512=672 [2, 1863, 672]
        x_in = torch.cat([x_in, style[:, None, :].repeat(1, T, 1)], dim=-1)  # [2, 1863, 864]

        x_in = self.cond_x_merge_linear(x_in)  # (N, T, D) [2, 1863, 512]

        x_mask = (
            sequence_mask(x_lens, max_length=x_in.size(1)).to(x.device).unsqueeze(1)
        )  # torch.Size([1, 1, 1863])True
        input_pos = self.input_pos[: x_in.size(1)]  # (T,) range（0，1863）
        x_mask_expanded = x_mask[:, None, :].repeat(1, 1, x_in.size(1), 1)  # torch.Size([1, 1, 1863, 1863]
        x_res = self.transformer(x_in, t1.unsqueeze(1), input_pos, x_mask_expanded)  # [2, 1863, 512]

        x_res = self.skip_linear(torch.cat([x_res, x], dim=-1))
        x = self.conv1(x_res)
        x = x.mT
        t2 = self.t_embedder2(t)
        x = self.wavenet(x, x_mask, g=t2.unsqueeze(2)).mT + self.res_projection(x_res)  # long residual connection
        x = self.final_layer(x, t1).mT
        # x [2,80,1863]
        return self.conv2(x)

    @patch_call(forward)
    def __call__(self) -> None: ...
