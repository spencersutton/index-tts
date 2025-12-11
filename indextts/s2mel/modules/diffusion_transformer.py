import math
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn.utils import weight_norm

from indextts.config import S2MelConfig
from indextts.s2mel.modules.commons import sequence_mask
from indextts.s2mel.modules.gpt_fast.model import ModelArgs, Transformer
from indextts.s2mel.modules.wavenet import WN


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    freqs: Tensor

    def __init__(
        self,
        hidden_size: int,
    ) -> None:
        super().__init__()
        frequency_embedding_size: Final = 256
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.max_period: Final = 10000
        self.scale: Final = 1000

        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def timestep_embedding(self, t: Tensor) -> Tensor:
        """Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        args = self.scale * t[:, None].float() * self.freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t)
        return self.mlp(t_freq)


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = weight_norm(nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


HIDDEN_DIM: Final = 512
NUM_HEADS: Final = 8
CONTENT_DIM: Final = 512


class DiT(torch.nn.Module):
    input_pos: Tensor

    def __init__(self, args: S2MelConfig) -> None:
        super().__init__()
        model_args = ModelArgs(
            block_size=16384,  # args.DiT.block_size,
            n_layer=args.DiT.depth,
            n_head=NUM_HEADS,
            dim=HIDDEN_DIM,
            head_dim=HIDDEN_DIM // NUM_HEADS,
            vocab_size=1024,
            uvit_skip_connection=True,
            time_as_token=False,
        )
        self.transformer = Transformer(model_args)
        self.in_channels = args.DiT.in_channels
        self.out_channels = args.DiT.in_channels

        self.cond_projection = nn.Linear(CONTENT_DIM, HIDDEN_DIM, bias=True)  # continuous content

        self.t_embedder = TimestepEmbedder(HIDDEN_DIM)

        input_pos = torch.arange(16384)
        self.register_buffer("input_pos", input_pos)

        self.t_embedder2 = TimestepEmbedder(HIDDEN_DIM)
        self.conv1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.conv2 = nn.Conv1d(HIDDEN_DIM, args.DiT.in_channels, 1)
        self.wavenet = WN(
            hidden_channels=HIDDEN_DIM,
            kernel_size=args.wavenet.kernel_size,
            dilation_rate=args.wavenet.dilation_rate,
            n_layers=args.wavenet.num_layers,
            gin_channels=HIDDEN_DIM,
            p_dropout=args.wavenet.p_dropout,
            causal=False,
        )
        self.final_layer = FinalLayer(HIDDEN_DIM, 1, HIDDEN_DIM)
        self.res_projection = nn.Linear(
            HIDDEN_DIM, HIDDEN_DIM
        )  # residual connection from tranformer output to final output

        self.skip_linear = nn.Linear(HIDDEN_DIM + args.DiT.in_channels, HIDDEN_DIM)

        self.cond_x_merge_linear = nn.Linear(
            HIDDEN_DIM + args.DiT.in_channels * 2 + args.style_encoder.dim,
            HIDDEN_DIM,
        )

    def setup_caches(self, max_batch_size: int, max_seq_length: int) -> None:
        self.transformer.setup_caches(max_batch_size, max_seq_length, use_kv_cache=False)

    def forward(
        self,
        x: Tensor,
        prompt_x: Tensor,
        x_lens: Tensor,
        t: Tensor,
        style: Tensor,
        cond: Tensor,
        mask_content: bool = False,
    ) -> Tensor:
        """x (Tensor): random noise
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
        cond_in_module = self.cond_projection

        _B, _, T = x.size()

        t1 = self.t_embedder(t)  # (N, D) # t1 [2, 512]
        cond = cond_in_module(cond)  # cond [2,1863,512]->[2,1863,512]

        x = x.transpose(1, 2)  # [2,1863,80]
        prompt_x = prompt_x.transpose(1, 2)  # [2,1863,80]

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
        x = x.transpose(1, 2)
        t2 = self.t_embedder2(t)
        x = self.wavenet(x, x_mask, g=t2.unsqueeze(2)).transpose(1, 2) + self.res_projection(
            x_res
        )  # long residual connection
        x = self.final_layer(x, t1).transpose(1, 2)
        return self.conv2(x)
        # x [2,80,1863]
