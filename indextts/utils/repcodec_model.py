# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from indextts.util import patch_call, unwrap


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(unwrap(m.bias), 0)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    act: nn.GELU
    dwconv: nn.Conv1d
    norm: nn.LayerNorm
    pwconv1: nn.Linear
    pwconv2: nn.Linear
    gamma: nn.Parameter

    def __init__(self) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(384, 384, kernel_size=7, padding=3, groups=384)  # depthwise conv
        self.norm = nn.LayerNorm(384, eps=1e-6)
        self.pwconv1 = nn.Linear(384, 2048)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2048, 384)
        self.gamma = nn.Parameter(1 / 12 * torch.ones(384))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.mT  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.mT  # (B, T, C) -> (B, C, T)

        return residual + x

    @patch_call(forward)
    def __call__(self) -> None: ...


class VocosBackbone(nn.Module):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization
    """

    embed: nn.Conv1d
    norm: nn.LayerNorm
    convnext: Sequence[ConvNeXtBlock]
    final_layer_norm: nn.LayerNorm

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Conv1d(1024, 384, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(384, eps=1e-6)
        self.convnext = nn.ModuleList([ConvNeXtBlock() for _ in range(12)])
        self.final_layer_norm = nn.LayerNorm(384, eps=1e-6)
        self.apply(_init_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        x = self.norm(x.mT)
        x = x.mT
        for conv_block in self.convnext:
            x = conv_block(x)
        return self.final_layer_norm(x.mT)

    @patch_call(forward)
    def __call__(self) -> None: ...


class FactorizedVectorQuantize(nn.Module):
    in_project: nn.Conv1d
    out_project: nn.Conv1d
    codebook: nn.Embedding

    def __init__(self) -> None:
        super().__init__()

        self.in_project = weight_norm(nn.Conv1d(1024, 8, kernel_size=1))
        self.out_project = weight_norm(nn.Conv1d(8, 1024, kernel_size=1))

        self.codebook = nn.Embedding(8192, 8)

    def forward(self, z: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z: Tensor[B x D x T]

        Returns
        -------
        z_q: Tensor[B x D x T]
            Quantized continuous representation of input
        """

        # Factorized codes project input into low-dimensional space
        z_e = self.in_project(z)
        z_q = self.decode_latents(z_e)

        z_q = z_e + (z_q - z_e).detach()

        return self.out_project(z_q)

    def decode_code(self, embed_id: Tensor) -> Tensor:
        return F.embedding(embed_id, self.codebook.weight).mT

    def decode_latents(self, latents: Tensor) -> Tensor:
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        # L2 normalize encodings and codebook
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance between encodings and codebook,
        # the distance is equal to cosine distance
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        return self.decode_code(indices)

    def vq2emb(self, vq: Tensor) -> Tensor:
        emb = self.decode_code(vq)
        return self.out_project(emb)

    @patch_call(forward)
    def __call__(self) -> None: ...


class ResidualVQ(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    quantizers: Sequence[FactorizedVectorQuantize]

    def __init__(self) -> None:
        super().__init__()

        quantizers = [FactorizedVectorQuantize()]
        self.quantizers = nn.ModuleList(quantizers)

    def forward(self, z: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z : Tensor[B x D x T]
        Returns
        -------
        "quantized_out" : Tensor[B x D x T]
            Quantized continuous representation of input
        """

        z_q_i = self.quantizers[0](z)

        # Create mask to apply quantizer dropout
        mask = torch.full((z.shape[0],), fill_value=0, device=z.device) < 1

        return z_q_i * mask[:, None, None]

    def vq2emb(self, vq: Tensor) -> Tensor:
        return self.quantizers[0].vq2emb(vq[0])

    @patch_call(forward)
    def __call__(self) -> None: ...


class RepCodec(nn.Module):
    quantizer: ResidualVQ

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(VocosBackbone(), nn.Linear(384, 1024))
        self.quantizer = ResidualVQ()

        self.apply(_init_weights)

    def quantize(self, x: Tensor) -> Tensor:
        x = self.encoder(x.mT).mT

        return self.quantizer(x).mT
