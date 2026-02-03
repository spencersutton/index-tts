# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections.abc import Sequence
from typing import override

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from indextts.util import patch_call, unwrap


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(unwrap(m.bias), 0)


class _ConvNeXtBlock(nn.Module):
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

    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init_value: float) -> None:
        super().__init__()

        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(torch.full([dim], layer_scale_init_value))

    @override
    def forward(self, x: Float[Tensor, "b c t"]) -> Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.mT  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x *= self.gamma
        x = x.mT  # (B, T, C) -> (B, C, T)

        return residual + x

    @patch_call(forward)
    def __call__(self) -> None: ...


class _VocosBackbone(nn.Module):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization
    """

    embed: nn.Conv1d
    norm: nn.LayerNorm
    convnext: Sequence[_ConvNeXtBlock]
    final_layer_norm: nn.LayerNorm

    def __init__(self, input_channels: int, dim: int, intermediate_dim: int, n_layers: int) -> None:
        super().__init__()

        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList([
            _ConvNeXtBlock(dim=dim, intermediate_dim=intermediate_dim, layer_scale_init_value=1 / n_layers)
            for _ in range(n_layers)
        ])  # pyright: ignore[reportAttributeAccessIssue]
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(_init_weights)

    @override
    def forward(self, x: Float[Tensor, "b c t"]) -> Tensor:
        x = self.embed(x)
        x = self.norm(x.mT)
        x = x.mT
        for conv_block in self.convnext:
            x = conv_block(x)
        return self.final_layer_norm(x.mT)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _FactorizedVectorQuantize(nn.Module):
    in_project: nn.Conv1d
    out_project: nn.Conv1d
    codebook: nn.Embedding

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()

        self.in_project = weight_norm(nn.Conv1d(input_dim, codebook_dim, kernel_size=1))
        self.out_project = weight_norm(nn.Conv1d(codebook_dim, input_dim, kernel_size=1))
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    @override
    def forward(self, z: Float[Tensor, "b d t"]) -> Tensor:
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

    def decode_code(self, embed_id: Int[Tensor, "b t"]) -> Tensor:
        return F.embedding(embed_id, self.codebook.weight).mT

    def decode_latents(self, latents: Float[Tensor, "b d t"]) -> Tensor:
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        # L2 normalize encodings and codebook
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance between encodings and codebook,
        # the distance is equal to cosine distance
        dist = (
            encodings.square().sum(1, keepdim=True)
            - 2 * encodings @ codebook.mT
            + codebook.square().sum(1, keepdim=True).mT
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        return self.decode_code(indices)

    def vq2emb(self, vq: Int[Tensor, "b d t"]) -> Tensor:
        emb = self.decode_code(vq[0])
        return self.out_project(emb)

    @patch_call(forward)
    def __call__(self) -> None: ...


class RepCodec(nn.Module):
    quantizer: _FactorizedVectorQuantize
    encoder: nn.Sequential

    @staticmethod
    def _remap_weights(_module: object, state_dict: dict[str, object], *_args: object) -> None:
        for k in list(state_dict.keys()):
            new_k = k.replace("quantizer.quantizers.0", "quantizer")
            state_dict[new_k] = state_dict.pop(k)

    def __init__(
        self,
        codebook_dim: int = 8,
        codebook_size: int = 8192,
        dim: int = 384,
        hidden_size: int = 1024,
        input_dim: int = 256,
        intermediate_dim: int = 2048,
        n_layers: int = 12,
    ) -> None:
        super().__init__()
        self.register_load_state_dict_pre_hook(self._remap_weights)

        self.encoder = nn.Sequential(
            _VocosBackbone(input_channels=hidden_size, dim=dim, intermediate_dim=intermediate_dim, n_layers=n_layers),
            nn.Linear(dim, hidden_size),
        )
        self.quantizer = _FactorizedVectorQuantize(
            input_dim=input_dim, codebook_size=codebook_size, codebook_dim=codebook_dim
        )

        self.apply(_init_weights)

    def quantize(self, x: Float[Tensor, "b t c"]) -> Tensor:
        x = self.encoder(x.mT).mT

        return self.quantizer(x).mT
