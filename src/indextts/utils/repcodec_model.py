# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Final, override

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from indextts.util import patch_call, unwrap

EPSILON: Final = 1e-6
IN_FEATURES: Final = 384
INTERMEDIATE_DIM: Final = 2048
KERNEL_SIZE = 7
N_LAYERS: Final = 12
OUT_FEATURES: Final = 1024


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(unwrap(m.bias), 0)


class _ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        IN_FEATURES (int): Number of input channels.
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
        self.dwconv = nn.Conv1d(
            IN_FEATURES, IN_FEATURES, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE // 2, groups=IN_FEATURES
        )  # depthwise conv
        self.norm = nn.LayerNorm(IN_FEATURES, eps=EPSILON)
        self.pwconv1 = nn.Linear(IN_FEATURES, INTERMEDIATE_DIM)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(INTERMEDIATE_DIM, IN_FEATURES)
        self.gamma = nn.Parameter(1 / N_LAYERS * torch.ones(IN_FEATURES))

    @override
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


class _VocosBackbone(nn.Module):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization
    """

    embed: nn.Conv1d
    norm: nn.LayerNorm
    convnext: nn.ModuleList[_ConvNeXtBlock]
    final_layer_norm: nn.LayerNorm

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Conv1d(OUT_FEATURES, IN_FEATURES, kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE // 2)
        self.norm = nn.LayerNorm(IN_FEATURES, eps=EPSILON)
        self.convnext = nn.ModuleList([_ConvNeXtBlock() for _ in range(N_LAYERS)])
        self.final_layer_norm = nn.LayerNorm(IN_FEATURES, eps=EPSILON)
        self.apply(_init_weights)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        x = self.norm(x.mT)
        x = x.mT
        for conv_block in self.convnext:
            x = conv_block(x)
        return self.final_layer_norm(x.mT)

    @patch_call(forward)
    def __call__(self) -> None: ...


IN_CHANNELS: Final = 1024
LATENT_DIM: Final = 8
CODEBOOK_SIZE: Final = 8192


class _FactorizedVectorQuantize(nn.Module):
    in_project: nn.Conv1d
    out_project: nn.Conv1d
    codebook: nn.Embedding

    def __init__(self) -> None:
        super().__init__()

        self.in_project = weight_norm(nn.Conv1d(IN_CHANNELS, LATENT_DIM, kernel_size=1))
        self.out_project = weight_norm(nn.Conv1d(LATENT_DIM, IN_CHANNELS, kernel_size=1))
        self.codebook = nn.Embedding(CODEBOOK_SIZE, LATENT_DIM)

    @override
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
        encodings = einops.rearrange(latents, "b d t -> (b t) d")
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
        indices = einops.rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        return self.decode_code(indices)

    def vq2emb(self, vq: Tensor) -> Tensor:
        emb = self.decode_code(vq[0])
        return self.out_project(emb)

    @patch_call(forward)
    def __call__(self) -> None: ...


class RepCodec(nn.Module):
    encoder: nn.Sequential
    quantizer: _FactorizedVectorQuantize

    @staticmethod
    def _remap_weights(_module: object, state_dict: dict[str, object], *_args: object) -> None:
        for k in list(state_dict.keys()):
            new_k = k.replace("quantizer.quantizers.0", "quantizer")
            state_dict[new_k] = state_dict.pop(k)

    def __init__(self) -> None:
        super().__init__()
        self.register_load_state_dict_pre_hook(self._remap_weights)

        self.encoder = nn.Sequential(_VocosBackbone(), nn.Linear(IN_FEATURES, OUT_FEATURES))
        self.quantizer = _FactorizedVectorQuantize()

        self.apply(_init_weights)

    def quantize(self, x: Tensor) -> Tensor:
        x = self.encoder(x.mT).mT

        return self.quantizer(x).mT
