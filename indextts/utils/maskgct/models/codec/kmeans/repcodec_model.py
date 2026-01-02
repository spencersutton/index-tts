# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import override

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import weight_norm

from indextts.util import patch_call
from indextts.utils.maskgct.models.codec.kmeans.vocos import VocosBackbone

INPUT_DIM = 1024
CODEBOOK_SIZE = 8192
CODEBOOK_DIM = 8
HIDDEN_SIZE = 1024
VOCOS_DIM = 384
VOCOS_INTERMEDIATE_DIM = 2048
VOCOS_NUM_LAYERS = 12
NUM_QUANTIZERS = 1
DOWNSAMPLE_SCALE = 1


class FactorizedVectorQuantize(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self._in_project = weight_norm(nn.Conv1d(INPUT_DIM, CODEBOOK_DIM, kernel_size=1))
        self._out_project = weight_norm(nn.Conv1d(CODEBOOK_DIM, INPUT_DIM, kernel_size=1))

        self.codebook = nn.Embedding(CODEBOOK_SIZE, CODEBOOK_DIM)

    @override
    def forward(self, z: Tensor) -> Tensor:
        """Parameters
        ----------
        z: Tensor[B x D x T]

        Returns
        -------
        z_q: Tensor[B x D x T]
            Quantized continuous representation of input
        """
        # Factorized codes project input into low-dimensional space if self.input_dim != self.codebook_dim
        z_e = self._in_project(z)
        encodings = z_e.transpose(1, 2).reshape(-1, z_e.size(1))
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
        indices = (-dist).max(1)[1].reshape(z_e.size(0), z_e.size(2))
        z_q = F.embedding(indices, self.codebook.weight).transpose(1, 2)

        z_q = z_e + (z_q - z_e).detach()

        return self._out_project(z_q)

    @patch_call(forward)
    def __call__(self) -> None: ...

    def vq2emb(self, vq: Tensor) -> Tensor:
        return self._out_project(F.embedding(vq, self.codebook.weight).transpose(1, 2))


class ResidualVQ(nn.Module):
    """Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    num_quantizers: int = 1
    quantizer: FactorizedVectorQuantize

    def __init__(self) -> None:
        super().__init__()

        self.quantizer = FactorizedVectorQuantize()

    @override
    def forward(self, z: Tensor) -> Tensor:
        """Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        "quantized_out" : Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q_i = self.quantizer(z)

        # Create mask to apply quantizer dropout
        mask = torch.full((z.shape[0],), fill_value=0, device=z.device) < 1

        return z_q_i * mask[:, None, None]

    @patch_call(forward)
    def __call__(self) -> None: ...


class RepCodec(nn.Module):
    encoder: nn.Sequential[VocosBackbone | nn.Linear]
    quantizer: ResidualVQ

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            VocosBackbone(
                input_channels=HIDDEN_SIZE,
                dim=VOCOS_DIM,
                intermediate_dim=VOCOS_INTERMEDIATE_DIM,
                num_layers=VOCOS_NUM_LAYERS,
            ),
            nn.Linear(VOCOS_DIM, HIDDEN_SIZE),
        )

        self.quantizer = ResidualVQ()

    def quantize(self, x: Tensor) -> Tensor:
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)

        quantized_out = self.quantizer(x)

        return quantized_out.transpose(1, 2)
