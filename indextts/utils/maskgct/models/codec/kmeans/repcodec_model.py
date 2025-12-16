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

        self.in_project = weight_norm(nn.Conv1d(INPUT_DIM, CODEBOOK_DIM, kernel_size=1))
        self.out_project = weight_norm(nn.Conv1d(CODEBOOK_DIM, INPUT_DIM, kernel_size=1))

        self.codebook = nn.Embedding(CODEBOOK_SIZE, CODEBOOK_DIM)

    @override
    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Parameters
        ----------
        z: Tensor[B x D x T]

        Returns
        -------
        z_q: Tensor[B x D x T]
            Quantized continuous representation of input
        commit_loss: Tensor[B]
            Commitment loss to train encoder to predict vectors closer to codebook entries
        codebook_loss: Tensor[B]
            Codebook loss to update the codebook
        indices: Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        z_e: Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)

        """
        # Factorized codes project input into low-dimensional space if self.input_dim != self.codebook_dim
        z_e = self.in_project(z)
        z_q, indices = self.decode_latents(z_e)

        # Compute commitment loss and codebook loss
        commit_loss = torch.zeros(z.shape[0], device=z.device)
        codebook_loss = torch.zeros(z.shape[0], device=z.device)

        z_q = z_e + (z_q - z_e).detach()

        z_q = self.out_project(z_q)

        return z_q, commit_loss, codebook_loss, indices, z_e

    @patch_call(forward)
    def __call__(self) -> None: ...

    def embed_code(self, embed_id: Tensor) -> Tensor:
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id: Tensor) -> Tensor:
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents: Tensor) -> tuple[Tensor, Tensor]:
        encodings = latents.transpose(1, 2).reshape(-1, latents.size(1))
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
        indices = (-dist).max(1)[1].reshape(latents.size(0), latents.size(2))
        z_q = self.decode_code(indices)

        return z_q, indices

    def vq2emb(self, vq: Tensor, out_proj: bool = True) -> Tensor:
        emb = self.decode_code(vq)
        if out_proj:
            emb = self.out_project(emb)
        return emb


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
    def forward(self, z: Tensor, n_quantizers: int = 1) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.

        Returns
        -------
        "quantized_out" : Tensor[B x D x T]
            Quantized continuous representation of input
        "all_indices" : Tensor[N x B x T]
            Codebook indices for each codebook
            (quantized discrete representation of input)
        "all_commit_losses" : Tensor[N]
        "all_codebook_losses" : Tensor[N]
        "all_quantized" : Tensor[N x B x D x T]

        """
        z_q_i, commit_loss_i, codebook_loss_i, indices_i, _z_e_i = self.quantizer(z)

        # Create mask to apply quantizer dropout
        mask = torch.full((z.shape[0],), fill_value=0, device=z.device) < n_quantizers

        return (
            z_q_i * mask[:, None, None],
            torch.stack([indices_i]),
            torch.stack([(commit_loss_i * mask).mean()]),
            torch.stack([(codebook_loss_i * mask).mean()]),
            torch.stack([z_q_i]),
        )

    @patch_call(forward)
    def __call__(self) -> None: ...

    def vq2emb(self, vq: Tensor, n_quantizers: int = 1) -> Tensor:
        if n_quantizers > 0:
            return self.quantizer.vq2emb(vq[0])
        return torch.tensor(0.0, device=vq.device)


def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        assert m.bias is not None
        nn.init.constant_(m.bias, 0)


class RepCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            VocosBackbone(
                input_channels=HIDDEN_SIZE,
                dim=VOCOS_DIM,
                intermediate_dim=VOCOS_INTERMEDIATE_DIM,
                num_layers=VOCOS_NUM_LAYERS,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(VOCOS_DIM, HIDDEN_SIZE),
        )
        self.decoder = nn.Sequential(
            VocosBackbone(
                input_channels=HIDDEN_SIZE,
                dim=VOCOS_DIM,
                intermediate_dim=VOCOS_INTERMEDIATE_DIM,
                num_layers=VOCOS_NUM_LAYERS,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(VOCOS_DIM, HIDDEN_SIZE),
        )

        self.quantizer = ResidualVQ()

        self.apply(init_weights)

    def quantize(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)

        (quantized_out, all_indices, _all_commit_losses, _all_codebook_losses, _) = self.quantizer(x)

        if all_indices.shape[0] == 1:
            return all_indices.squeeze(0), quantized_out.transpose(1, 2)
        return all_indices, quantized_out.transpose(1, 2)
