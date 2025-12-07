# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn.utils import weight_norm


class FactorizedVectorQuantize(nn.Module):
    input_dim: int = 1024
    codebook_size: int = 8192
    codebook_dim: int = 8

    def __init__(self) -> None:
        super().__init__()

        self.in_project = weight_norm(nn.Conv1d(self.input_dim, self.codebook_dim, kernel_size=1))
        self.out_project = weight_norm(nn.Conv1d(self.codebook_dim, self.input_dim, kernel_size=1))

        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Parameters
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

    def embed_code(self, embed_id: Tensor) -> Tensor:
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id: Tensor) -> Tensor:
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents: Tensor) -> tuple[Tensor, Tensor]:
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
        z_q = self.decode_code(indices)

        return z_q, indices

    def vq2emb(self, vq: Tensor, out_proj: bool = True) -> Tensor:
        emb = self.decode_code(vq)
        if out_proj:
            emb = self.out_project(emb)
        return emb

    def latent2dist(self, latents: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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
        )  # (b*t, k)

        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        dist = rearrange(dist, "(b t) k -> b t k", b=latents.size(0))
        z_q = self.decode_code(indices)

        return -dist, indices, z_q
