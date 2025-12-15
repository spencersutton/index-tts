# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import override

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn.utils import weight_norm

from indextts.util import patch_call
from indextts.utils.maskgct.models.codec.kmeans.vocos import VocosBackbone


class FactorizedVectorQuantize(nn.Module):
    input_dim: int = 1024
    codebook_size: int = 8192
    codebook_dim: int = 8

    def __init__(self) -> None:
        super().__init__()

        self.in_project = weight_norm(nn.Conv1d(self.input_dim, self.codebook_dim, kernel_size=1))
        self.out_project = weight_norm(nn.Conv1d(self.codebook_dim, self.input_dim, kernel_size=1))

        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)

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


class ResidualVQ(nn.Module):
    """Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    input_dim: int = 1024
    num_quantizers: int = 1
    quantizer: FactorizedVectorQuantize

    def __init__(self) -> None:
        super().__init__()

        self.quantizer = FactorizedVectorQuantize()

    @override
    def forward(self, z: Tensor, n_quantizers: int | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
        quantized_out = 0.0
        residual = z

        if n_quantizers is None:
            n_quantizers = self.num_quantizers

        z_q_i, commit_loss_i, codebook_loss_i, indices_i, _z_e_i = self.quantizer(residual)

        # Create mask to apply quantizer dropout
        mask = torch.full((z.shape[0],), fill_value=0, device=z.device) < n_quantizers
        quantized_out += z_q_i * mask[:, None, None]
        residual -= z_q_i

        commit_loss_i = (commit_loss_i * mask).mean()
        codebook_loss_i = (codebook_loss_i * mask).mean()

        all_commit_losses = [commit_loss_i]
        all_codebook_losses = [codebook_loss_i]
        all_indices = [indices_i]
        all_quantized = [z_q_i]

        all_commit_losses, all_codebook_losses, all_indices, all_quantized = map(
            torch.stack,
            (
                all_commit_losses,
                all_codebook_losses,
                all_indices,
                all_quantized,
            ),
        )

        return (
            quantized_out,
            all_indices,
            all_commit_losses,
            all_codebook_losses,
            all_quantized,
        )

    @patch_call(forward)
    def __call__(self) -> None: ...

    def vq2emb(self, vq: Tensor, n_quantizers: int | None = None) -> Tensor:
        if n_quantizers is None:
            n_quantizers = self.num_quantizers
        if n_quantizers > 0:
            return self.quantizer.vq2emb(vq[0])
        return torch.tensor(0.0, device=vq.device)

    def latent2dist(self, z: Tensor, n_quantizers: int | None = None) -> tuple[Tensor, Tensor]:
        if n_quantizers is None:
            n_quantizers = self.num_quantizers

        if n_quantizers > 0:
            dist_i, indices_i, _z_q_i = self.quantizer.latent2dist(z)
            return torch.stack([dist_i]), torch.stack([indices_i])

        # Return empty tensors if no quantizers
        return torch.stack([]), torch.stack([])


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        assert m.bias is not None
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        assert m.bias is not None
        nn.init.constant_(m.bias, 0)


codebook_size = 8192
codebook_dim = 8
hidden_size = 1024
vocos_dim = 384
vocos_intermediate_dim = 2048
vocos_num_layers = 12
num_quantizers = 1
downsample_scale = 1


class RepCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            VocosBackbone(
                input_channels=hidden_size,
                dim=vocos_dim,
                intermediate_dim=vocos_intermediate_dim,
                num_layers=vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(vocos_dim, hidden_size),
        )
        self.decoder = nn.Sequential(
            VocosBackbone(
                input_channels=hidden_size,
                dim=vocos_dim,
                intermediate_dim=vocos_intermediate_dim,
                num_layers=vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(vocos_dim, hidden_size),
        )

        self.quantizer = ResidualVQ()

        self.reset_parameters()

    def quantize(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encoder.forward(x.transpose(1, 2)).transpose(1, 2)

        (
            quantized_out,
            all_indices,
            _all_commit_losses,
            _all_codebook_losses,
            _,
        ) = self.quantizer(x)

        if all_indices.shape[0] == 1:
            return all_indices.squeeze(0), quantized_out.transpose(1, 2)
        return all_indices, quantized_out.transpose(1, 2)

    def reset_parameters(self) -> None:
        self.apply(init_weights)
