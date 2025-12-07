# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from indextts.utils.maskgct.models.codec.amphion_codec.quantize.lookup_free_quantize import LookupFreeQuantize
from indextts.utils.maskgct.models.codec.amphion_codec.quantize.vector_quantize import VectorQuantize
from torch import nn

from indextts.utils.maskgct.models.codec.amphion_codec.quantize.factorized_vector_quantize import (
    FactorizedVectorQuantize,
)


class ResidualVQ(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    input_dim = 1024
    num_quantizers = 1
    codebook_size = 8192
    codebook_dim = 8
    quantizer_dropout = 0.0

    def __init__(self) -> None:
        super().__init__()

        self.quantizers = nn.ModuleList([FactorizedVectorQuantize() for _ in range(self.num_quantizers)])

    def forward(
        self, z: torch.Tensor, n_quantizers: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
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

        all_commit_losses = []
        all_codebook_losses = []
        all_indices = []
        all_quantized = []

        if n_quantizers is None:
            n_quantizers = self.num_quantizers

        if self.training:
            n_quantizers: torch.Tensor = torch.ones((z.shape[0],)) * self.num_quantizers + 1
            dropout = torch.randint(1, self.num_quantizers + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commit_loss_i, codebook_loss_i, indices_i, _z_e_i = quantizer(residual)

            # Create mask to apply quantizer dropout
            mask = torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            quantized_out += z_q_i * mask[:, None, None]
            residual -= z_q_i

            commit_loss_i = (commit_loss_i * mask).mean()
            codebook_loss_i = (codebook_loss_i * mask).mean()

            all_commit_losses.append(commit_loss_i)
            all_codebook_losses.append(codebook_loss_i)
            all_indices.append(indices_i)
            all_quantized.append(z_q_i)

        all_commit_losses, all_codebook_losses, all_indices, all_quantized = map(
            torch.stack,
            (all_commit_losses, all_codebook_losses, all_indices, all_quantized),
        )

        return (
            quantized_out,
            all_indices,
            all_commit_losses,
            all_codebook_losses,
            all_quantized,
        )

    def vq2emb(self, vq, n_quantizers=None):
        quantized_out = 0.0
        if n_quantizers is None:
            n_quantizers = self.num_quantizers
        for idx, quantizer in enumerate(self.quantizers):
            if idx >= n_quantizers:
                break
            quantized_out += quantizer.vq2emb(vq[idx])
        return quantized_out

    def latent2dist(self, z: torch.Tensor, n_quantizers: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        quantized_out = 0.0
        residual = z

        all_dists = []
        all_indices = []

        if n_quantizers is None:
            n_quantizers = self.num_quantizers

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break
            dist_i, indices_i, z_q_i = quantizer.latent2dist(residual)
            all_dists.append(dist_i)
            all_indices.append(indices_i)

            quantized_out += z_q_i
            residual -= z_q_i

        all_dists = torch.stack(all_dists)
        all_indices = torch.stack(all_indices)

        return all_dists, all_indices
