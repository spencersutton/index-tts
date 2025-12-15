# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import override

import torch
from indextts.utils.maskgct.models.codec.amphion_codec.quantize.lookup_free_quantize import LookupFreeQuantize
from indextts.utils.maskgct.models.codec.amphion_codec.quantize.vector_quantize import VectorQuantize
from torch import Tensor, nn

from indextts.util import patch_call
from indextts.utils.maskgct.models.codec.amphion_codec.quantize.factorized_vector_quantize import (
    FactorizedVectorQuantize,
)


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
