# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

from indextts.utils.maskgct.models.codec.amphion_codec.quantize.factorized_vector_quantize import (
    FactorizedVectorQuantize,
)


class ResidualVQ(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(self) -> None:
        super().__init__()

        quantizers = [FactorizedVectorQuantize()]
        self.quantizers = cast(Sequence[FactorizedVectorQuantize], nn.ModuleList(quantizers))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : torch.Tensor[B x D x T]
        Returns
        -------
        "quantized_out" : torch.Tensor[B x D x T]
            Quantized continuous representation of input
        """

        z_q_i = self.quantizers[0](z)

        # Create mask to apply quantizer dropout
        mask = torch.full((z.shape[0],), fill_value=0, device=z.device) < 1

        return z_q_i * mask[:, None, None]

    def vq2emb(self, vq) -> torch.Tensor:
        return self.quantizers[0].vq2emb(vq[0])
