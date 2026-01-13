# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from indextts.util import patch_call


class FactorizedVectorQuantize(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.in_project = weight_norm(nn.Conv1d(1024, 8, kernel_size=1))
        self.out_project = weight_norm(nn.Conv1d(8, 1024, kernel_size=1))

        self.codebook = nn.Embedding(8192, 8)

    def forward(self, z):
        """
        Parameters
        ----------
        z: torch.Tensor[B x D x T]

        Returns
        -------
        z_q: torch.Tensor[B x D x T]
            Quantized continuous representation of input
        """

        # Factorized codes project input into low-dimensional space
        z_e = self.in_project(z)
        z_q = self.decode_latents(z_e)

        z_q = z_e + (z_q - z_e).detach()

        return self.out_project(z_q)

    def decode_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight).mT

    def decode_latents(self, latents):
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

    def vq2emb(self, vq):
        emb = self.decode_code(vq)
        return self.out_project(emb)

    @patch_call(forward)
    def __call__(self): ...
