# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

"""Positonal Encoding Module."""

import math
from typing import TYPE_CHECKING, override

import torch
from jaxtyping import Float
from torch import Tensor, nn

from indextts.util import patch_call


class RelPositionalEncoding(nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))

    Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        max_len (int): Maximum input length.
    """

    if TYPE_CHECKING:
        pe: Tensor = torch.empty(0)
    xscale: float

    def __init__(self, dim: int, max_len: int = 5000) -> None:
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.xscale = math.sqrt(dim)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = (torch.arange(0, dim, 2) * -(math.log(10000.0) / dim)).exp()
        pe[:, 0::2] = (position * div_term).sin()
        pe[:, 1::2] = (position * div_term).cos()
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    @override
    def forward(self, x: Float[Tensor, "b t d"]) -> tuple[Tensor, Tensor]:
        """Compute positional encoding.
        Args:
            x (Tensor): Input tensor (batch, time, `*`).
        Returns:
            Tensor: Encoded tensor (batch, time, `*`).
            Tensor: Positional embedding tensor (1, time, `*`).
        """
        x *= self.xscale
        return x, self.pe[:, : x.size(1)]

    @patch_call(forward)
    def __call__(self) -> None: ...
