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

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from indextts.util import patch_call

MAX_LEN = 5000
DIM = 512


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

    def __init__(self, d_model: int) -> None:
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.xscale = math.sqrt(DIM)
        self.dropout = nn.Dropout(0.0)

        pe = torch.zeros(MAX_LEN, DIM)
        position = torch.arange(0, MAX_LEN).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, DIM, 2) * -(math.log(10000.0) / DIM))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def position_encoding(self, offset: int | Tensor, size: int) -> Tensor:
        """For getting encoding in a streaming fashion

        Args:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding

        Returns:
            Tensor: Corresponding encoding
        """
        # How to subscript a Union type:
        #   https://github.com/pytorch/pytorch/issues/69434
        if isinstance(offset, int) or (isinstance(offset, Tensor) and offset.dim() == 0):
            assert offset + size < MAX_LEN
            pos_emb = self.pe[:, offset : offset + size]
        else:  # for batched streaming decoding on GPU
            assert torch.max(offset) + size < MAX_LEN
            index = offset.unsqueeze(1) + torch.arange(0, size).to(offset.device)  # B X T
            flag = index > 0
            # remove negative offset
            index *= flag
            pos_emb = F.embedding(index, self.pe[0])  # B X T X d_model

        return pos_emb

    def forward(self, x: Tensor, offset: int | Tensor = 0) -> tuple[Tensor, Tensor]:
        """Compute positional encoding.
        Args:
            x (Tensor): Input tensor (batch, time, `*`).
        Returns:
            Tensor: Encoded tensor (batch, time, `*`).
            Tensor: Positional embedding tensor (1, time, `*`).
        """
        self.pe = self.pe.to(x.device)
        x *= self.xscale
        pos_emb = self.position_encoding(offset, x.size(1))
        return self.dropout(x), self.dropout(pos_emb)

    @patch_call(forward)
    def __call__(self) -> None: ...
