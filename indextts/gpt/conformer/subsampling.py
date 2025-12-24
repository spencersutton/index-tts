# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)


"""Subsampling layer definition."""

from abc import ABC
from typing import override

from torch import Tensor, nn

from indextts.gpt.conformer.embedding import PositionalEncoding
from indextts.util import patch_call


class _BaseSubsampling(nn.Module, ABC):
    pos_enc: PositionalEncoding  # Module with position_encoding method

    def __init__(self) -> None:
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: int | Tensor, size: int) -> Tensor:
        return self.pos_enc.position_encoding(offset, size)


class Conv2dSubsampling2(_BaseSubsampling):
    """Convolutional 2D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float) -> None:
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, odim, 3, 2), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(odim * ((idim - 1) // 2), odim))
        self.pos_enc = PositionalEncoding(odim, dropout_rate)
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 2
        # 2 = (3 - 1) * 1
        self.right_context = 2

    @override
    def forward(self, x: Tensor, x_mask: Tensor, offset: int | Tensor = 0) -> tuple[Tensor, Tensor, Tensor]:
        """Subsample x.

        Args:
            x (Tensor): Input tensor (#batch, time, idim).
            x_mask (Tensor): Input mask (#batch, 1, time).

        Returns:
            Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.
            Tensor: positional encoding

        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2]

    @patch_call(forward)
    def __call__(self) -> None: ...
