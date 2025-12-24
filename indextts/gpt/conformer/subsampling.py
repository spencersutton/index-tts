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

from typing import override

from torch import Tensor, nn

from indextts.gpt.conformer.embedding import PositionalEncoding
from indextts.util import patch_call


class Conv2dSubsampling2(nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    positional_encoder: PositionalEncoding  # Module with position_encoding method
    conv: nn.Sequential
    out: nn.Sequential

    def __init__(self, idim: int, odim: int, dropout_rate: float) -> None:
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(1, odim, 3, 2), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(odim * ((idim - 1) // 2), odim))
        self.positional_encoder = PositionalEncoding(odim, dropout_rate)

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
        x, pos_emb = self.positional_encoder(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2]

    @patch_call(forward)
    def __call__(self) -> None: ...
