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

import torch
from torch import nn

from indextts.gpt.conformer.embedding import RelPositionalEncoding
from indextts.util import patch_call

INPUT_DIM = 1024
OUTPUT_DIM = 512


class Conv2dSubsampling2(nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(1, OUTPUT_DIM, 3, 2), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(OUTPUT_DIM * ((INPUT_DIM - 1) // 2), OUTPUT_DIM))
        self.pos_enc = RelPositionalEncoding(OUTPUT_DIM)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, offset: int | torch.Tensor = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.
            torch.Tensor: positional encoding

        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2]

    @patch_call(forward)
    def __call__(self) -> None: ...
