# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
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

"""Multi-Head Attention layer definition."""

import math
from typing import override

import torch
from beartype import beartype
from jaxtyping import Bool, Float
from torch import Tensor, nn

from indextts.util import patch_call


class RelPositionMultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
    """

    d_k: int
    h: int
    linear_k: nn.Linear
    linear_out: nn.Linear
    linear_pos: nn.Linear
    linear_q: nn.Linear
    linear_v: nn.Linear
    pos_bias_u: nn.Parameter
    pos_bias_v: nn.Parameter

    def __init__(self, dim: int, n_head: int) -> None:
        super().__init__()

        assert dim % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = dim // n_head
        self.h = n_head
        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)
        self.linear_out = nn.Linear(dim, dim)

        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(dim, dim, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    @override
    @beartype
    def forward(
        self,
        query: Float[Tensor, "batch time1 dim"],
        key: Float[Tensor, "batch time2 dim"],
        value: Float[Tensor, "batch time2 dim"],
        mask: Bool[Tensor, "batch time1_or_1 time2"],
        pos_emb: Float[Tensor, "batch_pos time2 dim"],
    ) -> Float[Tensor, "batch time1 dim"]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (Tensor): Query tensor (#batch, time1, size).
            key (Tensor): Key tensor (#batch, time2, size).
            value (Tensor): Value tensor (#batch, time2, size).
            mask (Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (Tensor): Positional embedding tensor
                (#batch, time2, size).
        Returns:
            Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        q_u = (q + self.pos_bias_u).transpose(1, 2)
        q_v = (q + self.pos_bias_v).transpose(1, 2)

        scores = (q_u @ k.mT + q_v @ p.mT) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)

    @beartype
    def forward_qkv(
        self,
        query: Float[Tensor, "batch time1 dim"],
        key: Float[Tensor, "batch time2 dim"],
        value: Float[Tensor, "batch time2 dim"],
    ) -> tuple[
        Float[Tensor, "batch heads time1 d_k"],
        Float[Tensor, "batch heads time2 d_k"],
        Float[Tensor, "batch heads time2 d_k"],
    ]:
        """Transform query, key and value.

        Args:
            query (Tensor): Query tensor (#batch, time1, size).
            key (Tensor): Key tensor (#batch, time2, size).
            value (Tensor): Value tensor (#batch, time2, size).

        Returns:
            Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    @beartype
    def forward_attention(
        self,
        value: Float[Tensor, "batch heads time2 d_k"],
        scores: Float[Tensor, "batch heads time1 time2"],
        mask: Bool[Tensor, "batch time1_or_1 time2"],
    ) -> Float[Tensor, "batch time1 dim"]:
        """Compute attention context vector.

        Args:
            value (Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.

        Returns:
            Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        mask = mask.unsqueeze(1) == 0
        mask = mask[..., : scores.size(-1)]
        scores = scores.masked_fill(mask, -float("inf"))
        attn = scores.softmax(dim=-1)

        x = attn @ value
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)

        return self.linear_out(x)  # (batch, time1, d_model)

    @patch_call(forward)
    def __call__(self) -> None: ...
