# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import cache
from typing import ClassVar, override

import torch
from beartype import beartype
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from indextts.util import patch_call


class _AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization"""

    dim: int
    norm: _RMSNorm
    project_layer: nn.Linear

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.dim = dim
        self.norm = _RMSNorm(dim)
        self.project_layer = nn.Linear(dim, 2 * dim)

    @override
    @beartype
    def forward(
        self, input: Float[Tensor, "batch seq dim"], embedding: Float[Tensor, "batch 1 dim"]
    ) -> Float[Tensor, "batch seq dim"]:
        weight, bias = self.project_layer(embedding).split(self.dim, dim=-1)
        return weight * self.norm.__call__(input) + bias

    @patch_call(forward)
    def __call__(self) -> None: ...


@cache
def _compute_frequencies(device: torch.device, block_size: int, heads: int) -> Float[Tensor, "block_size head_half 2"]:
    freq_seq = torch.arange(0, heads, 2)
    inv_freq = 1 / (10000 ** (freq_seq / heads))
    angles = torch.arange(block_size).outer(inv_freq)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return torch.view_as_real(freqs_cis).to(device)


class Transformer(nn.Module):
    block_size: ClassVar[int] = 2**14

    head_dim: int
    layers: nn.ModuleList[_TransformerBlock]
    norm: _AdaptiveLayerNorm

    def __init__(self, dim: int, n_head: int = 8, n_layer: int = 13) -> None:
        super().__init__()

        self.head_dim = dim // n_head

        self.layers = nn.ModuleList(_TransformerBlock(dim) for _ in range(n_layer))
        self.norm = _AdaptiveLayerNorm(dim)

    @override
    @beartype
    def forward(
        self, x: Float[Tensor, "batch seq dim"], c: Float[Tensor, "batch 1 dim"], input_pos: Int[Tensor, "seq"]
    ) -> Float[Tensor, "batch seq dim"]:
        computed_frequencies = _compute_frequencies(x.device, self.block_size, self.head_dim)
        freqs_cis = computed_frequencies[input_pos]
        mid = len(self.layers) // 2
        skip_stack: list[Tensor] = []
        for i, layer in enumerate(self.layers):
            skip_in_x = skip_stack.pop() if i > mid else None
            x = layer.__call__(x, c, freqs_cis, skip_in_x)
            if i < mid:
                skip_stack.append(x)
        return self.norm.__call__(x, c)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _TransformerBlock(nn.Module):
    attention_norm: _AdaptiveLayerNorm
    attention: _Attention
    dim: int
    feed_forward: _FeedForward
    ffn_norm: _AdaptiveLayerNorm
    skip_in_linear: nn.Linear

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.dim = dim
        self.attention = _Attention(dim)
        self.feed_forward = _FeedForward(dim)
        self.ffn_norm = _AdaptiveLayerNorm(dim)
        self.attention_norm = _AdaptiveLayerNorm(dim)
        self.skip_in_linear = nn.Linear(dim * 2, dim)

    @override
    @beartype
    def forward(
        self,
        x: Float[Tensor, "batch seq dim"],
        c: Float[Tensor, "batch 1 dim"],
        freqs_cis: Float[Tensor, "seq head_half 2"],
        skip_in_x: Float[Tensor, "batch seq dim"] | None = None,
    ) -> Float[Tensor, "batch seq dim"]:
        if skip_in_x is not None:
            x = self.skip_in_linear(torch.cat([x, skip_in_x], dim=-1))
        norm = self.attention_norm.__call__(x, c)
        h = x + self.attention.__call__(norm, freqs_cis)
        ffm_norm = self.ffn_norm.__call__(h, c)
        return h + self.feed_forward.__call__(ffm_norm)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _Attention(nn.Module):
    dim: int
    head_dim: int
    n_head: int
    wo: nn.Linear
    wqkv: nn.Linear

    def __init__(self, dim: int, n_head: int = 8) -> None:
        super().__init__()

        self.dim = dim
        self.n_head = n_head
        self.head_dim = dim // n_head

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    @override
    @beartype
    def forward(
        self, x: Float[Tensor, "batch seq dim"], freqs_cis: Float[Tensor, "seq head_half 2"]
    ) -> Float[Tensor, "batch seq dim"]:
        bsz, seq_len, _ = x.shape

        query_key_value = self.wqkv(x)
        q, k, v = query_key_value.split(self.dim, dim=-1)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim)

        q = _apply_rotary_emb(q, freqs_cis)
        k = _apply_rotary_emb(k, freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k = k.repeat_interleave(1, dim=1)
        v = v.repeat_interleave(1, dim=1)
        y = F.scaled_dot_product_attention(q, k, v)

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        return self.wo(y)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _FeedForward(nn.Module):
    w1: nn.Linear
    w2: nn.Linear
    w3: nn.Linear

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.w1 = nn.Linear(dim, dim * 3, bias=False)
        self.w3 = nn.Linear(dim, dim * 3, bias=False)
        self.w2 = nn.Linear(dim * 3, dim, bias=False)

    @override
    @beartype
    def forward(self, x: Float[Tensor, "batch seq dim"]) -> Float[Tensor, "batch seq dim"]:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    @patch_call(forward)
    def __call__(self) -> None: ...


class _RMSNorm(nn.Module):
    weight: nn.Parameter

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))

    @staticmethod
    @beartype
    def _norm(x: Float[Tensor, "batch seq dim"]) -> Float[Tensor, "batch seq dim"]:
        return x * (x.square().mean(dim=-1, keepdim=True) + 1e-5).rsqrt()

    @override
    @beartype
    def forward(self, x: Float[Tensor, "batch seq dim"]) -> Float[Tensor, "batch seq dim"]:
        return self._norm(x) * self.weight

    @patch_call(forward)
    def __call__(self) -> None: ...


@beartype
def _apply_rotary_emb(
    x: Float[Tensor, "batch seq heads head_half_2"], freqs_cis: Float[Tensor, "seq head_half 2"]
) -> Float[Tensor, "batch seq heads head_half_2"]:
    x_shaped = x.view(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, x_shaped.size(1), 1, x_shaped.size(3), 2)
    x0, x1 = x_shaped[..., 0], x_shaped[..., 1]
    f0, f1 = freqs_cis[..., 0], freqs_cis[..., 1]

    out1 = x0 * f0 - x1 * f1
    out2 = x1 * f0 + x0 * f1

    return torch.stack((out1, out2), dim=-1).flatten(3).type_as(x)
