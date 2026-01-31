# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections.abc import Sequence
from functools import cached_property
from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from indextts.util import patch_call


class _AdaptiveLayerNorm(nn.Module):
    r"""Adaptive Layer Normalization"""

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.dim = dim
        self.project_layer = nn.Linear(dim, 2 * dim)
        self.norm = _RMSNorm(dim=dim)

    @override
    def forward(self, input: Float[Tensor, "b t d"], embedding: Float[Tensor, "b d"] | None = None) -> Tensor:
        if embedding is None:
            return self.norm(input)
        weight, bias = torch.split(self.project_layer(embedding), self.dim, dim=-1)
        return weight * self.norm(input) + bias

    @patch_call(forward)
    def __call__(self) -> None: ...


class Transformer(nn.Module):
    layers: Sequence["_TransformerBlock"]
    norm: "_AdaptiveLayerNorm"

    def __init__(self, block_size: int, dim: int, n_head: int = 8, n_layer: int = 13) -> None:
        super().__init__()

        self.n_layer = n_layer
        self.block_size = block_size
        self.head_dim = dim // n_head

        self.layers = nn.ModuleList(_TransformerBlock(dim=dim) for _ in range(n_layer))  # pyright: ignore[reportAttributeAccessIssue]
        self.norm = _AdaptiveLayerNorm(dim=dim)

    @cached_property[Tensor]
    def freqs_cis(self) -> Tensor:
        dtype = self.norm.project_layer.weight.dtype
        device = self.norm.project_layer.weight.device

        freq_seq = torch.arange(0, self.head_dim, 2, device=device)
        inv_freq = (10000 ** (freq_seq / self.head_dim)).reciprocal()
        t = torch.arange(self.block_size, device=device, dtype=dtype)
        angles = torch.outer(t, inv_freq)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        return torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)

    @override
    def forward(self, x: Float[Tensor, "b t d"], c: Float[Tensor, "b d"], input_pos: Int[Tensor, "t"]) -> Tensor:
        freqs_cis = self.freqs_cis[input_pos]
        mid = self.n_layer // 2
        skip_stack: list[Tensor] = []
        for i, layer in enumerate(self.layers):
            skip_in_x = skip_stack.pop() if i > mid else None
            x = layer.__call__(x, c, input_pos, freqs_cis, skip_in_x)
            if i < mid:
                skip_stack.append(x)
        return self.norm.__call__(x, c)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _TransformerBlock(nn.Module):
    attention: "_Attention"
    feed_forward: "_FeedForward"
    ffn_norm: "_AdaptiveLayerNorm"
    attention_norm: "_AdaptiveLayerNorm"
    skip_in_linear: nn.Linear

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.dim = dim
        self.attention = _Attention(dim=dim)
        self.feed_forward = _FeedForward(dim=dim)
        self.ffn_norm = _AdaptiveLayerNorm(dim=dim)
        self.attention_norm = _AdaptiveLayerNorm(dim=dim)
        self.skip_in_linear = nn.Linear(dim * 2, dim)

    @override
    def forward(
        self,
        x: Float[Tensor, "b t d"],
        c: Float[Tensor, "b d"],
        input_pos: Int[Tensor, "t"],
        freqs_cis: Float[Tensor, ""],
        skip_in_x: Float[Tensor, "b t d"] | None = None,
    ) -> Tensor:
        if skip_in_x is not None:
            x = self.skip_in_linear(torch.cat([x, skip_in_x], dim=-1))
        h = x + self.attention.__call__(self.attention_norm(x, c), freqs_cis)
        return h + self.feed_forward.__call__(self.ffn_norm(h, c))

    @patch_call(forward)
    def __call__(self) -> None: ...


class _Attention(nn.Module):
    wqkv: nn.Linear
    wo: nn.Linear

    def __init__(self, dim: int, n_head: int = 8) -> None:
        super().__init__()

        self.dim = dim
        self.n_head = n_head
        self.head_dim = dim // n_head

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(self.dim, self.dim * 3, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

    @override
    def forward(self, x: Float[Tensor, "b t d"], freqs_cis: Float[Tensor, ""]) -> Tensor:
        bsz, seqlen, _ = x.shape

        query_key_value = self.wqkv(x)
        q, k, v = query_key_value.split((self.dim, self.dim, self.dim), dim=-1)
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim)

        q = _apply_rotary_emb(q, freqs_cis)
        k = _apply_rotary_emb(k, freqs_cis)

        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]

        k = k.repeat_interleave(1, dim=1)
        v = v.repeat_interleave(1, dim=1)
        y = F.scaled_dot_product_attention(q, k, v)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.wo(y)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _FeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.w1 = nn.Linear(dim, dim * 3, bias=False)
        self.w3 = nn.Linear(dim, dim * 3, bias=False)
        self.w2 = nn.Linear(dim * 3, dim, bias=False)

    @override
    def forward(self, x: Float[Tensor, "b t d"]) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    @patch_call(forward)
    def __call__(self) -> None: ...


class _RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))

    @staticmethod
    def _norm(x: Float[Tensor, "b t d"]) -> Tensor:
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5)

    @override
    def forward(self, x: Float[Tensor, "b t d"]) -> Tensor:
        return self._norm(x.float()).type_as(x) * self.weight

    @patch_call(forward)
    def __call__(self) -> None: ...


def _apply_rotary_emb(x: Float[Tensor, "b t h d"], freqs_cis: Float[Tensor, ""]) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
