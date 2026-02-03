# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections.abc import Sequence
from functools import cached_property
from typing import Final, override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from indextts.constants import DIM
from indextts.util import patch_call


class _AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization"""

    project_layer: nn.Linear
    norm: _RMSNorm

    def __init__(self) -> None:
        super().__init__()

        self.project_layer = nn.Linear(DIM, 1024)
        self.norm = _RMSNorm()

    @override
    def forward(self, input: Float[Tensor, "b t d"], embedding: Float[Tensor, "b t d"]) -> Tensor:
        weight, bias = self.project_layer(embedding).split(DIM, dim=-1)
        return weight * self.norm.__call__(input) + bias

    @patch_call(forward)
    def __call__(self) -> None: ...


class Transformer(nn.Module):
    layers: Sequence[_TransformerBlock]
    norm: _AdaptiveLayerNorm

    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.ModuleList(_TransformerBlock() for _ in range(13))  # pyright: ignore[reportAttributeAccessIssue]
        self.norm = _AdaptiveLayerNorm()

    @cached_property[Tensor]
    def freqs_cis(self) -> Tensor:
        dtype = self.norm.project_layer.weight.dtype
        device = self.norm.project_layer.weight.device

        freq_seq = torch.arange(0, 64, 2, device=device)
        inv_freq = (10000 ** (freq_seq / 64)).reciprocal()
        t = torch.arange(16384, device=device, dtype=dtype)
        angles = t.outer(inv_freq)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        return torch.view_as_real(freqs_cis)

    @override
    def forward(self, x: Float[Tensor, "b t d"], c: Float[Tensor, "b t d"], input_pos: Int[Tensor, "t"]) -> Tensor:  # noqa: UP037
        freqs_cis = self.freqs_cis[input_pos]
        mid = 13 // 2
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
    attention: _Attention
    feed_forward: _FeedForward
    ffn_norm: _AdaptiveLayerNorm
    attention_norm: _AdaptiveLayerNorm
    skip_in_linear: nn.Linear

    def __init__(self) -> None:
        super().__init__()

        self.attention = _Attention()
        self.feed_forward = _FeedForward()
        self.ffn_norm = _AdaptiveLayerNorm()
        self.attention_norm = _AdaptiveLayerNorm()
        self.skip_in_linear = nn.Linear(1024, DIM)

    @override
    def forward(
        self,
        x: Float[Tensor, "b t d"],
        c: Float[Tensor, "b t d"],
        freqs_cis: Float[Tensor, "b t d"],
        skip_in_x: Float[Tensor, "b t d"] | None = None,
    ) -> Tensor:
        if skip_in_x is not None:
            x = self.skip_in_linear(torch.cat([x, skip_in_x], dim=-1))
        norm = self.attention_norm.__call__(x, c)
        h = x + self.attention.__call__(norm, freqs_cis)
        ffm_norm = self.ffn_norm.__call__(h, c)
        return h + self.feed_forward.__call__(ffm_norm)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _Attention(nn.Module):
    wqkv: nn.Linear
    wo: nn.Linear
    n_head: Final = 8
    head_dim: Final = DIM // n_head

    def __init__(self) -> None:
        super().__init__()

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(DIM, DIM * 3, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)

    @override
    def forward(self, x: Float[Tensor, "b t d"], freqs_cis: Float[Tensor, "b t d"]) -> Tensor:
        bsz, seq_len, _ = x.shape

        query_key_value = self.wqkv(x)
        q, k, v = query_key_value.split(DIM, dim=-1)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim)

        q = _apply_rotary_emb(q, freqs_cis)
        k = _apply_rotary_emb(k, freqs_cis)

        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]

        k = k.repeat_interleave(1, dim=1)
        v = v.repeat_interleave(1, dim=1)
        y = F.scaled_dot_product_attention(q, k, v)

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, DIM)
        return self.wo(y)

    @patch_call(forward)
    def __call__(self) -> None: ...


class _FeedForward(nn.Module):
    w1: nn.Linear
    w2: nn.Linear
    w3: nn.Linear

    def __init__(self) -> None:
        super().__init__()

        self.w1 = nn.Linear(DIM, DIM * 3, bias=False)
        self.w3 = nn.Linear(DIM, DIM * 3, bias=False)
        self.w2 = nn.Linear(DIM * 3, DIM, bias=False)

    @override
    def forward(self, x: Float[Tensor, "b t d"]) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    @patch_call(forward)
    def __call__(self) -> None: ...


class _RMSNorm(nn.Module):
    weight: nn.Parameter

    def __init__(self) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(DIM))

    @staticmethod
    def _norm(x: Float[Tensor, "b t d"]) -> Tensor:
        return x * (x.square().mean(dim=-1, keepdim=True) + 1e-5).rsqrt()

    @override
    def forward(self, x: Float[Tensor, "b t d"]) -> Tensor:
        return self._norm(x) * self.weight

    @patch_call(forward)
    def __call__(self) -> None: ...


def _apply_rotary_emb(x: Float[Tensor, "b t h d"], freqs_cis: Float[Tensor, "b t d"]) -> Tensor:
    """Apply rotary embeddings to a (B, T, H, D) tensor.

    This implementation is intentionally allocation-light:
    - avoids `torch.stack(...).flatten(...)` (extra intermediate)
    - uses fused `addcmul_` where possible

    `freqs_cis` is expected to contain real/imag parts in the final dimension (size 2),
    and is reshaped for broadcasting over batch and heads.
    """

    # Reshape to pairs so we can apply complex rotation on the last dimension.
    # Shape: (B, T, H, D/2, 2)
    xshaped = x.reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)

    # out[..., 0] = x0 * f0 - x1 * f1
    # out[..., 1] = x1 * f0 + x0 * f1
    out = torch.empty_like(xshaped)
    torch.mul(xshaped[..., 0], freqs_cis[..., 0], out=out[..., 0])
    out[..., 0].addcmul_(xshaped[..., 1], freqs_cis[..., 1], value=-1.0)
    torch.mul(xshaped[..., 1], freqs_cis[..., 0], out=out[..., 1])
    out[..., 1].addcmul_(xshaped[..., 0], freqs_cis[..., 1], value=1.0)

    out = out.flatten(3)
    return out.type_as(x)
