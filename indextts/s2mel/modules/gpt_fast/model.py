# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections.abc import Sequence
from functools import cached_property
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from indextts.s2mel.modules.constants import BLOCK_SIZE, HIDDEN_DIM
from indextts.util import patch_call

DIM = HIDDEN_DIM
N_HEAD = 8
N_LAYER = 13
NORM_EPS = 1e-5
ROPE_BASE = 10000
HEAD_DIM = HIDDEN_DIM // N_HEAD
INTERMEDIATE_SIZE = DIM * 3


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class AdaptiveLayerNorm(nn.Module):
    r"""Adaptive Layer Normalization"""

    def __init__(self) -> None:
        super().__init__()
        self.project_layer = nn.Linear(DIM, 2 * DIM)
        self.norm = RMSNorm()

    def forward(self, input: Tensor, embedding: Tensor | None = None) -> Tensor:
        if embedding is None:
            return self.norm(input)
        weight, bias = torch.split(self.project_layer(embedding), DIM, dim=-1)
        return weight * self.norm(input) + bias

    @patch_call(forward)
    def __call__(self) -> None: ...


class KVCache(nn.Module):
    k_cache: Tensor
    v_cache: Tensor

    def __init__(
        self, max_batch_size: int, max_seq_length: int, n_heads: int, head_dim: int, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor) -> tuple[Tensor, Tensor]:
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Transformer(nn.Module):
    layers: Sequence["TransformerBlock"]
    norm: "AdaptiveLayerNorm"

    def __init__(self) -> None:
        super().__init__()

        self.layers = cast(Sequence[TransformerBlock], nn.ModuleList(TransformerBlock() for _ in range(N_LAYER)))
        self.norm = AdaptiveLayerNorm()

    @cached_property[Tensor]
    def freqs_cis(self) -> Tensor:
        dtype = self.norm.project_layer.weight.dtype
        device = self.norm.project_layer.weight.device

        freq_seq = torch.arange(0, HEAD_DIM, 2, device=device)
        inv_freq = (ROPE_BASE ** (freq_seq / HEAD_DIM)).reciprocal()
        t = torch.arange(BLOCK_SIZE, device=device, dtype=dtype)
        angles = torch.outer(t, inv_freq)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        return torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)

    def forward(self, x: Tensor, c: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
        freqs_cis = self.freqs_cis[input_pos]
        mid = N_LAYER // 2
        skip_stack: list[Tensor] = []
        for i, layer in enumerate(self.layers):
            skip_in_x = skip_stack.pop() if i > mid else None
            x = layer(x, c, input_pos, freqs_cis, mask, skip_in_x)
            if i < mid:
                skip_stack.append(x)
        return self.norm(x, c)

    @patch_call(forward)
    def __call__(self) -> None: ...


class TransformerBlock(nn.Module):
    attention: "Attention"
    feed_forward: "FeedForward"
    ffn_norm: "AdaptiveLayerNorm"
    attention_norm: "AdaptiveLayerNorm"
    skip_in_linear: nn.Linear

    def __init__(self) -> None:
        super().__init__()
        self.attention = Attention()
        self.feed_forward = FeedForward()
        self.ffn_norm = AdaptiveLayerNorm()
        self.attention_norm = AdaptiveLayerNorm()

        self.skip_in_linear = nn.Linear(DIM * 2, DIM)

    def forward(
        self, x: Tensor, c: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor, skip_in_x: Tensor | None = None
    ) -> Tensor:
        if skip_in_x is not None:
            x = self.skip_in_linear(torch.cat([x, skip_in_x], dim=-1))
        h = x + self.attention(self.attention_norm(x, c), freqs_cis, mask)
        return h + self.feed_forward(self.ffn_norm(h, c))

    @patch_call(forward)
    def __call__(self) -> None: ...


class Attention(nn.Module):
    wqkv: nn.Linear
    wo: nn.Linear

    def __init__(self) -> None:
        super().__init__()

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(DIM, INTERMEDIATE_SIZE, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape

        query_key_value = self.wqkv(x)
        q, k, v = query_key_value.split((DIM, DIM, DIM), dim=-1)

        q = q.view(bsz, seqlen, N_HEAD, HEAD_DIM)
        k = k.view(bsz, seqlen, N_HEAD, HEAD_DIM)
        v = v.view(bsz, seqlen, N_HEAD, HEAD_DIM)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]

        k = k.repeat_interleave(1, dim=1)
        v = v.repeat_interleave(1, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, DIM)
        return self.wo(y)

    @patch_call(forward)
    def __call__(self) -> None: ...


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(DIM, INTERMEDIATE_SIZE, bias=False)
        self.w3 = nn.Linear(DIM, INTERMEDIATE_SIZE, bias=False)
        self.w2 = nn.Linear(INTERMEDIATE_SIZE, DIM, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    @patch_call(forward)
    def __call__(self) -> None: ...


class RMSNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(DIM))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + NORM_EPS)

    def forward(self, x: Tensor) -> Tensor:
        return self._norm(x.float()).type_as(x) * self.weight

    @patch_call(forward)
    def __call__(self) -> None: ...


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
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
