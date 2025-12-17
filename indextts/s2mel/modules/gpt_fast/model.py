# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from: https://github.com/meta-pytorch/gpt-fast/blob/main/model.py

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from indextts.util import patch_call


def _find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class AdaptiveLayerNorm(nn.Module):
    r"""Adaptive Layer Normalization."""

    def __init__(self, d_model: int, norm: "RMSNorm") -> None:
        super().__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    @override
    def forward(self, input: Tensor, embedding: Tensor | None = None) -> Tensor:
        if embedding is None:
            return self.norm(input)
        weight, bias = torch.split(
            self.project_layer(embedding),
            split_size_or_sections=self.d_model,
            dim=-1,
        )
        return weight * self.norm(input) + bias

    @patch_call(forward)
    def __call__(self) -> None: ...


@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int | None = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: int = 10000
    norm_eps: float = 1e-5
    context_dim: int = 0
    uvit_skip_connection: bool = False
    time_as_token: bool = False

    def __post_init__(self) -> None:
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = _find_multiple(n_hidden, 256)


class Transformer(nn.Module):
    max_batch_size = -1
    max_seq_length = -1
    freqs_cis: Tensor | None = None
    mask_cache: Tensor | None = None
    layers: "Iterable[TransformerBlock]"
    causal_mask: Tensor | None = None
    use_kv_cache: bool = False
    uvit_skip_connection: bool = False
    layers_emit_skip: Sequence[int] = []
    layers_receive_skip: Sequence[int] = []

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))

    def setup_caches(self, max_batch_size: int, max_seq_length: int) -> None:
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        max_seq_length = _find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

        self.freqs_cis = _precompute_freqs_cis(
            self.config.block_size,
            self.config.head_dim,
            self.config.rope_base,
        )
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.layers_emit_skip = [i for i in range(self.config.n_layer) if i < self.config.n_layer // 2]
        self.layers_receive_skip = [i for i in range(self.config.n_layer) if i > self.config.n_layer // 2]

    @override
    def forward(
        self,
        x: Tensor,
        c: Tensor,
        input_pos: Tensor,
        mask: Tensor,
    ) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"

        if self.freqs_cis.device != input_pos.device:
            raise ValueError(
                f"Caches must be initialized on the same device as input_pos: {self.freqs_cis.device} vs {input_pos.device}"
            )
        freqs_cis = self.freqs_cis[input_pos]
        skip_in_x_list: list[Tensor] = []
        for i, layer in enumerate(self.layers):
            skip_in_x = None
            if i in self.layers_receive_skip:
                skip_in_x = skip_in_x_list.pop(-1)
            x = layer(x, c, input_pos, freqs_cis, mask, skip_in_x)
            if i in self.layers_emit_skip:
                skip_in_x_list.append(x)
        return self.norm(x, c)

    @patch_call(forward)
    def __call__(self) -> None: ...


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention: Attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))
        self.attention_norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))
        self.skip_in_linear = nn.Linear(config.dim * 2, config.dim)

    @override
    def forward(
        self,
        x: Tensor,
        c: Tensor | None,
        input_pos: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        skip_in_x: Tensor | None = None,
    ) -> Tensor:
        if skip_in_x is not None:
            x = self.skip_in_linear(torch.cat([x, skip_in_x], dim=-1))
        h = x + self.attention(self.attention_norm(x, c), freqs_cis, mask, input_pos)
        return h + self.feed_forward(self.ffn_norm(h, c))

    @patch_call(forward)
    def __call__(self) -> None: ...


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, is_cross_attention: bool = False) -> None:
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        if is_cross_attention:
            self.wq = nn.Linear(config.dim, config.n_head * config.head_dim, bias=False)
            self.wkv = nn.Linear(
                config.context_dim,
                2 * config.n_local_heads * config.head_dim,
                bias=False,
            )
        else:
            self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim

    @override
    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor | None,
        input_pos: Tensor,
        context: Tensor | None = None,
        context_freqs_cis: Tensor | None = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        if context is None:
            q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)
            context_seqlen = seqlen
        else:
            q = self.wq(x)
            k, v = self.wkv(context).split([kv_size, kv_size], dim=-1)
            context_seqlen = context.shape[1]

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)

        q = _apply_rotary_emb(q, freqs_cis)
        k = _apply_rotary_emb(k, context_freqs_cis if context_freqs_cis is not None else freqs_cis)

        q, k, v = (x.transpose(1, 2) for x in (q, k, v))

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.head_dim * self.n_head)

        return self.wo(y)

    @patch_call(forward)
    def __call__(self) -> None: ...


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        assert config.intermediate_size is not None
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    @patch_call(forward)
    def __call__(self) -> None: ...


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    @override
    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    @patch_call(forward)
    def __call__(self) -> None: ...


def _precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 10000,
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)


def _apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
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
