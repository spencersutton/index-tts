# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from indextts.s2mel.modules.constants import BLOCK_SIZE, HIDDEN_DIM
from indextts.util import patch_call

NUM_HEADS = 8
DEPTH = 13
HEAD_DIM = HIDDEN_DIM // NUM_HEADS
ROPE_BASE = 10000
NORM_EPS = 1e-5
N_LAYER = DEPTH
N_HEAD = NUM_HEADS
DIM = HIDDEN_DIM


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


INTERMEDIATE_SIZE = find_multiple(int((8 * HIDDEN_DIM) / 3), 256)


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
    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.ModuleList(TransformerBlock() for _ in range(N_LAYER))
        self.norm = AdaptiveLayerNorm()

        self.freqs_cis: Tensor | None = None
        self.mask_cache: Tensor | None = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size: int, max_seq_length: int) -> None:
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.norm.project_layer.weight.dtype
        device = self.norm.project_layer.weight.device

        self.freqs_cis = precompute_freqs_cis(BLOCK_SIZE, HEAD_DIM, dtype).to(device)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).to(device)
        self.use_kv_cache = False
        self.layers_emit_skip = [i for i in range(N_LAYER) if i < N_LAYER // 2]
        self.layers_receive_skip = [i for i in range(N_LAYER) if i > N_LAYER // 2]

    def forward(self, x: Tensor, c: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        freqs_cis = self.freqs_cis[input_pos]
        skip_in_x_list = []
        for i, layer in enumerate(self.layers):
            if i in self.layers_receive_skip:
                skip_in_x = skip_in_x_list.pop(-1)
            else:
                skip_in_x = None
            x = layer(x, c, input_pos, freqs_cis, mask, skip_in_x)
            if i in self.layers_emit_skip:
                skip_in_x_list.append(x)
        return self.norm(x, c)

    @patch_call(forward)
    def __call__(self) -> None: ...


class TransformerBlock(nn.Module):
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
    def __init__(self) -> None:
        super().__init__()
        assert DIM % N_HEAD == 0

        total_head_dim = (N_HEAD + 2 * N_HEAD) * HEAD_DIM
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(DIM, total_head_dim, bias=False)
        self.wo = nn.Linear(HEAD_DIM * N_HEAD, DIM, bias=False)

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = N_HEAD * HEAD_DIM
        query_key_value: Tensor = self.wqkv(x)
        q, k, v = query_key_value.split((kv_size, kv_size, kv_size), dim=-1)

        q = q.view(bsz, seqlen, N_HEAD, HEAD_DIM)
        k = k.view(bsz, seqlen, N_HEAD, HEAD_DIM)
        v = v.view(bsz, seqlen, N_HEAD, HEAD_DIM)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]

        k = k.repeat_interleave(1, dim=1)
        v = v.repeat_interleave(1, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, HEAD_DIM * N_HEAD)
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


def precompute_freqs_cis(seq_len: int, n_elem: int, dtype: torch.dtype = torch.bfloat16) -> Tensor:
    freqs = torch.as_tensor(1.0 / (ROPE_BASE ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


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
