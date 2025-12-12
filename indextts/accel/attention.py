# pyright: reportMissingImports=false, reportUnknownParameterType=false
import sys
from dataclasses import dataclass

if sys.platform == "darwin":
    msg = "flash attention is not supported on MacOS."
    raise ImportError(msg)

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from torch import Tensor, nn


@dataclass
class ForwardContext:
    is_prefill: bool = False
    cu_seqlens_q: Tensor | None = None
    cu_seqlens_k: Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: Tensor | None = None
    context_lens: Tensor | None = None
    block_tables: Tensor | None = None


_forward_context = ForwardContext()


def get_forward_context() -> ForwardContext:
    return _forward_context


def set_forward_context(
    is_prefill: bool,
    cu_seqlens_q: Tensor | None = None,
    cu_seqlens_k: Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: Tensor | None = None,
    context_lens: Tensor | None = None,
    block_tables: Tensor | None = None,
) -> None:
    global _forward_context  # noqa: PLW0603
    _forward_context = ForwardContext(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
    )


def reset_forward_context() -> None:
    global _forward_context  # noqa: PLW0603
    _forward_context = ForwardContext()


@triton.jit
def store_kvcache_kernel(
    key_ptr: tl.pointer_type,
    key_stride: int,
    value_ptr: tl.pointer_type,
    value_stride: int,
    k_cache_ptr: tl.pointer_type,
    v_cache_ptr: tl.pointer_type,
    slot_mapping_ptr: tl.pointer_type,
    D: tl.constexpr,  # noqa: N803
) -> None:
    BLOCK_SIZE: tl.constexpr = 2048
    idx: int = tl.program_id(0)
    slot: int = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    d_offset: int = 0
    while d_offset < D:
        cur_block_size: int = min(BLOCK_SIZE, D - d_offset)
        key_offsets = idx * key_stride + d_offset + tl.arange(0, BLOCK_SIZE)
        value_offsets = idx * value_stride + d_offset + tl.arange(0, BLOCK_SIZE)
        cache_offsets = slot * D + d_offset + tl.arange(0, BLOCK_SIZE)

        mask = tl.arange(0, BLOCK_SIZE) < cur_block_size
        key = tl.load(key_ptr + key_offsets, mask=mask, other=0.0)
        value = tl.load(value_ptr + value_offsets, mask=mask, other=0.0)
        tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
        tl.store(v_cache_ptr + cache_offsets, value, mask=mask)

        d_offset += BLOCK_SIZE


def store_kvcache(
    key: Tensor,
    value: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    slot_mapping: Tensor,
) -> None:
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[N,](  # pyright: ignore[reportIndexIssue]
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D,
    )


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        context = get_forward_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        else:
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
        return o
