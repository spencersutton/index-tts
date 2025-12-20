import torch

from ..generation.continuous_batching import PagedAttentionCache
from ..utils import is_flash_attn_2_available

if is_flash_attn_2_available(): ...

def paged_attention_forward(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor = ...,
    cache: PagedAttentionCache = ...,
    cumulative_seqlens_q=...,
    cumulative_seqlens_k=...,
    max_seqlen_q=...,
    max_seqlen_k=...,
    block_tables=...,
    implementation=...,
    **kwargs,
) -> torch.Tensor: ...
