from torch import Tensor

def flash_attn_varlen_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: Tensor | None,
    max_seqlen_k: int,
    cu_seqlens_k: Tensor | None,
    softmax_scale: float,
    causal: bool,
    block_table: Tensor | None,
) -> Tensor: ...
def flash_attn_with_kvcache(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    cache_seqlens: Tensor | None,
    block_table: Tensor | None,
    softmax_scale: float,
    causal: bool,
) -> Tensor: ...
