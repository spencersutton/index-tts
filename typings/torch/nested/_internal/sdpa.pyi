import torch
from typing import Optional

log = ...

def jagged_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = ...,
    dropout_p=...,
    is_causal=...,
    scale=...,
    enable_gqa=...,
):  # -> Tensor:
    ...
