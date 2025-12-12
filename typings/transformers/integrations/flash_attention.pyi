import torch
from typing import Optional

logger = ...
_use_top_left_mask = ...

def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = ...,
    scaling: Optional[float] = ...,
    sliding_window: Optional[int] = ...,
    softcap: Optional[float] = ...,
    **kwargs,
) -> tuple[torch.Tensor, None]: ...
