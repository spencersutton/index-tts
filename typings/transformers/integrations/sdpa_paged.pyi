import torch
from typing import Optional

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...
def sdpa_attention_paged_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = ...,
    scaling: Optional[float] = ...,
    is_causal: Optional[bool] = ...,
    **kwargs,
) -> tuple[torch.Tensor, None]: ...
