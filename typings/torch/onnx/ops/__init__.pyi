from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch

__all__ = ["aten_decompositions", "attention", "rotary_embedding", "symbolic", "symbolic_multi_out"]
if TYPE_CHECKING: ...
_TORCH_DTYPE_TO_ONNX_DTYPE = ...

def aten_decompositions() -> dict[torch._ops.OpOverload, Callable]: ...
def symbolic(
    domain_op: str,
    /,
    inputs: Sequence[torch.Tensor | None],
    attrs: dict[
        str,
        int | float | str | bool | Sequence[int] | Sequence[float] | Sequence[str] | Sequence[bool],
    ]
    | None = ...,
    *,
    dtype: torch.dtype | int,
    shape: Sequence[int | torch.SymInt],
    version: int | None = ...,
    metadata_props: dict[str, str] | None = ...,
) -> torch.Tensor: ...
def symbolic_multi_out(
    domain_op: str,
    /,
    inputs: Sequence[torch.Tensor | None],
    attrs: dict[
        str,
        int | float | str | bool | Sequence[int] | Sequence[float] | Sequence[str] | Sequence[bool],
    ]
    | None = ...,
    *,
    dtypes: Sequence[torch.dtype | int],
    shapes: Sequence[Sequence[int | torch.SymInt]],
    version: int | None = ...,
    metadata_props: dict[str, str] | None = ...,
) -> Sequence[torch.Tensor]: ...
def rotary_embedding(
    X: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    position_ids: torch.Tensor | None = ...,
    *,
    interleaved: bool = ...,
    num_heads: int = ...,
    rotary_embedding_dim: int = ...,
) -> torch.Tensor: ...
def attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attn_mask: torch.Tensor | None = ...,
    past_key: torch.Tensor | None = ...,
    past_value: torch.Tensor | None = ...,
    *,
    is_causal: bool = ...,
    kv_num_heads: int = ...,
    q_num_heads: int = ...,
    qk_matmul_output_mode: int = ...,
    scale: float | None = ...,
    softcap: float = ...,
    softmax_precision: int | None = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
