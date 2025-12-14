from torch import Tensor
from torch.types import _device as Device
from torch.types import _dtype as DType

__all__ = [
    "as_nested_tensor",
    "masked_select",
    "narrow",
    "nested_tensor",
    "nested_tensor_from_jagged",
    "to_padded_tensor",
]

def as_nested_tensor(
    ts: Tensor | list[Tensor] | tuple[Tensor, ...], dtype: DType | None = ..., device: Device | None = ..., layout=...
) -> Tensor: ...

to_padded_tensor = ...

def nested_tensor(tensor_list, *, dtype=..., layout=..., device=..., requires_grad=..., pin_memory=...) -> Tensor: ...
def narrow(tensor: Tensor, dim: int, start: int | Tensor, length: int | Tensor, layout=...) -> Tensor: ...
def nested_tensor_from_jagged(
    values: Tensor,
    offsets: Tensor | None = ...,
    lengths: Tensor | None = ...,
    jagged_dim: int | None = ...,
    min_seqlen: int | None = ...,
    max_seqlen: int | None = ...,
) -> Tensor: ...
def masked_select(tensor: Tensor, mask: Tensor) -> Tensor: ...
