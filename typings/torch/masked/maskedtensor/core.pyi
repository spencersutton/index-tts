from typing import Any, TypeIs

import torch

__all__ = ["MaskedTensor", "is_masked_tensor"]

def is_masked_tensor(obj: Any, /) -> TypeIs[MaskedTensor]:
    """
    Returns True if the input is a MaskedTensor, else False

    Args:
        a: any input

    Examples:

        >>> # xdoctest: +SKIP
        >>> from torch.masked import MaskedTensor
        >>> data = torch.arange(6).reshape(2, 3)
        >>> mask = torch.tensor([[True, False, False], [True, True, False]])
        >>> mt = MaskedTensor(data, mask)
        >>> is_masked_tensor(mt)
        True
    """

class MaskedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, mask, requires_grad=...): ...
    def __init__(self, data, mask, requires_grad=...) -> None: ...
    @classmethod
    def __torch_function__(cls, func, types, args=..., kwargs=...) -> _NotImplementedType: ...
    @classmethod
    def unary(cls, fn, data, mask) -> MaskedTensor: ...
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs) -> Any | _NotImplementedType: ...
    def __lt__(self, other) -> bool: ...
    def to_tensor(self, value) -> Any: ...
    def get_data(self) -> Any | None: ...
    def get_mask(self): ...
    def is_sparse_coo(self) -> bool: ...
    def is_sparse_csr(self) -> bool: ...
    @property
    def is_sparse(self) -> bool: ...
