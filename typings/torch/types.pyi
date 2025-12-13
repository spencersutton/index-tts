from builtins import (  # noqa: F401
    bool as _bool,
    bytes as _bytes,
    complex as _complex,
    float as _float,
    int as _int,
    str as _str,
)
from torch import layout as _layout

import os
from collections.abc import Sequence
from typing import Any, IO, Self, TYPE_CHECKING, TypeAlias
from torch import (
    DispatchKey as DispatchKey,
    Size as Size,
    SymBool as SymBool,
    SymFloat as SymFloat,
    SymInt as SymInt,
    Tensor as Tensor,
    device as _device,
    dtype as _dtype,
)
from torch.autograd.graph import GradientEdge

if TYPE_CHECKING: ...
__all__ = ["Device", "FileLike", "Number", "Storage"]
type _TensorOrTensors = Tensor | Sequence[Tensor]
type _TensorOrTensorsOrGradEdge = Tensor | Sequence[Tensor] | GradientEdge | Sequence[GradientEdge]
type _size = Size | list[int] | tuple[int, ...]
type _symsize = Size | Sequence[int | SymInt]
type _dispatchkey = str | DispatchKey
type IntLikeType = int | SymInt
type FloatLikeType = float | SymFloat
type BoolLikeType = bool | SymBool
py_sym_types = ...
type PySymType = SymInt | SymFloat | SymBool
type Number = int | float | bool
_Number = ...
type FileLike = str | os.PathLike[str] | IO[bytes]
type Device = _device | str | int | None

class Storage:
    _cdata: int
    device: _device
    dtype: _dtype
    _torch_load_uninitialized: bool
    def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...
    def element_size(self) -> int: ...
    def is_shared(self) -> bool: ...
    def share_memory_(self) -> Self: ...
    def nbytes(self) -> int: ...
    def cpu(self) -> Self: ...
    def data_ptr(self) -> int: ...
    def from_file(self, filename: str, shared: bool = ..., nbytes: int = ...) -> Self: ...
