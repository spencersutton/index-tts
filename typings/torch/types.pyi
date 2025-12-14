import os
from builtins import (  # noqa: F401
    bool as _bool,
)
from builtins import (
    bytes as _bytes,
)
from builtins import (
    complex as _complex,
)
from builtins import (
    float as _float,
)
from builtins import (
    int as _int,
)
from builtins import (
    str as _str,
)
from collections.abc import Sequence
from typing import IO, TYPE_CHECKING, Any, Self, TypeAlias

from torch import (
    DispatchKey as DispatchKey,
)
from torch import (
    Size as Size,
)
from torch import (
    SymBool as SymBool,
)
from torch import (
    SymFloat as SymFloat,
)
from torch import (
    SymInt as SymInt,
)
from torch import (
    Tensor as Tensor,
)
from torch import (
    device as _device,
)
from torch import (
    dtype as _dtype,
)
from torch import layout as _layout
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
