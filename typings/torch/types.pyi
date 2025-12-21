# pyright: reportUnusedImport=false
import os
from collections.abc import Sequence
from typing import IO, Any, Self

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

__all__ = ["Device", "FileLike", "Number", "Storage"]

type _TensorOrTensors = Tensor | Sequence[Tensor]
type _TensorOrTensorsOrGradEdge = Tensor | Sequence[Tensor] | GradientEdge | Sequence[GradientEdge]

type _size = Size | list[int] | tuple[int, ...]
type _symsize = Size | Sequence[int | SymInt]
type _dispatchkey = str | DispatchKey

type IntLikeType = int | SymInt
type FloatLikeType = float | SymFloat
type BoolLikeType = bool | SymBool
type PySymType = SymInt | SymFloat | SymBool
type Number = int | float | bool
type FileLike = str | os.PathLike[str] | IO[bytes]
type Device = _device | str | int | None

class Storage:
    _cdata: int
    device: _device
    dtype: _dtype
    _torch_load_uninitialized: bool

    def __deepcopy__(self, memo: dict[int, Any]) -> Self: ...
    def _new_shared(self, size: int) -> Self: ...
    def _write_file(
        self,
        f: Any,
        is_real_file: bool,
        save_size: bool,
        element_size: int,
    ) -> None: ...
    def element_size(self) -> int: ...
    def is_shared(self) -> bool: ...
    def share_memory_(self) -> Self: ...
    def nbytes(self) -> int: ...
    def cpu(self) -> Self: ...
    def data_ptr(self) -> int: ...
    def from_file(
        self,
        filename: str,
        shared: bool = False,
        nbytes: int = 0,
    ) -> Self: ...
    def _new_with_file(
        self,
        f: Any,
        element_size: int,
    ) -> Self: ...
