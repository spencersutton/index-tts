from typing import Generic, TypeVar

import torch

__all__ = ["Await"]
W = TypeVar("W")

class _PyAwaitMeta(type(torch._C._Await), type(Generic)): ...
class _Await(torch._C._Await, Generic[W], metaclass=_PyAwaitMeta): ...
