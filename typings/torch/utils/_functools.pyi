from collections.abc import Callable
from typing import Concatenate, ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")
_C = TypeVar("_C")
_cache_sentinel = ...

def cache_method[C, **P, T](f: Callable[Concatenate[_C, _P], _T]) -> Callable[Concatenate[_C, _P], _T]: ...
