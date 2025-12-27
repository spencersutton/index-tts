from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import ParamSpec, TypeVar

import torch._ops
from torch._C import DispatchKey

__all__ = ["enable_pre_dispatch", "enable_python_dispatcher", "no_python_dispatcher"]
no_python_dispatcher = ...
enable_python_dispatcher = ...
enable_pre_dispatch = ...
CROSSREF_FUNCTIONALIZE = ...
_P = ParamSpec("_P")
_T = TypeVar("_T")

def all_py_loaded_overloads() -> Iterator[torch._ops.OpOverload]:
    """
    Warning: the set of overloads this will report is very subtle.  It is precisely
    the set of torch.ops functions that have actually been accessed from Python
    (e.g., we actually called torch.ops.aten.blah at some point.  This is DIFFERENT
    from the set of registered operators, which will in general be a larger set,
    as this would include all operators which we ran C++ static initializers or
    Python operator registration on.  This does not eagerly populate the list on
    torch.ops.aten; this list is lazy!

    In other words, this is good for traversing over everything that has an
    OpOverload object allocated in Python.  We use it for cache invalidation, but
    don't rely on this list being complete.

    Note that even if we did report all C++ registered overloads, this isn't guaranteed
    to be complete either, as a subsequent lazy load of a library which triggers more
    registrations could add more things to the set.
    """

@contextmanager
def suspend_functionalization(): ...
def check_tensor_metadata_matches(nv, rv, desc): ...
def check_metadata_matches(n, r, desc): ...

class Lit:
    def __init__(self, s) -> None: ...

def make_crossref_functionalize[P, T](
    op: torch._ops.OpOverload[_P, _T], final_key: DispatchKey
) -> Callable[_P, _T] | DispatchKey: ...
@contextmanager
def enable_crossref_functionalize(): ...
