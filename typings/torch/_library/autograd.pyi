import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from torch import _ops

class InfoProtocol(Protocol):
    _backward_fn: Callable | None
    _setup_context_fn: Callable | None

@dataclasses.dataclass
class Info:
    _backward_fn: Callable | None
    _setup_context_fn: Callable | None

def make_autograd_impl(op: _ops.OpOverload, info: InfoProtocol) -> Callable: ...
def supports_tensorlist(cls: Any) -> Any: ...
def not_list_of_tensor(tree):  # -> bool:
    ...
def not_list_of_optional_tensor(tree):  # -> bool:
    ...

flatten = ...
unflatten = ...
spec_t = ...
