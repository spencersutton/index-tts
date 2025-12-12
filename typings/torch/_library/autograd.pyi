import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol
from torch import _ops

class InfoProtocol(Protocol):
    _backward_fn: Optional[Callable]
    _setup_context_fn: Optional[Callable]

@dataclasses.dataclass
class Info:
    _backward_fn: Optional[Callable]
    _setup_context_fn: Optional[Callable]

def make_autograd_impl(op: _ops.OpOverload, info: InfoProtocol) -> Callable: ...
def supports_tensorlist(cls: Any) -> Any: ...
def not_list_of_tensor(tree):  # -> bool:
    ...
def not_list_of_optional_tensor(tree):  # -> bool:
    ...

flatten = ...
unflatten = ...
spec_t = ...
