import dataclasses
from collections.abc import Callable
from typing import Any, Protocol

from torch import _ops

class InfoProtocol(Protocol):
    _backward_fn: Callable | None
    _setup_context_fn: Callable | None

@dataclasses.dataclass
class Info:
    """Info(_backward_fn: Optional[Callable], _setup_context_fn: Optional[Callable])"""

    _backward_fn: Callable | None
    _setup_context_fn: Callable | None

def make_autograd_impl(op: _ops.OpOverload, info: InfoProtocol) -> Callable: ...
def supports_tensorlist(cls: Any) -> Any:
    """
    Allows a given autograd.Function class to support List[Tensor] inputs/outputs.

    Regular autograd.Function has a constraint that it only directly supports autograd for
    Tensors. Applying @supports_tensorlist enables an autograd.Function to support
    autograd for List[Tensor] inputs and outputs.
    """

def not_list_of_tensor(tree): ...
def not_list_of_optional_tensor(tree): ...

flatten = ...
unflatten = ...
spec_t = ...
