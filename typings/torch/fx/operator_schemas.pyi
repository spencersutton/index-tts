import inspect
from collections.abc import Callable
from typing import Any, NamedTuple

import torch

from ._compatibility import compatibility
from .node import Argument

__all__ = [
    "ArgsKwargsPair",
    "check_for_mutable_operation",
    "create_type_hint",
    "get_signature_for_torch_op",
    "normalize_function",
    "normalize_module",
    "type_matches",
]

@compatibility(is_backward_compatible=False)
class ArgsKwargsPair(NamedTuple):
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

_manual_overrides: dict[Callable, list[inspect.Signature]] = ...

class _FakeGlobalNamespace:
    def __getattr__(self, name): ...

_type_eval_globals = ...
_SCHEMA_TO_SIGNATURE_CACHE: dict[tuple[str, str], inspect.Signature] = ...

@compatibility(is_backward_compatible=False)
def check_for_mutable_operation(target: Callable, args: tuple[Argument, ...], kwargs: dict[str, Argument]): ...
@compatibility(is_backward_compatible=False)
def get_signature_for_torch_op(op: Callable, return_schemas: bool = ...): ...
@compatibility(is_backward_compatible=False)
def create_type_hint(x): ...
@compatibility(is_backward_compatible=False)
def type_matches(signature_type: Any, argument_type: Any): ...
@compatibility(is_backward_compatible=False)
def normalize_function(
    target: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = ...,
    arg_types: tuple[Any] | None = ...,
    kwarg_types: dict[str, Any] | None = ...,
    normalize_to_only_use_kwargs: bool = ...,
) -> ArgsKwargsPair | None: ...
@compatibility(is_backward_compatible=False)
def normalize_module(
    root: torch.nn.Module,
    target: str,
    args: tuple[Any],
    kwargs: dict[str, Any] | None = ...,
    normalize_to_only_use_kwargs: bool = ...,
) -> ArgsKwargsPair | None: ...
