from collections.abc import Callable
from typing import Concatenate, Generic, ParamSpec, Protocol, TypeVar

import torch.nn as nn
from torch.distributed._composable_state import _State

_T = TypeVar("_T", covariant=True)
_P = ParamSpec("_P")

def generate_state_key(string=...):  # -> str:
    ...

STATE_KEY = ...
REGISTRY_KEY = ...

class RegistryItem: ...

_TState = TypeVar("_TState", bound=_State, covariant=True)
_M = TypeVar("_M", nn.Module, list[nn.Module])

class _ContractFn[**P, T, TState: _State](Protocol):
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T: ...
    def state(self, module: nn.Module) -> _TState: ...

def contract[TState: _State](
    state_cls: type[_TState] = ...,
) -> Callable[
    [Callable[Concatenate[_M, _P], _M]],
    _ContractFn[Concatenate[_M, _P], _M, _TState],
]: ...
