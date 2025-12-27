from enum import Enum
from typing import Any

import torch
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

class _EffectType(Enum):
    ORDERED = ...

type OpType = torch._ops.HigherOrderOperator | torch._ops.OpOverload
SIDE_EFFECTS = ...

class WithEffects(HigherOrderOperator):
    """
    with_effects(token, op, args, kwargs) -> (new_token, op_results)

    This HOP helps ensure ordering between side effectful ops like prints or ops
    using torchbind objects. This is needed to ensure a traced graph from
    AOTAutograd is functional so that future optimization passes do not reorder
    these operators. This is done through threading "effect tokens" through the
    graph to enforce data dependence between side effectful ops.

    The tokens are basically dummy values (torch.tensor([])). We create a token
    per "effect type", which are enumerated in the _EffectType enum.
    """
    def __init__(self) -> None: ...
    def __call__(self, token, op: OpType, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> tuple[Any, ...]: ...

with_effects = ...

def has_aliasing(op: OpType): ...
def has_effects(op, args, kwargs) -> bool: ...
def get_effect_key(op, args, kwargs) -> _EffectType | None: ...
def new_token_tensor() -> torch.Tensor: ...
@with_effects.py_impl(DispatchKey.CompositeExplicitAutograd)
def with_effects_dense(
    token: torch.Tensor, op: torch._ops.OpOverload, *args: tuple[Any, ...], **kwargs: dict[str, Any]
) -> tuple[torch.Tensor, ...]: ...
@with_effects.py_impl(FakeTensorMode)
def with_effects_fake(
    mode, token: torch.Tensor, op: torch._ops.OpOverload, *args: tuple[Any, ...], **kwargs: dict[str, Any]
) -> tuple[torch.Tensor, ...]: ...
@with_effects.py_impl(ProxyTorchDispatchMode)
def with_effects_proxy(
    mode, token: torch.Tensor, op: torch._ops.OpOverload, *args: tuple[Any, ...], **kwargs: dict[str, Any]
) -> tuple[torch.Tensor, ...]: ...
def handle_effects(
    allow_token_discovery: bool,
    tokens: dict[_EffectType, torch.Tensor],
    op: OpType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """
    Args:
        allow_token_discovery: Whether or not we are discovering tokens. If this
        is true, we will create a token for every side effect type seen that
        does not have a token assigned yet.  If this is false, the tokens
        should've all been created ahead of time, so we will error if there is
        no token mapping to every effect type.

        tokens: Map of effect type to tokens. This is to chain operators of the
        same effects together so that they do not get reordered in later
        optimization passes.
    """
