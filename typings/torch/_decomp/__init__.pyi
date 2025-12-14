import inspect
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import lru_cache, partial, wraps
from itertools import chain
from typing import TYPE_CHECKING, Optional, ParamSpec, TypeVar, Union

import torch
import torch._decomp.decompositions
import torch._refs
import torch.library
from torch._ops import HigherOrderOperator, OperatorBase, OpOverload, OpOverloadPacket
from torch._prims_common import CustomOutParamAnnotation
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.export.decomp_utils import CustomDecompTable
from torch.utils import _pytree as pytree

if TYPE_CHECKING: ...
__all__ = [
    "decomposition_table",
    "pre_autograd_decomposition_table",
    "meta_table",
    "register_decomposition",
    "get_decompositions",
    "core_aten_decompositions",
    "_should_decompose_because_unsafe_op",
]
_T = TypeVar("_T")
_P = ParamSpec("_P")
global_decomposition_table: dict[str, dict[torch._ops.OperatorBase, Callable]] = ...
decomposition_table = ...
pre_autograd_decomposition_table = ...
meta_table = ...

def register_decomposition(
    aten_op, registry=..., *, type=..., unsafe=...
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
def get_decompositions(
    aten_ops: Sequence[torch._ops.OperatorBase | OpOverloadPacket], type: str = ...
) -> dict[torch._ops.OperatorBase, Callable]: ...
def remove_decompositions(
    decompositions: dict[torch._ops.OperatorBase, Callable], aten_ops: Sequence[OpOverload | OpOverloadPacket]
) -> None: ...
def core_aten_decompositions() -> CustomDecompTable: ...
