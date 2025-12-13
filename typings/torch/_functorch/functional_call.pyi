import torch
import torch.nn as nn
from collections.abc import Sequence
from typing import Any, Optional, Union
from torch import Tensor
from torch._functorch.utils import exposed_in

@exposed_in("torch.func")
def functional_call(
    module: torch.nn.Module,
    parameter_and_buffer_dicts: dict[str, Tensor] | Sequence[dict[str, Tensor]],
    args: Any | tuple | None = ...,
    kwargs: dict[str, Any] | None = ...,
    *,
    tie_weights: bool = ...,
    strict: bool = ...,
): ...
@exposed_in("torch.func")
def stack_module_state(models: Sequence[nn.Module] | nn.ModuleList) -> tuple[dict[str, Any], dict[str, Any]]: ...
def construct_stacked_leaf(tensors: tuple[Tensor, ...] | list[Tensor], name: str) -> Tensor: ...
