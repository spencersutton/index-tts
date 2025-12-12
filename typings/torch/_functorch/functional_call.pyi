import torch
import torch.nn as nn
from collections.abc import Sequence
from typing import Any, Optional, Union
from torch import Tensor
from torch._functorch.utils import exposed_in

@exposed_in("torch.func")
def functional_call(
    module: torch.nn.Module,
    parameter_and_buffer_dicts: Union[dict[str, Tensor], Sequence[dict[str, Tensor]]],
    args: Optional[Union[Any, tuple]] = ...,
    kwargs: Optional[dict[str, Any]] = ...,
    *,
    tie_weights: bool = ...,
    strict: bool = ...,
): ...
@exposed_in("torch.func")
def stack_module_state(models: Union[Sequence[nn.Module], nn.ModuleList]) -> tuple[dict[str, Any], dict[str, Any]]: ...
def construct_stacked_leaf(tensors: Union[tuple[Tensor, ...], list[Tensor]], name: str) -> Tensor: ...
