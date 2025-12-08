from typing import Any
from warnings import deprecated

import torch
from torch import Tensor

__all__ = ["functional_call"]

@deprecated(
    "`torch.nn.utils.stateless.functional_call` is deprecated as of PyTorch 2.0 "
    "and will be removed in a future version of PyTorch. "
    "Please use `torch.func.functional_call` instead which is a drop-in replacement.",
    category=FutureWarning,
)
def functional_call(
    module: torch.nn.Module,
    parameters_and_buffers: dict[str, Tensor],
    args: Any | tuple | None = ...,
    kwargs: dict[str, Any] | None = ...,
    *,
    tie_weights: bool = ...,
    strict: bool = ...,
) -> Any: ...
