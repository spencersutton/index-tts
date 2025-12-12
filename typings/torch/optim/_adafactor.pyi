import torch
from typing import Optional, Union
from torch import Tensor
from .optimizer import Optimizer, ParamsT, _disable_dynamo_if_unsupported

__all__ = ["Adafactor", "adafactor"]

class Adafactor(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = ...,
        beta2_decay: float = ...,
        eps: tuple[Optional[float], float] = ...,
        d: float = ...,
        weight_decay: float = ...,
        *,
        foreach: Optional[bool] = ...,
        maximize: bool = ...,
    ) -> None: ...
    def __setstate__(self, state):  # -> None:
        ...
    @torch.no_grad()
    def step(self, closure=...):  # -> None:

        ...

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adafactor)
def adafactor(
    params: list[Tensor],
    grads: list[Tensor],
    row_vars: list[Optional[Tensor]],
    col_vars: list[Optional[Tensor]],
    variances: list[Optional[Tensor]],
    state_steps: list[Tensor],
    foreach: Optional[bool] = ...,
    grad_scale: Optional[Tensor] = ...,
    found_inf: Optional[Tensor] = ...,
    has_complex: bool = ...,
    *,
    d: float,
    lr: Union[float, Tensor],
    beta2_decay: float,
    weight_decay: float,
    eps1: float,
    eps2: float,
    maximize: bool,
):  # -> None:

    ...
