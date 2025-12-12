import torch
from typing import Optional
from torch import Tensor

"""Locally Optimal Block Preconditioned Conjugate Gradient methods."""
__all__ = ["lobpcg"]

class LOBPCGAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A: Tensor,
        k: Optional[int] = ...,
        B: Optional[Tensor] = ...,
        X: Optional[Tensor] = ...,
        n: Optional[int] = ...,
        iK: Optional[Tensor] = ...,
        niter: Optional[int] = ...,
        tol: Optional[float] = ...,
        largest: Optional[bool] = ...,
        method: Optional[str] = ...,
        tracker: None = ...,
        ortho_iparams: Optional[dict[str, int]] = ...,
        ortho_fparams: Optional[dict[str, float]] = ...,
        ortho_bparams: Optional[dict[str, bool]] = ...,
    ) -> tuple[Tensor, Tensor]: ...
    @staticmethod
    def backward(ctx, D_grad, U_grad):  # -> tuple[None, ...]:
        ...

def lobpcg(
    A: Tensor,
    k: Optional[int] = ...,
    B: Optional[Tensor] = ...,
    X: Optional[Tensor] = ...,
    n: Optional[int] = ...,
    iK: Optional[Tensor] = ...,
    niter: Optional[int] = ...,
    tol: Optional[float] = ...,
    largest: Optional[bool] = ...,
    method: Optional[str] = ...,
    tracker: None = ...,
    ortho_iparams: Optional[dict[str, int]] = ...,
    ortho_fparams: Optional[dict[str, float]] = ...,
    ortho_bparams: Optional[dict[str, bool]] = ...,
) -> tuple[Tensor, Tensor]: ...

class LOBPCG:
    def __init__(
        self,
        A: Optional[Tensor],
        B: Optional[Tensor],
        X: Tensor,
        iK: Optional[Tensor],
        iparams: dict[str, int],
        fparams: dict[str, float],
        bparams: dict[str, bool],
        method: str,
        tracker: None,
    ) -> None: ...
    def update(self):  # -> None:

        ...
    def update_residual(self):  # -> None:

        ...
    def update_converged_count(self):  # -> int:

        ...
    def stop_iteration(self):  # -> bool:

        ...
    def run(self):  # -> None:

        ...
    @torch.jit.unused
    def call_tracker(self):  # -> None:

        ...

LOBPCG_call_tracker_orig = ...

def LOBPCG_call_tracker(self):  # -> None:
    ...
