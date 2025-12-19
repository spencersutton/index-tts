import torch
from torch import Tensor

__all__ = ["lobpcg"]

class LOBPCGAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A: Tensor,
        k: int | None = ...,
        B: Tensor | None = ...,
        X: Tensor | None = ...,
        n: int | None = ...,
        iK: Tensor | None = ...,
        niter: int | None = ...,
        tol: float | None = ...,
        largest: bool | None = ...,
        method: str | None = ...,
        tracker: None = ...,
        ortho_iparams: dict[str, int] | None = ...,
        ortho_fparams: dict[str, float] | None = ...,
        ortho_bparams: dict[str, bool] | None = ...,
    ) -> tuple[Tensor, Tensor]: ...
    @staticmethod
    def backward(ctx, D_grad, U_grad): ...

def lobpcg(
    A: Tensor,
    k: int | None = ...,
    B: Tensor | None = ...,
    X: Tensor | None = ...,
    n: int | None = ...,
    iK: Tensor | None = ...,
    niter: int | None = ...,
    tol: float | None = ...,
    largest: bool | None = ...,
    method: str | None = ...,
    tracker: None = ...,
    ortho_iparams: dict[str, int] | None = ...,
    ortho_fparams: dict[str, float] | None = ...,
    ortho_bparams: dict[str, bool] | None = ...,
) -> tuple[Tensor, Tensor]: ...

class LOBPCG:
    def __init__(
        self,
        A: Tensor | None,
        B: Tensor | None,
        X: Tensor,
        iK: Tensor | None,
        iparams: dict[str, int],
        fparams: dict[str, float],
        bparams: dict[str, bool],
        method: str,
        tracker: None,
    ) -> None: ...
    def update(self): ...
    def update_residual(self): ...
    def update_converged_count(self): ...
    def stop_iteration(self): ...
    def run(self): ...
    @torch.jit.unused
    def call_tracker(self): ...

LOBPCG_call_tracker_orig = ...

def LOBPCG_call_tracker(self): ...
