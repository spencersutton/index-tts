from torch import Tensor

from .optimizer import Optimizer, ParamsT, _disable_dynamo_if_unsupported, _use_grad_for_differentiable

__all__ = ["ASGD", "asgd"]

class ASGD(Optimizer):
    """
    Implements Averaged Stochastic Gradient Descent.

    It has been proposed in `Acceleration of stochastic approximation by
    averaging`_.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): learning rate (default: 1e-2)
        lambd (float, optional): decay term (default: 1e-4)
        alpha (float, optional): power for eta update (default: 0.75)
        t0 (float, optional): point at which to start averaging (default: 1e6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. Note that the foreach implementation uses
            ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
            being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
            parameters through the optimizer at a time or switch this flag to False (default: None)
        maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)
        differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)
        capturable (bool, optional): whether this instance is safe to
            capture in a graph, whether for CUDA graphs or for torch.compile support.
            Tensors are only capturable when on supported :ref:`accelerators<accelerators>`.
            Passing True can impair ungraphed performance, so if you don't intend to graph
            capture this instance, leave it False (default: False)

    .. _Acceleration of stochastic approximation by averaging:
        https://meyn.ece.ufl.edu/wp-content/uploads/sites/77/archive/spm_files/Courses/ECE555-2011/555media/poljud92.pdf
    """
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        lambd: float = ...,
        alpha: float = ...,
        t0: float = ...,
        weight_decay: float = ...,
        foreach: bool | None = ...,
        maximize: bool = ...,
        differentiable: bool = ...,
        capturable: bool = ...,
    ) -> None: ...
    def __setstate__(self, state) -> None: ...
    @_use_grad_for_differentiable
    def step(self, closure=...) -> None:
        """
        Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_asgd)
def asgd(
    params: list[Tensor],
    grads: list[Tensor],
    axs: list[Tensor],
    mus: list[Tensor],
    etas: list[Tensor],
    state_steps: list[Tensor],
    foreach: bool | None = ...,
    maximize: bool = ...,
    differentiable: bool = ...,
    capturable: bool = ...,
    has_complex: bool = ...,
    *,
    lambd: float,
    lr: float,
    t0: float,
    alpha: float,
    weight_decay: float,
) -> None:
    """
    Functional API that performs asgd algorithm computation.

    See :class:`~torch.optim.ASGD` for details.
    """
