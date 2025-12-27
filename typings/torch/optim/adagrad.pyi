from torch import Tensor

from .optimizer import Optimizer, ParamsT, _use_grad_for_differentiable

__all__ = ["Adagrad", "adagrad"]

class Adagrad(Optimizer):
    r"""
    Implements Adagrad algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{12mm}    \tau \text{ (initial accumulator value)}, \: \eta\text{ (lr decay)}\\
            &\textbf{initialize} :  state\_sum_0 \leftarrow \tau                          \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \tilde{\gamma}    \leftarrow \gamma / (1 +(t-1) \eta)                  \\
            &\hspace{5mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda \theta_{t-1}                             \\
            &\hspace{5mm}state\_sum_t  \leftarrow  state\_sum_{t-1} + g^2_t                      \\
            &\hspace{5mm}\theta_t \leftarrow
                \theta_{t-1}- \tilde{\gamma} \frac{g_t}{\sqrt{state\_sum_t}+\epsilon}            \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        initial_accumulator_value (float, optional): initial value of the
            sum of squares of gradients (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
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
        fused (bool, optional): whether the fused implementation (CPU only) is used.
            Currently, `torch.float64`, `torch.float32`, `torch.float16`, and `torch.bfloat16`
            are supported. (default: None). Please note that the fused implementations does not
            support sparse or complex gradients.
    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        lr_decay: float = ...,
        weight_decay: float = ...,
        initial_accumulator_value: float = ...,
        eps: float = ...,
        foreach: bool | None = ...,
        *,
        maximize: bool = ...,
        differentiable: bool = ...,
        fused: bool | None = ...,
    ) -> None: ...
    def __setstate__(self, state) -> None: ...
    def share_memory(self) -> None:
        """Calls tensor.share_memory_() on the state sum tensors."""
    @_use_grad_for_differentiable
    def step(self, closure=...) -> None:
        """
        Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

def adagrad(
    params: list[Tensor],
    grads: list[Tensor],
    state_sums: list[Tensor],
    state_steps: list[Tensor],
    fused: bool | None = ...,
    grad_scale: Tensor | None = ...,
    found_inf: Tensor | None = ...,
    has_sparse_grad: bool = ...,
    foreach: bool | None = ...,
    differentiable: bool = ...,
    has_complex: bool = ...,
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    maximize: bool,
) -> None:
    """
    Functional API that performs Adagrad algorithm computation.

    See :class:`~torch.optim.Adagrad` for details.
    """
