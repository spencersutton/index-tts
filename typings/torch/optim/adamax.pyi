from torch import Tensor

from .optimizer import Optimizer, ParamsT, _disable_dynamo_if_unsupported, _use_grad_for_differentiable

__all__ = ["Adamax", "adamax"]

class Adamax(Optimizer):
    r"""
    Implements Adamax algorithm (a variant of Adam based on infinity norm).

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)},
                \: \lambda \text{ (weight decay)},                                                \\
            &\hspace{13mm}    \epsilon \text{ (epsilon)}                                          \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                u_0 \leftarrow 0 \text{ ( infinity norm)}                                 \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t      \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t               \\
            &\hspace{5mm}u_t      \leftarrow   \mathrm{max}(\beta_2 u_{t-1}, |g_{t}|+\epsilon)   \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1} - \frac{\gamma m_t}{(1-\beta^t_1) u_t} \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adam: A Method for Stochastic Optimization`_.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
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

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        betas: tuple[float, float] = ...,
        eps: float = ...,
        weight_decay: float = ...,
        foreach: bool | None = ...,
        *,
        maximize: bool = ...,
        differentiable: bool = ...,
        capturable: bool = ...,
    ) -> None: ...
    def __setstate__(self, state) -> None: ...
    @_use_grad_for_differentiable
    def step(self, closure=...) -> None:
        """
        Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adamax)
def adamax(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_infs: list[Tensor],
    state_steps: list[Tensor],
    foreach: bool | None = ...,
    maximize: bool = ...,
    differentiable: bool = ...,
    capturable: bool = ...,
    has_complex: bool = ...,
    *,
    eps: float,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
) -> None:
    """
    Functional API that performs adamax algorithm computation.

    See :class:`~torch.optim.Adamax` for details.
    """
