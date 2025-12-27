from torch import Tensor

from .optimizer import Optimizer, ParamsT, _disable_dynamo_if_unsupported, _use_grad_for_differentiable

__all__ = ["Adadelta", "adadelta"]

class Adadelta(Optimizer):
    r"""
    Implements Adadelta algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)},
                \: f(\theta) \text{ (objective)}, \: \rho \text{ (decay)},
                \: \lambda \text{ (weight decay)}                                                \\
            &\textbf{initialize} :  v_0  \leftarrow 0 \: \text{ (square avg)},
                \: u_0 \leftarrow 0 \: \text{ (accumulate variables)}                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm} v_t      \leftarrow v_{t-1} \rho + g^2_t (1 - \rho)                    \\
            &\hspace{5mm}\Delta x_t    \leftarrow   \frac{\sqrt{u_{t-1} +
                \epsilon }}{ \sqrt{v_t + \epsilon}  }g_t \hspace{21mm}                           \\
            &\hspace{5mm} u_t  \leftarrow   u_{t-1}  \rho +
                 \Delta x^2_t  (1 - \rho)                                                        \\
            &\hspace{5mm}\theta_t      \leftarrow   \theta_{t-1} - \gamma  \Delta x_t            \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `ADADELTA: An Adaptive Learning Rate Method`_.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9). A higher value of `rho` will
            result in a slower average, which can be helpful for preventing
            oscillations in the learning process.
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. Note that the foreach implementation uses
            ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
            being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
            parameters through the optimizer at a time or switch this flag to False (default: None)
        capturable (bool, optional): whether this instance is safe to
            capture in a graph, whether for CUDA graphs or for torch.compile support.
            Tensors are only capturable when on supported :ref:`accelerators<accelerators>`.
            Passing True can impair ungraphed performance, so if you don't intend to graph
            capture this instance, leave it False (default: False)
        maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)
        differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)

    .. _ADADELTA\: An Adaptive Learning Rate Method:
        https://arxiv.org/abs/1212.5701
    """
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        rho: float = ...,
        eps: float = ...,
        weight_decay: float = ...,
        foreach: bool | None = ...,
        *,
        capturable: bool = ...,
        maximize: bool = ...,
        differentiable: bool = ...,
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

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adadelta)
def adadelta(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    acc_deltas: list[Tensor],
    state_steps: list[Tensor],
    capturable: bool = ...,
    foreach: bool | None = ...,
    differentiable: bool = ...,
    has_complex: bool = ...,
    *,
    lr: float,
    rho: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
) -> None:
    """
    Functional API that performs Adadelta algorithm computation.

    See :class:`~torch.optim.Adadelta` for details.
    """
