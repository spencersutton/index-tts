"""Implementation for the Resilient backpropagation."""

from torch import Tensor

from .optimizer import Optimizer, ParamsT, _disable_dynamo_if_unsupported, _use_grad_for_differentiable

__all__ = ["Rprop", "rprop"]

class Rprop(Optimizer):
    r"""
    Implements the resilient backpropagation algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \theta_0 \in \mathbf{R}^d \text{ (params)},f(\theta)
                \text{ (objective)},                                                             \\
            &\hspace{13mm}      \eta_{+/-} \text{ (etaplus, etaminus)}, \Gamma_{max/min}
                \text{ (step sizes)}                                                             \\
            &\textbf{initialize} :   g^0_{prev} \leftarrow 0,
                \: \eta_0 \leftarrow \text{lr (learning rate)}                                   \\
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \textbf{for} \text{  } i = 0, 1, \ldots, d-1 \: \mathbf{do}            \\
            &\hspace{10mm}  \textbf{if} \:   g^i_{prev} g^i_t  > 0                               \\
            &\hspace{15mm}  \eta^i_t \leftarrow \mathrm{min}(\eta^i_{t-1} \eta_{+},
                \Gamma_{max})                                                                    \\
            &\hspace{10mm}  \textbf{else if}  \:  g^i_{prev} g^i_t < 0                           \\
            &\hspace{15mm}  \eta^i_t \leftarrow \mathrm{max}(\eta^i_{t-1} \eta_{-},
                \Gamma_{min})                                                                    \\
            &\hspace{15mm}  g^i_t \leftarrow 0                                                   \\
            &\hspace{10mm}  \textbf{else}  \:                                                    \\
            &\hspace{15mm}  \eta^i_t \leftarrow \eta^i_{t-1}                                     \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1}- \eta_t \mathrm{sign}(g_t)             \\
            &\hspace{5mm}g_{prev} \leftarrow  g_t                                                \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to the paper
    `A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm
    <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.1417>`_.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, optional): learning rate (default: 1e-2)
        etas (Tuple[float, float], optional): pair of (etaminus, etaplus), that
            are multiplicative increase and decrease factors
            (default: (0.5, 1.2))
        step_sizes (Tuple[float, float], optional): a pair of minimal and
            maximal allowed step sizes (default: (1e-6, 50))
        capturable (bool, optional): whether this instance is safe to
            capture in a graph, whether for CUDA graphs or for torch.compile support.
            Tensors are only capturable when on supported :ref:`accelerators<accelerators>`.
            Passing True can impair ungraphed performance, so if you don't intend to graph
            capture this instance, leave it False (default: False)
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
    """
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        etas: tuple[float, float] = ...,
        step_sizes: tuple[float, float] = ...,
        *,
        capturable: bool = ...,
        foreach: bool | None = ...,
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

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_rprop)
def rprop(
    params: list[Tensor],
    grads: list[Tensor],
    prevs: list[Tensor],
    step_sizes: list[Tensor],
    state_steps: list[Tensor],
    foreach: bool | None = ...,
    capturable: bool = ...,
    maximize: bool = ...,
    differentiable: bool = ...,
    has_complex: bool = ...,
    *,
    step_size_min: float,
    step_size_max: float,
    etaminus: float,
    etaplus: float,
) -> None:
    """
    Functional API that performs rprop algorithm computation.

    See :class:`~torch.optim.Rprop` for details.
    """
