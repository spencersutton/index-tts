"""Implementation for the RMSprop algorithm."""

from torch import Tensor

from .optimizer import Optimizer, ParamsT, _disable_dynamo_if_unsupported, _use_grad_for_differentiable

__all__ = ["RMSprop", "rmsprop"]

class RMSprop(Optimizer):
    r"""
    Implements RMSprop algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \alpha \text{ (alpha)}, \: \gamma \text{ (lr)},
                \: \theta_0 \text{ (params)}, \: f(\theta) \text{ (objective)}                   \\
            &\hspace{13mm}   \lambda \text{ (weight decay)},\: \mu \text{ (momentum)},
                \: centered, \: \epsilon \text{ (epsilon)}                                       \\
            &\textbf{initialize} : v_0 \leftarrow 0 \text{ (square average)}, \:
                \textbf{b}_0 \leftarrow 0 \text{ (buffer)}, \: g^{ave}_0 \leftarrow 0     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}v_t           \leftarrow   \alpha v_{t-1} + (1 - \alpha) g^2_t
                \hspace{8mm}                                                                     \\
            &\hspace{5mm} \tilde{v_t} \leftarrow v_t                                             \\
            &\hspace{5mm}if \: centered                                                          \\
            &\hspace{10mm} g^{ave}_t \leftarrow g^{ave}_{t-1} \alpha + (1-\alpha) g_t            \\
            &\hspace{10mm} \tilde{v_t} \leftarrow \tilde{v_t} -  \big(g^{ave}_{t} \big)^2        \\
            &\hspace{5mm}if \: \mu > 0                                                           \\
            &\hspace{10mm} \textbf{b}_t\leftarrow \mu \textbf{b}_{t-1} +
                g_t/ \big(\sqrt{\tilde{v_t}} +  \epsilon \big)                                   \\
            &\hspace{10mm} \theta_t \leftarrow \theta_{t-1} - \gamma \textbf{b}_t                \\
            &\hspace{5mm} else                                                                   \\
            &\hspace{10mm}\theta_t      \leftarrow   \theta_{t-1} -
                \gamma  g_t/ \big(\sqrt{\tilde{v_t}} + \epsilon \big)  \hspace{3mm}              \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to
    `lecture notes <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_ by G. Hinton.
    and centered version `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.
    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\gamma/(\sqrt{v} + \epsilon)` where :math:`\gamma`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): learning rate (default: 1e-2)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        momentum (float, optional): momentum factor (default: 0)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
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
        alpha: float = ...,
        eps: float = ...,
        weight_decay: float = ...,
        momentum: float = ...,
        centered: bool = ...,
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

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_rmsprop)
def rmsprop(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    grad_avgs: list[Tensor],
    momentum_buffer_list: list[Tensor],
    state_steps: list[Tensor],
    foreach: bool | None = ...,
    maximize: bool = ...,
    differentiable: bool = ...,
    capturable: bool = ...,
    has_complex: bool = ...,
    *,
    lr: float,
    alpha: float,
    eps: float,
    weight_decay: float,
    momentum: float,
    centered: bool,
) -> None:
    """
    Functional API that performs rmsprop algorithm computation.

    See :class:`~torch.optim.RMSProp` for details.
    """
