"""Implementation for the NAdam algorithm."""

from torch import Tensor

from .optimizer import Optimizer, ParamsT, _disable_dynamo_if_unsupported, _use_grad_for_differentiable

__all__ = ["NAdam", "nadam"]

class NAdam(Optimizer):
    r"""
    Implements NAdam algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma_t \text{ (lr)}, \: \beta_1,\beta_2 \text{ (betas)},
                \: \theta_0 \text{ (params)}, \: f(\theta) \text{ (objective)}                   \\
            &\hspace{13mm} \: \lambda \text{ (weight decay)}, \:\psi \text{ (momentum decay)}    \\
            &\hspace{13mm} \: \textit{decoupled\_weight\_decay}, \:\textit{maximize}             \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0 \leftarrow 0 \text{ ( second moment)}                                 \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1}                                       \\
            &\hspace{5mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{10mm}\textbf{if} \: \textit{decoupled\_weight\_decay}                       \\
            &\hspace{15mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}                    \\
            &\hspace{10mm}\textbf{else}                                                          \\
            &\hspace{15mm} g_t \leftarrow g_t + \lambda \theta_{t-1}                             \\
            &\hspace{5mm} \mu_t \leftarrow \beta_1 \big(1 - \frac{1}{2}  0.96^{t \psi} \big)     \\
            &\hspace{5mm} \mu_{t+1} \leftarrow \beta_1 \big(1 - \frac{1}{2} 0.96^{(t+1)\psi}\big)\\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow \mu_{t+1} m_t/(1-\prod_{i=1}^{t+1}\mu_i)\\[-1.ex]
            & \hspace{11mm} + (1-\mu_t) g_t /(1-\prod_{i=1}^{t} \mu_{i})                         \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Incorporating Nesterov Momentum into Adam`_.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        momentum_decay (float, optional): momentum momentum_decay (default: 4e-3)
        decoupled_weight_decay (bool, optional): whether to decouple the weight
            decay as in AdamW to obtain NAdamW. If True, the algorithm does not
            accumulate weight decay in the momentum nor variance. (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. Note that the foreach implementation uses
            ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
            being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
            parameters through the optimizer at a time or switch this flag to False (default: None)
        maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)
        capturable (bool, optional): whether this instance is safe to
            capture in a graph, whether for CUDA graphs or for torch.compile support.
            Tensors are only capturable when on supported :ref:`accelerators<accelerators>`.
            Passing True can impair ungraphed performance, so if you don't intend to graph
            capture this instance, leave it False (default: False)
        differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)

    .. _Incorporating Nesterov Momentum into Adam:
        https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    """
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        betas: tuple[float, float] = ...,
        eps: float = ...,
        weight_decay: float = ...,
        momentum_decay: float = ...,
        decoupled_weight_decay: bool = ...,
        *,
        foreach: bool | None = ...,
        maximize: bool = ...,
        capturable: bool = ...,
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

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_nadam)
def nadam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    mu_products: list[Tensor],
    state_steps: list[Tensor],
    decoupled_weight_decay: bool = ...,
    foreach: bool | None = ...,
    capturable: bool = ...,
    differentiable: bool = ...,
    has_complex: bool = ...,
    maximize: bool = ...,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    momentum_decay: float,
    eps: float,
) -> None:
    """
    Functional API that performs NAdam algorithm computation.

    See :class:`~torch.optim.NAdam` for details.
    """
