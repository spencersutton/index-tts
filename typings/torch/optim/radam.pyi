"""Implementation for the RAdam algorithm."""

from torch import Tensor

from .optimizer import Optimizer, ParamsT, _disable_dynamo_if_unsupported, _use_grad_for_differentiable

__all__ = ["RAdam", "radam"]

class RAdam(Optimizer):
    r"""
    Implements RAdam algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \beta_1, \beta_2
                \text{ (betas)}, \: \theta_0 \text{ (params)}, \:f(\theta) \text{ (objective)}, \:
                \lambda \text{ (weightdecay)}, \:\textit{maximize}                               \\
            &\hspace{13mm} \epsilon \text{ (epsilon)}, \textit{decoupled\_weight\_decay}         \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0 \leftarrow 0 \text{ ( second moment)},                                       \\
            &\hspace{18mm} \rho_{\infty} \leftarrow 2/(1-\beta_2) -1                      \\[-1.ex]
            &\rule{110mm}{0.4pt}  \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{6mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{12mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{6mm}\textbf{else}                                                           \\
            &\hspace{12mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{6mm} \theta_t \leftarrow \theta_{t-1}                                       \\
            &\hspace{6mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{12mm}\textbf{if} \: \textit{decoupled\_weight\_decay}                       \\
            &\hspace{18mm} \theta_t \leftarrow \theta_{t} - \gamma \lambda \theta_{t}            \\
            &\hspace{12mm}\textbf{else}                                                          \\
            &\hspace{18mm} g_t \leftarrow g_t + \lambda \theta_{t}                               \\
            &\hspace{6mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{6mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{6mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{6mm}\rho_t \leftarrow \rho_{\infty} -
                2 t \beta^t_2 /\big(1-\beta_2^t \big)                                    \\[0.1.ex]
            &\hspace{6mm}\textbf{if} \: \rho_t > 5                                               \\
            &\hspace{12mm} l_t \leftarrow \frac{\sqrt{ (1-\beta^t_2) }}{ \sqrt{v_t} +\epsilon  } \\
            &\hspace{12mm} r_t \leftarrow
      \sqrt{\frac{(\rho_t-4)(\rho_t-2)\rho_{\infty}}{(\rho_{\infty}-4)(\rho_{\infty}-2) \rho_t}} \\
            &\hspace{12mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t} r_t l_t        \\
            &\hspace{6mm}\textbf{else}                                                           \\
            &\hspace{12mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}                \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `On the variance of the adaptive learning rate and beyond`_.

    This implementation provides an option to use either the original weight_decay implementation as in Adam
    (where the weight_decay is applied to the gradient) or the one from AdamW (where weight_decay is applied
    to the weight) through the decoupled_weight_decay option. When decoupled_weight_decay is set to False
    (default), it uses the original Adam style weight decay, otherwise, it uses the AdamW style which
    corresponds more closely to the `author's implementation`_ in the RAdam paper. Further information
    about decoupled weight decay can be found in `Decoupled Weight Decay Regularization`_.


    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        decoupled_weight_decay (bool, optional): whether to decouple the weight
            decay as in AdamW to obtain RAdamW. If True, the algorithm does not
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

    .. _On the variance of the adaptive learning rate and beyond:
        https://arxiv.org/abs/1908.03265
    .. _author's implementation:
        https://github.com/LiyuanLucasLiu/RAdam
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

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_radam)
def radam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    decoupled_weight_decay: bool = ...,
    foreach: bool | None = ...,
    differentiable: bool = ...,
    capturable: bool = ...,
    has_complex: bool = ...,
    maximize: bool = ...,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
) -> None:
    """
    Functional API that performs RAdam algorithm computation.

    See :class:`~torch.optim.RAdam` for details.
    """
