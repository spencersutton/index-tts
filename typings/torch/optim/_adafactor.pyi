import torch
from torch import Tensor

from .optimizer import Optimizer, ParamsT, _disable_dynamo_if_unsupported

__all__ = ["Adafactor", "adafactor"]

class Adafactor(Optimizer):
    r"""
    Implements Adafactor algorithm.

    .. math::
        \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \tau
                \text{(}\beta_2\text{ decay)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},    \\
            &\hspace{15mm}      \: \epsilon_1, \epsilon_2 \text{ (epsilons)}, \: d \text{(clipping threshold)}, \\
            &\hspace{15mm}      \: \lambda \text{(weight decay)},
                \: \textit{maximize}                                                             \\
            &\textbf{initialize} : \: R_0 \leftarrow 0 \text{ (second moment row factor)},       \\
            &\hspace{23mm} \: C_0 \leftarrow 0 \text{ (second moment col factor)},               \\
            &\hspace{23mm} \: \widehat{V}_0 \leftarrow 0 \text{ (second moment for vectors)}     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}G_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}G_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\widehat{\beta}_{2_t} \leftarrow 1 - t^{\tau}                           \\
            &\hspace{5mm}\rho_t         \leftarrow min(lr, \frac{1}{\sqrt{t}})                   \\
            &\hspace{5mm}\alpha_t       \leftarrow max(\epsilon_2,
                \text{RMS}(\theta_{t-1}))\rho_t                                                  \\
            &\hspace{5mm}\theta_t       \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}    \\
            &\hspace{5mm}\textbf{if} \: \text{dim}(G_t) > 1:                                     \\
            &\hspace{10mm}R_t           \leftarrow \widehat{\beta}_{2_t}R_{t-1}+
                (1-\widehat{\beta}_{2_t})(G_t \odot G_t) \cdot 1_m                               \\
            &\hspace{10mm}C_t           \leftarrow \widehat{\beta}_{2_t}C_{t-1}+
                (1-\widehat{\beta}_{2_t}) 1^\top_n \cdot (G_t \odot G_t)                         \\
            &\hspace{10mm}\widehat{V}_t \leftarrow
                \frac{R_t \cdot C_t}{max(1^\top_n \cdot R_t, \epsilon_1)}                        \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\widehat{V}_t \leftarrow \widehat{\beta}_{2_t}\widehat{V}_{t-1}+
                (1-\widehat{\beta}_{2_t}) \cdot (G_t \odot G_t)                                  \\
            &\hspace{5mm}U_t            \leftarrow
                \frac{G_t}{max(\sqrt{\widehat{V}_t}, \epsilon_1)}                                \\
            &\hspace{5mm}\widehat{U}_t  \leftarrow \frac{U_t}{max(1, \frac{\text{RMS}(U_t)}{d})} \\
            &\hspace{5mm}\theta_t       \leftarrow \theta_{t-1} - \alpha_t \widehat{U}_t         \\

            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
        \end{aligned}

    For further details regarding the algorithm we refer to `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`_.

    Args:
        params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named
        lr (float, Tensor, optional): unlike other optimizers, Adafactor does not require a
            learning rate, and Noam Shazeer and Mitchell Stern do not use lr at all.
            Deviating from the paper, this implementation uses lr for applying weight
            decay and as the maximum value for relative step size rho_t. Note that in
            the paper, a constant of 0.01 is used as the maximum value for relative
            step size, and so we set 0.01 as the default value. (default: 1e-2)
        beta2_decay (float, optional): the decay rate of beta2. beta2 standardly refers
            to the coefficient used for computing the running average of the gradient
            squared. (default: -0.8)
        eps (Tuple[float, float], optional): epsilon1 is the term added to the denominator
            of the update calculation to improve numerical stability. This use of epsilon1
            deviates from the algorithm written in the paper! See note below for more details.
            epsilon2 is the term used to avoid having too small a weight update when applying
            parameter scaling. (default: (None, 1e-3))
        d (float, optional): the clipping threshold, used to avoid larger-than-desired
            updates.
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        foreach (bool, optional): whether foreach implementation of optimizer is used. Note
            that the foreach implementation uses ~ sizeof(params) more peak memory than the
            for-loop version due to the intermediates being a tensorlist vs just one tensor.
            As Adafactor is commonly used when memory is prohibitive, Adafactor will default
            to the slower single tensor for-loop implementation unless this flag is explicitly
            True. This behavior is contrary to other optimizers, which will attempt defaulting
            to foreach on CUDA for faster runtime. (default: None)
        maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)
    .. Note::
        The implementation of Adafactor subtly differs from Noam Shazeer and Mitchell Stern
        and implementations in some other frameworks with its use of learning rate and
        :math:`\epsilon_1`.

        Regarding the learning rate hyperparameter: Noam Shazeer and Mitchell Stern do not
        use lr at all, as the stated algorithm uses :math:`\rho_t` and update clipping to
        affect the step size.

        This implementation allows `lr` to influence the maximum value for :math:`\rho_t`:

        .. math::
            \begin{aligned}
                &\hspace{5mm}\rho_t \leftarrow min(lr, \frac{1}{\sqrt{t}})
            \end{aligned}

        This differs from Noam Shazeer and Mitchell Stern, who use a constant of 0.01 as
        the maximum value of :math:`\rho_t`

        .. math::
            \begin{aligned}
                &\hspace{5mm}\rho_t \leftarrow min(0.01, \frac{1}{\sqrt{t}})
            \end{aligned}

        Noam Shazeer and Mitchell Stern do not enforce an opinion on how weight decay should
        be computed, and so we use the learning rate as a coefficient for decoupled weight
        decay, similar to what is suggested in `Decoupled Weight Decay Regularization`_.

        Regarding the use of :math:`\epsilon_1`: The implementation attempts to replicate the
        presumed intention of Noam Shazeer and Mitchell Stern to use :math:`\epsilon_1` as
        a stabilizing term when the squared gradient becomes small.

        This stabilization can be written as

        .. math::
            \begin{aligned}
                &\hspace{5mm}R_t \leftarrow \widehat{\beta}_{2_t}R_{t-1}+
                    (1-\widehat{\beta}_{2_t})(G_t \odot G_t + 1_n \cdot 1^\top_m) \cdot 1_m          \\
                &\hspace{5mm}C_t \leftarrow \widehat{\beta}_{2_t}C_{t-1}+
                    (1-\widehat{\beta}_{2_t}) 1^\top_n \cdot (G_t \odot G_t + 1_n \cdot 1^\top_m)    \\
                &\hspace{5mm}\widehat{V}_t \leftarrow
                    \frac{R_t \cdot C_t}{max(1^\top_n \cdot R_t, \epsilon_1)}                        \\
                &\hspace{5mm}U_t \leftarrow \frac{G_t}{max(\sqrt{\widehat{V}_t}, \epsilon_1)}        \\
            \end{aligned}

        where the row and column factors of gradient squared :math:`R_t` and :math:`C_t`
        are left alone, and we apply :math:`\epsilon_1` at the final calculation of
        the variance estimate :math:`\widehat{V}_t` and for the update :math:`U_t`.

        This is in contrast to Noam Shazeer and Mitchell Stern and other frameworks which
        apply :math:`\epsilon_1` to both row and column factors of the squared gradient, but
        not in the calculations after:

        .. math::
            \begin{aligned}
                &\hspace{5mm}R_t \leftarrow \widehat{\beta}_{2_t}R_{t-1}+
                            (1-\widehat{\beta}_{2_t})(G_t \odot G_t + \epsilon_1 1_n \cdot 1^\top_m) \cdot 1_m          \\
                &\hspace{5mm}C_t \leftarrow \widehat{\beta}_{2_t}C_{t-1}+
                            (1-\widehat{\beta}_{2_t}) 1^\top_n \cdot (G_t \odot G_t + \epsilon_1 1_n \cdot 1^\top_m)    \\
                &\hspace{5mm}\widehat{V}_t \leftarrow \frac{R_t \cdot C_t}{1^\top_n \cdot R_t}                          \\
                &\hspace{5mm}U_t \leftarrow \frac{G_t}{\sqrt{\widehat{V}_t}}                                            \\
            \end{aligned}

        You may note that Noam Shazeer and Mitchell Stern describe using the sum of squared gradients,
        while this implementation uses the mean instead. This choice is mathematically equivalent and
        allows for greater numerical stability for large sums.

    .. _Adafactor\: Adaptive Learning Rates with Sublinear Memory Cost:
        https://arxiv.org/pdf/1804.04235
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    """
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = ...,
        beta2_decay: float = ...,
        eps: tuple[float | None, float] = ...,
        d: float = ...,
        weight_decay: float = ...,
        *,
        foreach: bool | None = ...,
        maximize: bool = ...,
    ) -> None: ...
    def __setstate__(self, state): ...
    @torch.no_grad()
    def step(self, closure=...):
        """
        Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adafactor)
def adafactor(
    params: list[Tensor],
    grads: list[Tensor],
    row_vars: list[Tensor | None],
    col_vars: list[Tensor | None],
    variances: list[Tensor | None],
    state_steps: list[Tensor],
    foreach: bool | None = ...,
    grad_scale: Tensor | None = ...,
    found_inf: Tensor | None = ...,
    has_complex: bool = ...,
    *,
    d: float,
    lr: float | Tensor,
    beta2_decay: float,
    weight_decay: float,
    eps1: float,
    eps2: float,
    maximize: bool,
):
    """
    Functional API that performs Adafactor algorithm computation.

    See :class:`~torch.optim.Adafactor` for details.
    """
