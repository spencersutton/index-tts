from collections.abc import Callable
from functools import total_ordering

from torch import Tensor

from .distribution import Distribution

_KL_REGISTRY: dict[tuple[type, type], Callable] = ...
_KL_MEMOIZE: dict[tuple[type, type], Callable] = ...
__all__ = ["kl_divergence", "register_kl"]

def register_kl(type_p, type_q) -> Callable[..., Any]:
    """
    Decorator to register a pairwise function with :meth:`kl_divergence`.
    Usage::

        @register_kl(Normal, Normal)
        def kl_normal_normal(p, q):
            # insert implementation here

    Lookup returns the most specific (type,type) match ordered by subclass. If
    the match is ambiguous, a `RuntimeWarning` is raised. For example to
    resolve the ambiguous situation::

        @register_kl(BaseP, DerivedQ)
        def kl_version1(p, q): ...
        @register_kl(DerivedP, BaseQ)
        def kl_version2(p, q): ...

    you should register a third most-specific implementation, e.g.::

        register_kl(DerivedP, DerivedQ)(kl_version1)  # Break the tie.

    Args:
        type_p (type): A subclass of :class:`~torch.distributions.Distribution`.
        type_q (type): A subclass of :class:`~torch.distributions.Distribution`.
    """

@total_ordering
class _Match:
    __slots__ = ...
    def __init__(self, *types) -> None: ...
    def __eq__(self, other) -> bool: ...
    def __le__(self, other) -> bool: ...

def kl_divergence(p: Distribution, q: Distribution) -> Tensor:
    r"""
    Compute Kullback-Leibler divergence :math:`KL(p \| q)` between two distributions.

    .. math::

        KL(p \| q) = \int p(x) \log\frac {p(x)} {q(x)} \,dx

    Args:
        p (Distribution): A :class:`~torch.distributions.Distribution` object.
        q (Distribution): A :class:`~torch.distributions.Distribution` object.

    Returns:
        Tensor: A batch of KL divergences of shape `batch_shape`.

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_kl`.
    KL divergence is currently implemented for the following distribution pairs:
            * :class:`~torch.distributions.Bernoulli` and :class:`~torch.distributions.Bernoulli`
            * :class:`~torch.distributions.Bernoulli` and :class:`~torch.distributions.Poisson`
            * :class:`~torch.distributions.Beta` and :class:`~torch.distributions.Beta`
            * :class:`~torch.distributions.Beta` and :class:`~torch.distributions.ContinuousBernoulli`
            * :class:`~torch.distributions.Beta` and :class:`~torch.distributions.Exponential`
            * :class:`~torch.distributions.Beta` and :class:`~torch.distributions.Gamma`
            * :class:`~torch.distributions.Beta` and :class:`~torch.distributions.Normal`
            * :class:`~torch.distributions.Beta` and :class:`~torch.distributions.Pareto`
            * :class:`~torch.distributions.Beta` and :class:`~torch.distributions.Uniform`
            * :class:`~torch.distributions.Binomial` and :class:`~torch.distributions.Binomial`
            * :class:`~torch.distributions.Categorical` and :class:`~torch.distributions.Categorical`
            * :class:`~torch.distributions.Cauchy` and :class:`~torch.distributions.Cauchy`
            * :class:`~torch.distributions.ContinuousBernoulli` and :class:`~torch.distributions.ContinuousBernoulli`
            * :class:`~torch.distributions.ContinuousBernoulli` and :class:`~torch.distributions.Exponential`
            * :class:`~torch.distributions.ContinuousBernoulli` and :class:`~torch.distributions.Normal`
            * :class:`~torch.distributions.ContinuousBernoulli` and :class:`~torch.distributions.Pareto`
            * :class:`~torch.distributions.ContinuousBernoulli` and :class:`~torch.distributions.Uniform`
            * :class:`~torch.distributions.Dirichlet` and :class:`~torch.distributions.Dirichlet`
            * :class:`~torch.distributions.Exponential` and :class:`~torch.distributions.Beta`
            * :class:`~torch.distributions.Exponential` and :class:`~torch.distributions.ContinuousBernoulli`
            * :class:`~torch.distributions.Exponential` and :class:`~torch.distributions.Exponential`
            * :class:`~torch.distributions.Exponential` and :class:`~torch.distributions.Gamma`
            * :class:`~torch.distributions.Exponential` and :class:`~torch.distributions.Gumbel`
            * :class:`~torch.distributions.Exponential` and :class:`~torch.distributions.Normal`
            * :class:`~torch.distributions.Exponential` and :class:`~torch.distributions.Pareto`
            * :class:`~torch.distributions.Exponential` and :class:`~torch.distributions.Uniform`
            * :class:`~torch.distributions.ExponentialFamily` and :class:`~torch.distributions.ExponentialFamily`
            * :class:`~torch.distributions.Gamma` and :class:`~torch.distributions.Beta`
            * :class:`~torch.distributions.Gamma` and :class:`~torch.distributions.ContinuousBernoulli`
            * :class:`~torch.distributions.Gamma` and :class:`~torch.distributions.Exponential`
            * :class:`~torch.distributions.Gamma` and :class:`~torch.distributions.Gamma`
            * :class:`~torch.distributions.Gamma` and :class:`~torch.distributions.Gumbel`
            * :class:`~torch.distributions.Gamma` and :class:`~torch.distributions.Normal`
            * :class:`~torch.distributions.Gamma` and :class:`~torch.distributions.Pareto`
            * :class:`~torch.distributions.Gamma` and :class:`~torch.distributions.Uniform`
            * :class:`~torch.distributions.Geometric` and :class:`~torch.distributions.Geometric`
            * :class:`~torch.distributions.Gumbel` and :class:`~torch.distributions.Beta`
            * :class:`~torch.distributions.Gumbel` and :class:`~torch.distributions.ContinuousBernoulli`
            * :class:`~torch.distributions.Gumbel` and :class:`~torch.distributions.Exponential`
            * :class:`~torch.distributions.Gumbel` and :class:`~torch.distributions.Gamma`
            * :class:`~torch.distributions.Gumbel` and :class:`~torch.distributions.Gumbel`
            * :class:`~torch.distributions.Gumbel` and :class:`~torch.distributions.Normal`
            * :class:`~torch.distributions.Gumbel` and :class:`~torch.distributions.Pareto`
            * :class:`~torch.distributions.Gumbel` and :class:`~torch.distributions.Uniform`
            * :class:`~torch.distributions.HalfNormal` and :class:`~torch.distributions.HalfNormal`
            * :class:`~torch.distributions.Independent` and :class:`~torch.distributions.Independent`
            * :class:`~torch.distributions.Laplace` and :class:`~torch.distributions.Beta`
            * :class:`~torch.distributions.Laplace` and :class:`~torch.distributions.ContinuousBernoulli`
            * :class:`~torch.distributions.Laplace` and :class:`~torch.distributions.Exponential`
            * :class:`~torch.distributions.Laplace` and :class:`~torch.distributions.Gamma`
            * :class:`~torch.distributions.Laplace` and :class:`~torch.distributions.Laplace`
            * :class:`~torch.distributions.Laplace` and :class:`~torch.distributions.Normal`
            * :class:`~torch.distributions.Laplace` and :class:`~torch.distributions.Pareto`
            * :class:`~torch.distributions.Laplace` and :class:`~torch.distributions.Uniform`
            * :class:`~torch.distributions.LowRankMultivariateNormal` and :class:`~torch.distributions.LowRankMultivariateNormal`
            * :class:`~torch.distributions.LowRankMultivariateNormal` and :class:`~torch.distributions.MultivariateNormal`
            * :class:`~torch.distributions.MultivariateNormal` and :class:`~torch.distributions.LowRankMultivariateNormal`
            * :class:`~torch.distributions.MultivariateNormal` and :class:`~torch.distributions.MultivariateNormal`
            * :class:`~torch.distributions.Normal` and :class:`~torch.distributions.Beta`
            * :class:`~torch.distributions.Normal` and :class:`~torch.distributions.ContinuousBernoulli`
            * :class:`~torch.distributions.Normal` and :class:`~torch.distributions.Exponential`
            * :class:`~torch.distributions.Normal` and :class:`~torch.distributions.Gamma`
            * :class:`~torch.distributions.Normal` and :class:`~torch.distributions.Gumbel`
            * :class:`~torch.distributions.Normal` and :class:`~torch.distributions.Laplace`
            * :class:`~torch.distributions.Normal` and :class:`~torch.distributions.Normal`
            * :class:`~torch.distributions.Normal` and :class:`~torch.distributions.Pareto`
            * :class:`~torch.distributions.Normal` and :class:`~torch.distributions.Uniform`
            * :class:`~torch.distributions.OneHotCategorical` and :class:`~torch.distributions.OneHotCategorical`
            * :class:`~torch.distributions.Pareto` and :class:`~torch.distributions.Beta`
            * :class:`~torch.distributions.Pareto` and :class:`~torch.distributions.ContinuousBernoulli`
            * :class:`~torch.distributions.Pareto` and :class:`~torch.distributions.Exponential`
            * :class:`~torch.distributions.Pareto` and :class:`~torch.distributions.Gamma`
            * :class:`~torch.distributions.Pareto` and :class:`~torch.distributions.Normal`
            * :class:`~torch.distributions.Pareto` and :class:`~torch.distributions.Pareto`
            * :class:`~torch.distributions.Pareto` and :class:`~torch.distributions.Uniform`
            * :class:`~torch.distributions.Poisson` and :class:`~torch.distributions.Bernoulli`
            * :class:`~torch.distributions.Poisson` and :class:`~torch.distributions.Binomial`
            * :class:`~torch.distributions.Poisson` and :class:`~torch.distributions.Poisson`
            * :class:`~torch.distributions.TransformedDistribution` and :class:`~torch.distributions.TransformedDistribution`
            * :class:`~torch.distributions.Uniform` and :class:`~torch.distributions.Beta`
            * :class:`~torch.distributions.Uniform` and :class:`~torch.distributions.ContinuousBernoulli`
            * :class:`~torch.distributions.Uniform` and :class:`~torch.distributions.Exponential`
            * :class:`~torch.distributions.Uniform` and :class:`~torch.distributions.Gamma`
            * :class:`~torch.distributions.Uniform` and :class:`~torch.distributions.Gumbel`
            * :class:`~torch.distributions.Uniform` and :class:`~torch.distributions.Normal`
            * :class:`~torch.distributions.Uniform` and :class:`~torch.distributions.Pareto`
            * :class:`~torch.distributions.Uniform` and :class:`~torch.distributions.Uniform`
    """
