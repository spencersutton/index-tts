# Implementation adapted from https://github.com/EdwardDixon/snake under the MIT license.
#   LICENSE is in incl_licenses directory.

# Further Adapted from https://github.com/NVIDIA/BigVGAN under the MIT license.

from typing import override

import torch
from torch import Tensor, nn, pow, sin
from torch.nn import Parameter


class Snake(nn.Module):
    """
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    in_features: int
    alpha_logscale: bool
    alpha: Parameter
    no_div_by_zero: float

    def __init__(
        self, in_features: int, alpha: float = 1.0, alpha_trainable: bool = True, alpha_logscale: bool = False
    ) -> None:
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        """
        super().__init__()
        self.in_features = in_features

        # Initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # Log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:  # Linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def _parameter(self) -> Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # Line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        return alpha

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # Line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        return x + (1.0 / (self._parameter() + self.no_div_by_zero)) * pow(sin(x * alpha), 2)


class SnakeBeta(Snake):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by
            Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    beta: Parameter

    def __init__(
        self, in_features: int, alpha: float = 1.0, alpha_trainable: bool = True, alpha_logscale: bool = False
    ) -> None:
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super().__init__(in_features, alpha, alpha_trainable, alpha_logscale)

        # Initialize beta
        self.beta = Parameter(self.alpha.detach().clone())
        self.beta.requires_grad = alpha_trainable

    @override
    def _parameter(self) -> Tensor:
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            beta = torch.exp(beta)
        return beta
