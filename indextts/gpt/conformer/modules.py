"""Feed-forward and convolution modules for Conformer encoder.

This module contains the basic building blocks used in the Conformer encoder.
"""

from __future__ import annotations

from typing import override

import torch
from torch import Tensor, nn

from indextts.util import patch_call


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward layer.

    FeedForward are applied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimension.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (nn.Module): Activation function
    """

    def __init__(self, idim: int, hidden_units: int, dropout_rate: float, activation: nn.SiLU) -> None:
        """Construct a PositionwiseFeedForward object."""
        super().__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    @override
    def forward(self, xs: Tensor) -> Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)

        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))

    @patch_call(forward)
    def __call__(self) -> None: ...


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.

    Implements a depth-wise separable convolution with gating mechanism
    used in the Conformer architecture.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation: nn.SiLU = nn.SiLU(),
        bias: bool = True,
    ) -> None:
        """Construct an ConvolutionModule object.

        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            activation: Activation function to use.
            bias: Whether to use bias in convolution layers.
        """
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # kernel_size should be an odd number for non-causal convolution
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2

        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        self.use_layer_norm = True
        self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    @override
    def forward(
        self,
        x: Tensor,
        mask_pad: Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: Tensor = torch.zeros((0, 0, 0)),
    ) -> tuple[Tensor, Tensor]:
        """Compute convolution module.

        Args:
            x (Tensor): Input tensor (#batch, time, channels).
            mask_pad (Tensor): Used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (Tensor): Left context cache, it is only used in causal
                convolution (#batch, channels, cache_t), (0, 0, 0) means fake cache.

        Returns:
            Tensor: Output tensor (#batch, time, channels).
            Tensor: New cache tensor.
        """
        # Exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (#batch, channels, time)

        # Mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        # Return empty cache tensor (no caching in this implementation)
        new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)

        # Mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2), new_cache

    @patch_call(forward)
    def __call__(self) -> None: ...
