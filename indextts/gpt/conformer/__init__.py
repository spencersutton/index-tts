"""Conformer encoder modules for speech processing.

This package contains the implementation of the Conformer encoder architecture,
which combines convolutional neural networks with self-attention mechanisms.
"""

from indextts.gpt.conformer.encoder_layer import ConformerEncoderLayer
from indextts.gpt.conformer.modules import ConvolutionModule, PositionwiseFeedForward

__all__ = [
    "ConformerEncoderLayer",
    "ConvolutionModule",
    "PositionwiseFeedForward",
]
