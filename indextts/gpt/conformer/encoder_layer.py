"""Conformer encoder layer implementation.

This module contains the ConformerEncoderLayer class which combines
attention, convolution, and feed-forward modules into a single layer.
"""

from __future__ import annotations

from typing import override

import torch
from torch import Tensor, nn

from indextts.gpt.conformer.attention import RelPositionMultiHeadedAttention
from indextts.gpt.conformer.modules import ConvolutionModule, PositionwiseFeedForward
from indextts.util import patch_call


class ConformerEncoderLayer(nn.Module):
    """Encoder layer module for Conformer.

    Combines multi-headed self-attention, convolution, and feed-forward
    modules with residual connections and layer normalization.

    Args:
        size (int): Input dimension.
        self_attn (RelPositionMultiHeadedAttention): Self-attention module instance.
        feed_forward (PositionwiseFeedForward): Feed-forward module instance.
        conv_module (ConvolutionModule): Convolution module instance.
        dropout_rate (float): Dropout rate.
    """

    feed_forward: PositionwiseFeedForward | None

    def __init__(
        self,
        size: int,
        self_attn: RelPositionMultiHeadedAttention,
        feed_forward: PositionwiseFeedForward | None = None,
        conv_module: ConvolutionModule | None = None,
        dropout_rate: float = 0.1,
    ) -> None:
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        assert conv_module is not None
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)  # for the MHA module
        self.norm_conv = nn.LayerNorm(size, eps=1e-5)  # for the CNN module
        self.norm_final = nn.LayerNorm(size, eps=1e-5)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size

    @override
    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        pos_emb: Tensor,
        mask_pad: Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute encoded features.

        Args:
            x (Tensor): (#batch, time, size)
            mask (Tensor): Mask tensor for the input (#batch, time, time),
                (0, 0, 0) means fake mask.
            pos_emb (Tensor): Positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (Tensor): Batch padding mask used for conv module.
                (#batch, 1, time), (0, 0, 0) means fake mask.
            att_cache (Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)

        Returns:
            Tensor: Output tensor (#batch, time, size).
            Tensor: Mask tensor (#batch, time, time).
            Tensor: att_cache tensor, (#batch=1, head, cache_t1 + time, d_k * 2).
            Tensor: cnn_cache tensor (#batch, size, cache_t2).
        """
        # Multi-headed self-attention module
        residual = x
        x = self.norm_mha(x)

        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, att_cache)
        x = residual + self.dropout(x_att)

        # Convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype)
        residual = x
        x = self.norm_conv(x)
        x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
        x = residual + self.dropout(x)

        # Feed forward module
        residual = x
        x = self.norm_ff(x)

        assert self.feed_forward is not None
        x = residual + self.dropout(self.feed_forward(x))

        x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache

    @patch_call(forward)
    def __call__(self) -> None: ...
