from __future__ import annotations

from typing import override

import torch
from torch import Tensor, nn

from indextts.gpt.conformer.attention import RelPositionMultiHeadedAttention
from indextts.gpt.conformer.encoder_layer import ConformerEncoderLayer
from indextts.gpt.conformer.modules import ConvolutionModule, PositionwiseFeedForward
from indextts.gpt.conformer.subsampling import Conv2dSubsampling2
from indextts.util import patch_call


class ConformerEncoder(nn.Module):
    """Conformer encoder module."""

    encoders: nn.ModuleList[ConformerEncoderLayer]
    after_norm: nn.LayerNorm
    embed: Conv2dSubsampling2

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.0,
        cnn_module_kernel: int = 15,
    ) -> None:
        super().__init__()

        self.embed = Conv2dSubsampling2(input_size, output_size, dropout_rate)
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)
        activation = nn.SiLU()

        self.encoders = nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                RelPositionMultiHeadedAttention(attention_heads, output_size, dropout_rate),
                PositionwiseFeedForward(output_size, linear_units, dropout_rate, activation),
                ConvolutionModule(output_size, cnn_module_kernel, activation),
                dropout_rate,
            )
            for _ in range(num_blocks)
        ])

    @override
    def forward(self, xs: Tensor) -> tuple[Tensor, Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        xs, pos_emb, masks = self.embed(
            xs,
            (torch.arange(0, xs.size(1), device=xs.device).unsqueeze(0) < xs.size(2)).unsqueeze(1),
        )
        chunk_masks = masks
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, masks)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return self.after_norm(xs), masks

    @patch_call(forward)
    def __call__(self) -> None: ...
