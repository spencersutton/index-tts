from __future__ import annotations

from abc import ABC
from typing import override

from torch import Tensor, nn

from indextts.gpt.conformer.attention import RelPositionMultiHeadedAttention
from indextts.gpt.conformer.encoder_layer import ConformerEncoderLayer
from indextts.gpt.conformer.modules import ConvolutionModule, PositionwiseFeedForward
from indextts.gpt.conformer.subsampling import Conv2dSubsampling2
from indextts.util import patch_call
from indextts.utils.common import make_pad_mask


class BaseEncoder(nn.Module, ABC):
    encoders: nn.ModuleList[ConformerEncoderLayer]

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
    ) -> None:
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            concat_after (bool): whether to concat attention layer's input
                and output.
                True: x -> x + linear(concat(x, att(x)))
                False: x -> x + att(x)
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
        global_cmvn (Optional[nn.Module]): Optional GlobalCMVN module
        use_dynamic_left_chunk (bool): whether use dynamic left chunk in
            dynamic chunk training.

        """
        super().__init__()
        self._output_size = output_size

        self.embed = Conv2dSubsampling2(input_size, output_size, dropout_rate)

        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)

    def output_size(self) -> int:
        return self._output_size

    @override
    def forward(
        self,
        xs: Tensor,
        xs_lens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        xs, pos_emb, masks = self.embed(xs, masks)
        chunk_masks = masks
        mask_pad = masks  # (B, 1, T/subsample_rate)
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    @patch_call(forward)
    def __call__(self) -> None: ...


class ConformerEncoder(BaseEncoder):
    """Conformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        macaron_style: bool = False,
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
    ) -> None:
        """Construct ConformerEncoder.

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
        """

        super().__init__(
            input_size,
            output_size,
            attention_heads,
            linear_units,
            num_blocks,
            dropout_rate,
            input_layer,
            pos_enc_layer_type,
            normalize_before,
            concat_after,
        )

        activation = nn.SiLU()

        self.encoders = nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                RelPositionMultiHeadedAttention(attention_heads, output_size, dropout_rate),
                PositionwiseFeedForward(output_size, linear_units, dropout_rate, activation),
                ConvolutionModule(output_size, cnn_module_kernel, activation),
                dropout_rate,
                normalize_before,
                concat_after,
            )
            for _ in range(num_blocks)
        ])
