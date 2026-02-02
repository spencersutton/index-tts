from collections.abc import Sequence
from typing import cast, override

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from indextts.gpt.conformer.attention import RelPositionMultiHeadedAttention
from indextts.gpt.conformer.subsampling import Conv2dSubsampling2
from indextts.util import patch_call


def make_pad_mask(lengths: Int[Tensor, "b"], max_len: int = 0) -> Bool[Tensor, "b t"]:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (Tensor): Batch of lengths (B,).
    Returns:
        Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else int(lengths.max().item())
    seq_range = torch.arange(max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    return seq_range_expand >= seq_length_expand


class _PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        activation (nn.Module): Activation function
    """

    def __init__(self, idim: int, hidden_units: int, activation: nn.SiLU) -> None:
        """Construct a PositionwiseFeedForward object."""
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = nn.Dropout(0.0)
        self.w_2 = nn.Linear(hidden_units, idim)

    @override
    def forward(self, xs: Float[Tensor, "b t d"]) -> Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))

    @patch_call(forward)
    def __call__(self) -> None: ...


class _ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""

    def __init__(self, dim: int, activation: nn.SiLU) -> None:
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=15, padding=7, groups=dim)

        self.norm = nn.LayerNorm(dim)

        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.activation = activation

    @override
    def forward(
        self,
        x: Float[Tensor, "b t c"],
        mask_pad: Bool[Tensor, "b 1 t"] = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: Float[Tensor, "b c t"] = torch.zeros((0, 0, 0)),
    ) -> tuple[Tensor, Tensor]:
        """Compute convolution module.
        Args:
            x (Tensor): Input tensor (#batch, time, channels).
            mask_pad (Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.mT  # (#batch, channels, time)

        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        # It's better we just return None if no cache is required,
        # However, for JIT export, here we just fake one tensor instead of
        # None.
        new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = F.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x).mT
        x = self.activation(self.norm(x)).mT
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        return x.mT, new_cache

    @patch_call(forward)
    def __call__(self) -> None: ...


class _ConformerEncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
    """

    def __init__(
        self,
        size: int,
        self_attn: RelPositionMultiHeadedAttention,
        feed_forward: _PositionwiseFeedForward,
        conv_module: _ConvolutionModule,
    ) -> None:
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)  # for the MHA module
        self.ff_scale = 1.0
        self.norm_conv = nn.LayerNorm(size, eps=1e-5)  # for the CNN module
        self.norm_final = nn.LayerNorm(size, eps=1e-5)  # for the final output of the block
        self.dropout = nn.Dropout(0.0)
        self.size = size
        self.concat_linear = nn.Identity()

    @override
    def forward(
        self,
        x: Float[Tensor, "b t c"],
        mask: Bool[Tensor, "b t c"],
        pos_emb: Float[Tensor, "b t c"],
        mask_pad: Bool[Tensor, "b 1 t"] = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: Float[Tensor, "b h t d"] = torch.zeros((0, 0, 0, 0)),
        cnn_cache: Float[Tensor, "b c t"] = torch.zeros((0, 0, 0)),
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute encoded features.

        Args:
            x (Tensor): (#batch, time, size)
            mask (Tensor): Mask tensor for the input (#batch, time, time),
                (0, 0, 0) means fake mask.
            pos_emb (Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (Tensor): batch padding mask used for conv module.
                (#batch, 1, time), (0, 0, 0) means fake mask.
            att_cache (Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)
        Returns:
            Tensor: Output tensor (#batch, time, size).
            Tensor: Mask tensor (#batch, time, time).
            Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            Tensor: cnn_cache tensor (#batch, size, cache_t2).
        """

        # multi-headed self-attention module
        residual = x
        x = self.norm_mha.__call__(x)

        x_att, new_att_cache = self.self_attn.__call__(x, x, x, mask, pos_emb, att_cache)
        x = residual + self.dropout.__call__(x_att)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        residual = x
        x = self.norm_conv.__call__(x)
        x, new_cnn_cache = self.conv_module.__call__(x, mask_pad, cnn_cache)
        x = residual + self.dropout.__call__(x)

        # feed forward module
        residual = x
        x = self.norm_ff.__call__(x)

        x = residual + self.ff_scale * self.dropout.__call__(self.feed_forward.__call__(x))
        x = self.norm_final.__call__(x)

        return x, mask, new_att_cache, new_cnn_cache

    @patch_call(forward)
    def __call__(self) -> None: ...


class ConformerEncoder(nn.Module):
    """Conformer encoder module."""

    def __init__(self, dim: int, attention_heads: int = 4, linear_units: int = 2048, num_blocks: int = 6) -> None:
        """
        Args:
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
        """
        super().__init__()

        self.embed = Conv2dSubsampling2(input_dim=1024, output_dim=dim)
        self.after_norm = nn.LayerNorm(dim, eps=1e-5)
        activation = nn.SiLU()

        self.encoders = cast(  # pyright: ignore[reportInvalidCast]
            Sequence[_ConformerEncoderLayer],
            nn.ModuleList([
                _ConformerEncoderLayer(
                    dim,
                    RelPositionMultiHeadedAttention(attention_heads, dim),
                    _PositionwiseFeedForward(dim, linear_units, activation=activation),
                    _ConvolutionModule(dim, activation),
                )
                for _ in range(num_blocks)
            ]),
        )

    @override
    def forward(self, xs: Float[Tensor, "b t d"]) -> tuple[Tensor, Tensor]:
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
        xs_lens = torch.tensor([xs.shape[-1]], device=xs.device)
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        xs, pos_emb, masks = self.embed.__call__(xs, masks)
        chunk_masks = masks
        mask_pad = masks  # (B, 1, T/subsample_rate)
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer.__call__(xs, chunk_masks, pos_emb, mask_pad)
        xs = self.after_norm.__call__(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    @patch_call(forward)
    def __call__(self) -> None: ...
