from collections.abc import Sequence
from typing import cast, override

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from indextts.gpt.conformer.attention import RelPositionMultiHeadedAttention
from indextts.gpt.conformer.subsampling import Conv2dSubsampling2
from indextts.util import patch_call


def make_pad_mask(lengths: Int[Tensor, "b"], max_len: int = 0) -> Bool[Tensor, "b t"]:  # noqa: UP037
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

    w_1: nn.Linear
    activation: nn.SiLU
    w_2: nn.Linear

    def __init__(self, idim: int, hidden_units: int, activation: nn.SiLU) -> None:
        """Construct a PositionwiseFeedForward object."""
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = activation
        self.w_2 = nn.Linear(hidden_units, idim)

    @override
    def forward(self, xs: Float[Tensor, "b t d"]) -> Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.activation(self.w_1(xs)))

    @patch_call(forward)
    def __call__(self) -> None: ...


class _ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""

    pointwise_conv1: nn.Conv1d
    depthwise_conv: nn.Conv1d
    norm: nn.LayerNorm
    pointwise_conv2: nn.Conv1d
    activation: nn.SiLU

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
    def forward(self, x: Float[Tensor, "b t c"], mask_pad: Bool[Tensor, "b 1 t"]) -> Tensor:
        """Compute convolution module.
        Args:
            x (Tensor): Input tensor (#batch, time, channels).
            mask_pad (Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
        Returns:
            Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.mT  # (#batch, channels, time)

        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = F.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x).mT
        x = self.norm(x)
        x = self.activation(x).mT
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        return x.mT

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

    self_attn: RelPositionMultiHeadedAttention
    feed_forward: _PositionwiseFeedForward
    conv_module: _ConvolutionModule
    norm_ff: nn.LayerNorm
    norm_mha: nn.LayerNorm
    norm_conv: nn.LayerNorm
    norm_final: nn.LayerNorm
    size: int
    concat_linear: nn.Identity

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
        self.norm_ff = nn.LayerNorm(size)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size)  # for the MHA module
        self.norm_conv = nn.LayerNorm(size)  # for the CNN module
        self.norm_final = nn.LayerNorm(size)  # for the final output of the block
        self.size = size
        self.concat_linear = nn.Identity()

    @override
    def forward(
        self,
        x: Float[Tensor, "b t c"],
        mask: Bool[Tensor, "b t c"],
        pos_emb: Float[Tensor, "b t c"],
        mask_pad: Bool[Tensor, "b 1 t"],
    ) -> tuple[Tensor, Tensor]:
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
        Returns:
            Tensor: Output tensor (#batch, time, size).
            Tensor: Mask tensor (#batch, time, time).
        """

        # multi-headed self-attention module
        norm = self.norm_mha(x)
        x += self.self_attn.__call__(norm, norm, norm, mask, pos_emb)
        x += self.conv_module.__call__(self.norm_conv(x), mask_pad)
        x += self.feed_forward.__call__(self.norm_ff(x))
        x = self.norm_final(x)
        return x, mask

    @patch_call(forward)
    def __call__(self) -> None: ...


class ConformerEncoder(nn.Module):
    """Conformer encoder module."""

    embed: Conv2dSubsampling2
    after_norm: nn.LayerNorm
    encoders: Sequence[_ConformerEncoderLayer]

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
        self.after_norm = nn.LayerNorm(dim)
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
        Returns:
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
            xs, chunk_masks = layer.__call__(xs, chunk_masks, pos_emb, mask_pad)
        xs = self.after_norm.__call__(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    @patch_call(forward)
    def __call__(self) -> None: ...
