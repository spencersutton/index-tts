
import torch
from torch import Tensor, nn
from torch.nn import Module

_LG = ...

class LayerNorm(nn.LayerNorm):
    def forward(self, input: Tensor) -> Tensor: ...

class ConvLayerBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        layer_norm: Module | None,
    ) -> None: ...
    def forward(self, x: Tensor, length: Tensor | None) -> tuple[Tensor, Tensor | None]: ...

class FeatureExtractor(Module):
    def __init__(self, conv_layers: nn.ModuleList) -> None: ...
    def forward(self, x: Tensor, length: Tensor | None) -> tuple[Tensor, Tensor | None]: ...

class FeatureProjection(Module):
    def __init__(self, in_features: int, out_features: int, dropout: float) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class ConvolutionalPositionalEmbedding(Module):
    def __init__(self, embed_dim: int, kernel_size: int, groups: int) -> None: ...
    def __prepare_scriptable__(self):  # -> Self:
        ...
    def forward(self, x):  # -> Tensor:
        ...

class SelfAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = ...) -> None: ...
    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = ...,
        position_bias: Tensor | None = ...,
        key_padding_mask: Tensor | None = ...,
    ) -> tuple[Tensor, Tensor | None]: ...

class FeedForward(Module):
    def __init__(
        self, io_features: int, intermediate_features: int, intermediate_dropout: float, output_dropout: float
    ) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class EncoderLayer(Module):
    def __init__(self, attention: Module, dropout: float, layer_norm_first: bool, feed_forward: Module) -> None: ...
    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = ...,
        position_bias: Tensor | None = ...,
        key_padding_mask: Tensor | None = ...,
    ) -> tuple[Tensor, Tensor | None]: ...

class Transformer(Module):
    def __init__(
        self, pos_conv_embed: Module, dropout: float, layers: Module, layer_norm_first: bool, layer_drop: float
    ) -> None: ...
    def forward(self, x: Tensor, attention_mask: Tensor | None = ..., position_bias: Tensor | None = ...) -> Tensor: ...
    def get_intermediate_outputs(
        self, x: Tensor, attention_mask: Tensor | None = ..., num_layers: int | None = ...
    ) -> list[Tensor]: ...

class Encoder(Module):
    def __init__(self, feature_projection: Module, transformer: Module) -> None: ...
    def forward(self, features: Tensor, lengths: Tensor | None = ...) -> Tensor: ...
    def extract_features(
        self, features: Tensor, lengths: Tensor | None = ..., num_layers: int | None = ...
    ) -> list[Tensor]: ...

class MaskGenerator(Module):
    def __init__(
        self,
        encoder_embed_dim: int,
        mask_prob: float,
        mask_selection: str,
        mask_other: float,
        mask_length: int,
        no_mask_overlap: bool,
        mask_min_space: int,
        mask_channel_prob: float,
        mask_channel_selection: str,
        mask_channel_other: float,
        mask_channel_length: int,
        no_mask_channel_overlap: bool,
        mask_channel_min_space: int,
    ) -> None: ...
    def forward(self, x: Tensor, padding_mask: Tensor | None) -> Tensor: ...

class LogitGenerator(Module):
    def __init__(
        self, encoder_embed_dim: int, num_classes: int, final_dim: int, skip_masked: bool, skip_nomask: bool
    ) -> None: ...
    def forward(self, x: Tensor, label: Tensor, mask_m: Tensor, mask_u: Tensor) -> tuple[Tensor, Tensor]: ...

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale): ...
    @staticmethod
    def backward(ctx, grad):  # -> tuple[Any, None]:
        ...
