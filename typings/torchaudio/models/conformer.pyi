
import torch

__all__ = ["Conformer"]

class _ConvolutionModule(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = ...,
        bias: bool = ...,
        use_group_norm: bool = ...,
    ) -> None: ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...

class _FeedForwardModule(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = ...) -> None: ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...

class ConformerLayer(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = ...,
        use_group_norm: bool = ...,
        convolution_first: bool = ...,
    ) -> None: ...
    def forward(self, input: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor: ...

class Conformer(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = ...,
        use_group_norm: bool = ...,
        convolution_first: bool = ...,
    ) -> None: ...
    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
