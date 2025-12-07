import torch

"""Implements Conv-TasNet with building blocks of it.

Based on https://github.com/naplab/Conv-TasNet/tree/e66d82a8f956a69749ec8a4ae382217faa097c5c
"""

class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        io_channels: int,
        hidden_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int = ...,
        no_residual: bool = ...,
    ) -> None: ...
    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]: ...

class MaskGenerator(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_sources: int,
        kernel_size: int,
        num_feats: int,
        num_hidden: int,
        num_layers: int,
        num_stacks: int,
        msk_activate: str,
    ) -> None: ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...

class ConvTasNet(torch.nn.Module):
    def __init__(
        self,
        num_sources: int = ...,
        enc_kernel_size: int = ...,
        enc_num_feats: int = ...,
        msk_kernel_size: int = ...,
        msk_num_feats: int = ...,
        msk_num_hidden_feats: int = ...,
        msk_num_layers: int = ...,
        msk_num_stacks: int = ...,
        msk_activate: str = ...,
    ) -> None: ...
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...

def conv_tasnet_base(num_sources: int = ...) -> ConvTasNet: ...
