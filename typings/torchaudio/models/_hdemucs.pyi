from typing import Any

import torch
from torch import nn

class _ScaledEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float = ..., smooth: bool = ...) -> None: ...
    @property
    def weight(self) -> torch.Tensor: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class _HEncLayer(torch.nn.Module):
    def __init__(
        self,
        chin: int,
        chout: int,
        kernel_size: int = ...,
        stride: int = ...,
        norm_groups: int = ...,
        empty: bool = ...,
        freq: bool = ...,
        norm_type: str = ...,
        context: int = ...,
        dconv_kw: dict[str, Any] | None = ...,
        pad: bool = ...,
    ) -> None: ...
    def forward(self, x: torch.Tensor, inject: torch.Tensor | None = ...) -> torch.Tensor: ...

class _HDecLayer(torch.nn.Module):
    def __init__(
        self,
        chin: int,
        chout: int,
        last: bool = ...,
        kernel_size: int = ...,
        stride: int = ...,
        norm_groups: int = ...,
        empty: bool = ...,
        freq: bool = ...,
        norm_type: str = ...,
        context: int = ...,
        dconv_kw: dict[str, Any] | None = ...,
        pad: bool = ...,
    ) -> None: ...
    def forward(self, x: torch.Tensor, skip: torch.Tensor | None, length):  # -> tuple[Tensor | Any, Tensor]:
        ...

class HDemucs(torch.nn.Module):
    def __init__(
        self,
        sources: list[str],
        audio_channels: int = ...,
        channels: int = ...,
        growth: int = ...,
        nfft: int = ...,
        depth: int = ...,
        freq_emb: float = ...,
        emb_scale: int = ...,
        emb_smooth: bool = ...,
        kernel_size: int = ...,
        time_stride: int = ...,
        stride: int = ...,
        context: int = ...,
        context_enc: int = ...,
        norm_starts: int = ...,
        norm_groups: int = ...,
        dconv_depth: int = ...,
        dconv_comp: int = ...,
        dconv_attn: int = ...,
        dconv_lstm: int = ...,
        dconv_init: float = ...,
    ) -> None: ...
    def forward(self, input: torch.Tensor):  # -> Tensor | Any:
        ...

class _DConv(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        compress: float = ...,
        depth: int = ...,
        init: float = ...,
        norm_type: str = ...,
        attn: bool = ...,
        heads: int = ...,
        ndecay: int = ...,
        lstm: bool = ...,
        kernel_size: int = ...,
    ) -> None: ...
    def forward(self, x): ...

class _BLSTM(torch.nn.Module):
    def __init__(self, dim, layers: int = ..., skip: bool = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class _LocalState(nn.Module):
    def __init__(self, channels: int, heads: int = ..., ndecay: int = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class _LayerScale(nn.Module):
    def __init__(self, channels: int, init: float = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

def hdemucs_low(sources: list[str]) -> HDemucs: ...
def hdemucs_medium(sources: list[str]) -> HDemucs: ...
def hdemucs_high(sources: list[str]) -> HDemucs: ...
