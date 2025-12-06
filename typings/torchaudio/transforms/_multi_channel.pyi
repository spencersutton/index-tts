
import torch
from torch import Tensor

__all__ = []

class PSD(torch.nn.Module):
    def __init__(self, multi_mask: bool = ..., normalize: bool = ..., eps: float = ...) -> None: ...
    def forward(self, specgram: torch.Tensor, mask: torch.Tensor | None = ...):  # -> Tensor:
        ...

class MVDR(torch.nn.Module):
    def __init__(
        self,
        ref_channel: int = ...,
        solution: str = ...,
        multi_mask: bool = ...,
        diag_loading: bool = ...,
        diag_eps: float = ...,
        online: bool = ...,
    ) -> None: ...
    def forward(
        self, specgram: torch.Tensor, mask_s: torch.Tensor, mask_n: torch.Tensor | None = ...
    ) -> torch.Tensor: ...

class RTFMVDR(torch.nn.Module):
    def forward(
        self,
        specgram: Tensor,
        rtf: Tensor,
        psd_n: Tensor,
        reference_channel: int | Tensor,
        diagonal_loading: bool = ...,
        diag_eps: float = ...,
        eps: float = ...,
    ) -> Tensor: ...

class SoudenMVDR(torch.nn.Module):
    def forward(
        self,
        specgram: Tensor,
        psd_s: Tensor,
        psd_n: Tensor,
        reference_channel: int | Tensor,
        diagonal_loading: bool = ...,
        diag_eps: float = ...,
        eps: float = ...,
    ) -> torch.Tensor: ...
