import typing

from audiotools import AudioSignal
from torch import nn

class L1Loss(nn.L1Loss):
    def __init__(self, attribute: str = ..., weight: float = ..., **kwargs) -> None: ...
    def forward(self, x: AudioSignal, y: AudioSignal):  # -> Tensor:
        ...

class SISDRLoss(nn.Module):
    def __init__(
        self,
        scaling: int = ...,
        reduction: str = ...,
        zero_mean: int = ...,
        clip_min: int = ...,
        weight: float = ...,
    ) -> None: ...
    def forward(self, x: AudioSignal, y: AudioSignal):  # -> Tensor:
        ...

class MultiScaleSTFTLoss(nn.Module):
    def __init__(
        self,
        window_lengths: list[int] = ...,
        loss_fn: typing.Callable = ...,
        clamp_eps: float = ...,
        mag_weight: float = ...,
        log_weight: float = ...,
        pow: float = ...,
        weight: float = ...,
        match_stride: bool = ...,
        window_type: str = ...,
    ) -> None: ...
    def forward(self, x: AudioSignal, y: AudioSignal):  # -> float:
        ...

class MelSpectrogramLoss(nn.Module):
    def __init__(
        self,
        n_mels: list[int] = ...,
        window_lengths: list[int] = ...,
        loss_fn: typing.Callable = ...,
        clamp_eps: float = ...,
        mag_weight: float = ...,
        log_weight: float = ...,
        pow: float = ...,
        weight: float = ...,
        match_stride: bool = ...,
        mel_fmin: list[float] = ...,
        mel_fmax: list[float] = ...,
        window_type: str = ...,
    ) -> None: ...
    def forward(self, x: AudioSignal, y: AudioSignal):  # -> float:
        ...

class GANLoss(nn.Module):
    def __init__(self, discriminator) -> None: ...
    def forward(self, fake, real):  # -> tuple[Any, Any]:
        ...
    def discriminator_loss(self, fake, real):  # -> Tensor | Literal[0]:
        ...
    def generator_loss(self, fake, real):  # -> tuple[Tensor | Literal[0], Tensor | Literal[0]]:
        ...
