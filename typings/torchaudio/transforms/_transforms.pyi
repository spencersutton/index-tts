from collections.abc import Callable, Sequence

import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter

from indextts.util import patch_call

__all__ = []

class Spectrogram(torch.nn.Module):
    __constants__ = ...
    def __init__(
        self,
        n_fft: int = ...,
        win_length: int | None = ...,
        hop_length: int | None = ...,
        pad: int = ...,
        window_fn: Callable[..., Tensor] = ...,
        power: float | None = ...,
        normalized: bool | str = ...,
        wkwargs: dict | None = ...,
        center: bool = ...,
        pad_mode: str = ...,
        onesided: bool = ...,
        return_complex: bool | None = ...,
    ) -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...

class InverseSpectrogram(torch.nn.Module):
    __constants__ = ...
    def __init__(
        self,
        n_fft: int = ...,
        win_length: int | None = ...,
        hop_length: int | None = ...,
        pad: int = ...,
        window_fn: Callable[..., Tensor] = ...,
        normalized: bool | str = ...,
        wkwargs: dict | None = ...,
        center: bool = ...,
        pad_mode: str = ...,
        onesided: bool = ...,
    ) -> None: ...
    def forward(self, spectrogram: Tensor, length: int | None = ...) -> Tensor: ...

class GriffinLim(torch.nn.Module):
    __constants__ = ...
    def __init__(
        self,
        n_fft: int = ...,
        n_iter: int = ...,
        win_length: int | None = ...,
        hop_length: int | None = ...,
        window_fn: Callable[..., Tensor] = ...,
        power: float = ...,
        wkwargs: dict | None = ...,
        momentum: float = ...,
        length: int | None = ...,
        rand_init: bool = ...,
    ) -> None: ...
    def forward(self, specgram: Tensor) -> Tensor: ...

class AmplitudeToDB(torch.nn.Module):
    __constants__ = ...
    def __init__(self, stype: str = ..., top_db: float | None = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class MelScale(torch.nn.Module):
    __constants__ = ...
    def __init__(
        self,
        n_mels: int = ...,
        sample_rate: int = ...,
        f_min: float = ...,
        f_max: float | None = ...,
        n_stft: int = ...,
        norm: str | None = ...,
        mel_scale: str = ...,
    ) -> None: ...
    def forward(self, specgram: Tensor) -> Tensor: ...

class InverseMelScale(torch.nn.Module):
    __constants__ = ...
    def __init__(
        self,
        n_stft: int,
        n_mels: int = ...,
        sample_rate: int = ...,
        f_min: float = ...,
        f_max: float | None = ...,
        norm: str | None = ...,
        mel_scale: str = ...,
        driver: str = ...,
    ) -> None: ...
    def forward(self, melspec: Tensor) -> Tensor: ...

class MelSpectrogram(torch.nn.Module):
    __constants__ = ...
    def __init__(
        self,
        sample_rate: int = ...,
        n_fft: int = ...,
        win_length: int | None = ...,
        hop_length: int | None = ...,
        f_min: float = ...,
        f_max: float | None = ...,
        pad: int = ...,
        n_mels: int = ...,
        window_fn: Callable[..., Tensor] = ...,
        power: float = ...,
        normalized: bool = ...,
        wkwargs: dict | None = ...,
        center: bool = ...,
        pad_mode: str = ...,
        onesided: bool | None = ...,
        norm: str | None = ...,
        mel_scale: str = ...,
    ) -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...

class MFCC(torch.nn.Module):
    __constants__ = ...
    def __init__(
        self,
        sample_rate: int = ...,
        n_mfcc: int = ...,
        dct_type: int = ...,
        norm: str = ...,
        log_mels: bool = ...,
        melkwargs: dict | None = ...,
    ) -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...

class LFCC(torch.nn.Module):
    __constants__ = ...
    def __init__(
        self,
        sample_rate: int = ...,
        n_filter: int = ...,
        f_min: float = ...,
        f_max: float | None = ...,
        n_lfcc: int = ...,
        dct_type: int = ...,
        norm: str = ...,
        log_lf: bool = ...,
        speckwargs: dict | None = ...,
    ) -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...

class MuLawEncoding(torch.nn.Module):
    __constants__ = ...
    def __init__(self, quantization_channels: int = ...) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class MuLawDecoding(torch.nn.Module):
    __constants__ = ...
    def __init__(self, quantization_channels: int = ...) -> None: ...
    def forward(self, x_mu: Tensor) -> Tensor: ...

class Resample(torch.nn.Module):
    def __init__(
        self,
        orig_freq: int = ...,
        new_freq: int = ...,
        resampling_method: str = ...,
        lowpass_filter_width: int = ...,
        rolloff: float = ...,
        beta: float | None = ...,
        *,
        dtype: torch.dtype | None = ...,
    ) -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...
    @patch_call(forward)
    def __call__(self) -> None: ...

class ComputeDeltas(torch.nn.Module):
    __constants__ = ...
    def __init__(self, win_length: int = ..., mode: str = ...) -> None: ...
    def forward(self, specgram: Tensor) -> Tensor: ...

class TimeStretch(torch.nn.Module):
    __constants__ = ...
    def __init__(
        self,
        hop_length: int | None = ...,
        n_freq: int = ...,
        fixed_rate: float | None = ...,
    ) -> None: ...
    def forward(self, complex_specgrams: Tensor, overriding_rate: float | None = ...) -> Tensor: ...

class Fade(torch.nn.Module):
    def __init__(
        self,
        fade_in_len: int = ...,
        fade_out_len: int = ...,
        fade_shape: str = ...,
    ) -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...

class _AxisMasking(torch.nn.Module):
    __constants__ = ...
    def __init__(self, mask_param: int, axis: int, iid_masks: bool, p: float = ...) -> None: ...
    def forward(self, specgram: Tensor, mask_value: float | torch.Tensor = ...) -> Tensor: ...

class FrequencyMasking(_AxisMasking):
    def __init__(self, freq_mask_param: int, iid_masks: bool = ...) -> None: ...

class TimeMasking(_AxisMasking):
    def __init__(self, time_mask_param: int, iid_masks: bool = ..., p: float = ...) -> None: ...

class SpecAugment(torch.nn.Module):
    __constants__ = ...
    def __init__(
        self,
        n_time_masks: int,
        time_mask_param: int,
        n_freq_masks: int,
        freq_mask_param: int,
        iid_masks: bool = ...,
        p: float = ...,
        zero_masking: bool = ...,
    ) -> None: ...
    def forward(self, specgram: Tensor) -> Tensor: ...

class Loudness(torch.nn.Module):
    __constants__ = ...
    def __init__(self, sample_rate: int) -> None: ...
    def forward(self, wavefrom: Tensor): ...

class Vol(torch.nn.Module):
    def __init__(self, gain: float, gain_type: str = ...) -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...

class SlidingWindowCmn(torch.nn.Module):
    def __init__(
        self,
        cmn_window: int = ...,
        min_cmn_window: int = ...,
        center: bool = ...,
        norm_vars: bool = ...,
    ) -> None: ...
    def forward(self, specgram: Tensor) -> Tensor: ...

class Vad(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int,
        trigger_level: float = ...,
        trigger_time: float = ...,
        search_time: float = ...,
        allowed_gap: float = ...,
        pre_trigger_time: float = ...,
        boot_time: float = ...,
        noise_up_time: float = ...,
        noise_down_time: float = ...,
        noise_reduction_amount: float = ...,
        measure_freq: float = ...,
        measure_duration: float | None = ...,
        measure_smooth_time: float = ...,
        hp_filter_freq: float = ...,
        lp_filter_freq: float = ...,
        hp_lifter_freq: float = ...,
        lp_lifter_freq: float = ...,
    ) -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...

class SpectralCentroid(torch.nn.Module):
    __constants__ = ...
    def __init__(
        self,
        sample_rate: int,
        n_fft: int = ...,
        win_length: int | None = ...,
        hop_length: int | None = ...,
        pad: int = ...,
        window_fn: Callable[..., Tensor] = ...,
        wkwargs: dict | None = ...,
    ) -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...

class PitchShift(LazyModuleMixin, torch.nn.Module):
    __constants__ = ...
    kernel: UninitializedParameter
    width: int
    def __init__(
        self,
        sample_rate: int,
        n_steps: int,
        bins_per_octave: int = ...,
        n_fft: int = ...,
        win_length: int | None = ...,
        hop_length: int | None = ...,
        window_fn: Callable[..., Tensor] = ...,
        wkwargs: dict | None = ...,
    ) -> None: ...
    def initialize_parameters(self, input): ...
    def forward(self, waveform: Tensor) -> Tensor: ...

class RNNTLoss(torch.nn.Module):
    def __init__(
        self,
        blank: int = ...,
        clamp: float = ...,
        reduction: str = ...,
        fused_log_softmax: bool = ...,
    ) -> None: ...
    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        logit_lengths: Tensor,
        target_lengths: Tensor,
    ): ...

class Convolve(torch.nn.Module):
    def __init__(self, mode: str = ...) -> None: ...
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

class FFTConvolve(torch.nn.Module):
    def __init__(self, mode: str = ...) -> None: ...
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

class Speed(torch.nn.Module):
    def __init__(self, orig_freq, factor) -> None: ...
    def forward(self, waveform, lengths: torch.Tensor | None = ...) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class SpeedPerturbation(torch.nn.Module):
    def __init__(self, orig_freq: int, factors: Sequence[float]) -> None: ...
    def forward(
        self, waveform: torch.Tensor, lengths: torch.Tensor | None = ...
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class AddNoise(torch.nn.Module):
    def forward(
        self,
        waveform: torch.Tensor,
        noise: torch.Tensor,
        snr: torch.Tensor,
        lengths: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...

class Preemphasis(torch.nn.Module):
    def __init__(self, coeff: float = ...) -> None: ...
    def forward(self, waveform: torch.Tensor) -> torch.Tensor: ...

class Deemphasis(torch.nn.Module):
    def __init__(self, coeff: float = ...) -> None: ...
    def forward(self, waveform: torch.Tensor) -> torch.Tensor: ...
