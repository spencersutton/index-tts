import torch
from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from typing import Callable, Sequence

class Spectrogram(torch.nn.Module):
    __constants__: Incomplete
    n_fft: Incomplete
    win_length: Incomplete
    hop_length: Incomplete
    pad: Incomplete
    power: Incomplete
    normalized: Incomplete
    center: Incomplete
    pad_mode: Incomplete
    onesided: Incomplete
    def __init__(
        self,
        n_fft: int = 400,
        win_length: int | None = None,
        hop_length: int | None = None,
        pad: int = 0,
        window_fn: Callable[..., Tensor] = ...,
        power: float | None = 2.0,
        normalized: bool | str = False,
        wkwargs: dict | None = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        return_complex: bool | None = None,
    ) -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...
    def __call__(self, waveform: Tensor) -> Tensor: ...

class InverseSpectrogram(torch.nn.Module):
    __constants__: Incomplete
    n_fft: Incomplete
    win_length: Incomplete
    hop_length: Incomplete
    pad: Incomplete
    normalized: Incomplete
    center: Incomplete
    pad_mode: Incomplete
    onesided: Incomplete
    def __init__(
        self,
        n_fft: int = 400,
        win_length: int | None = None,
        hop_length: int | None = None,
        pad: int = 0,
        window_fn: Callable[..., Tensor] = ...,
        normalized: bool | str = False,
        wkwargs: dict | None = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
    ) -> None: ...
    def forward(self, spectrogram: Tensor, length: int | None = None) -> Tensor: ...

class GriffinLim(torch.nn.Module):
    __constants__: Incomplete
    n_fft: Incomplete
    n_iter: Incomplete
    win_length: Incomplete
    hop_length: Incomplete
    length: Incomplete
    power: Incomplete
    momentum: Incomplete
    rand_init: Incomplete
    def __init__(
        self,
        n_fft: int = 400,
        n_iter: int = 32,
        win_length: int | None = None,
        hop_length: int | None = None,
        window_fn: Callable[..., Tensor] = ...,
        power: float = 2.0,
        wkwargs: dict | None = None,
        momentum: float = 0.99,
        length: int | None = None,
        rand_init: bool = True,
    ) -> None: ...
    def forward(self, specgram: Tensor) -> Tensor: ...

class AmplitudeToDB(torch.nn.Module):
    __constants__: Incomplete
    stype: Incomplete
    top_db: Incomplete
    multiplier: Incomplete
    amin: float
    ref_value: float
    db_multiplier: Incomplete
    def __init__(self, stype: str = "power", top_db: float | None = None) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class MelScale(torch.nn.Module):
    __constants__: Incomplete
    n_mels: Incomplete
    sample_rate: Incomplete
    f_max: Incomplete
    f_min: Incomplete
    norm: Incomplete
    mel_scale: Incomplete
    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: float | None = None,
        n_stft: int = 201,
        norm: str | None = None,
        mel_scale: str = "htk",
    ) -> None: ...
    def forward(self, specgram: Tensor) -> Tensor: ...

class InverseMelScale(torch.nn.Module):
    __constants__: Incomplete
    n_mels: Incomplete
    sample_rate: Incomplete
    f_max: Incomplete
    f_min: Incomplete
    driver: Incomplete
    def __init__(
        self,
        n_stft: int,
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: float | None = None,
        norm: str | None = None,
        mel_scale: str = "htk",
        driver: str = "gels",
    ) -> None: ...
    def forward(self, melspec: Tensor) -> Tensor: ...

class MelSpectrogram(torch.nn.Module):
    __constants__: Incomplete
    sample_rate: Incomplete
    n_fft: Incomplete
    win_length: Incomplete
    hop_length: Incomplete
    pad: Incomplete
    power: Incomplete
    normalized: Incomplete
    n_mels: Incomplete
    f_max: Incomplete
    f_min: Incomplete
    spectrogram: Incomplete
    mel_scale: Incomplete
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int | None = None,
        hop_length: int | None = None,
        f_min: float = 0.0,
        f_max: float | None = None,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: Callable[..., Tensor] = ...,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: dict | None = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool | None = None,
        norm: str | None = None,
        mel_scale: str = "htk",
    ) -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...

class MFCC(torch.nn.Module):
    __constants__: Incomplete
    sample_rate: Incomplete
    n_mfcc: Incomplete
    dct_type: Incomplete
    norm: Incomplete
    top_db: float
    amplitude_to_DB: Incomplete
    MelSpectrogram: Incomplete
    log_mels: Incomplete
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        dct_type: int = 2,
        norm: str = "ortho",
        log_mels: bool = False,
        melkwargs: dict | None = None,
    ) -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...

class LFCC(torch.nn.Module):
    __constants__: Incomplete
    sample_rate: Incomplete
    f_min: Incomplete
    f_max: Incomplete
    n_filter: Incomplete
    n_lfcc: Incomplete
    dct_type: Incomplete
    norm: Incomplete
    top_db: float
    amplitude_to_DB: Incomplete
    Spectrogram: Incomplete
    log_lf: Incomplete
    def __init__(
        self,
        sample_rate: int = 16000,
        n_filter: int = 128,
        f_min: float = 0.0,
        f_max: float | None = None,
        n_lfcc: int = 40,
        dct_type: int = 2,
        norm: str = "ortho",
        log_lf: bool = False,
        speckwargs: dict | None = None,
    ) -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...

class MuLawEncoding(torch.nn.Module):
    __constants__: Incomplete
    quantization_channels: Incomplete
    def __init__(self, quantization_channels: int = 256) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class MuLawDecoding(torch.nn.Module):
    __constants__: Incomplete
    quantization_channels: Incomplete
    def __init__(self, quantization_channels: int = 256) -> None: ...
    def forward(self, x_mu: Tensor) -> Tensor: ...

class Resample(torch.nn.Module):
    orig_freq: int
    new_freq: int
    gcd: int
    resampling_method: int
    lowpass_filter_width: int
    rolloff: int
    beta: int
    def __init__(
        self,
        orig_freq: int = 16000,
        new_freq: int = 16000,
        resampling_method: str = "sinc_interp_hann",
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        beta: float | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...
    def __call__(self, waveform: Tensor) -> Tensor: ...

class ComputeDeltas(torch.nn.Module):
    __constants__: Incomplete
    win_length: Incomplete
    mode: Incomplete
    def __init__(self, win_length: int = 5, mode: str = "replicate") -> None: ...
    def forward(self, specgram: Tensor) -> Tensor: ...

class TimeStretch(torch.nn.Module):
    __constants__: Incomplete
    fixed_rate: Incomplete
    def __init__(self, hop_length: int | None = None, n_freq: int = 201, fixed_rate: float | None = None) -> None: ...
    def forward(self, complex_specgrams: Tensor, overriding_rate: float | None = None) -> Tensor: ...

class Fade(torch.nn.Module):
    fade_in_len: Incomplete
    fade_out_len: Incomplete
    fade_shape: Incomplete
    def __init__(self, fade_in_len: int = 0, fade_out_len: int = 0, fade_shape: str = "linear") -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...

class _AxisMasking(torch.nn.Module):
    __constants__: Incomplete
    mask_param: Incomplete
    axis: Incomplete
    iid_masks: Incomplete
    p: Incomplete
    def __init__(self, mask_param: int, axis: int, iid_masks: bool, p: float = 1.0) -> None: ...
    def forward(self, specgram: Tensor, mask_value: float = 0.0) -> Tensor: ...

class FrequencyMasking(_AxisMasking):
    def __init__(self, freq_mask_param: int, iid_masks: bool = False) -> None: ...

class TimeMasking(_AxisMasking):
    def __init__(self, time_mask_param: int, iid_masks: bool = False, p: float = 1.0) -> None: ...

class SpecAugment(torch.nn.Module):
    __constants__: Incomplete
    n_time_masks: Incomplete
    time_mask_param: Incomplete
    n_freq_masks: Incomplete
    freq_mask_param: Incomplete
    iid_masks: Incomplete
    p: Incomplete
    zero_masking: Incomplete
    def __init__(
        self,
        n_time_masks: int,
        time_mask_param: int,
        n_freq_masks: int,
        freq_mask_param: int,
        iid_masks: bool = True,
        p: float = 1.0,
        zero_masking: bool = False,
    ) -> None: ...
    def forward(self, specgram: Tensor) -> Tensor: ...

class Loudness(torch.nn.Module):
    __constants__: Incomplete
    sample_rate: Incomplete
    def __init__(self, sample_rate: int) -> None: ...
    def forward(self, wavefrom: Tensor): ...

class Vol(torch.nn.Module):
    gain: Incomplete
    gain_type: Incomplete
    def __init__(self, gain: float, gain_type: str = "amplitude") -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...

class SlidingWindowCmn(torch.nn.Module):
    cmn_window: Incomplete
    min_cmn_window: Incomplete
    center: Incomplete
    norm_vars: Incomplete
    def __init__(
        self, cmn_window: int = 600, min_cmn_window: int = 100, center: bool = False, norm_vars: bool = False
    ) -> None: ...
    def forward(self, specgram: Tensor) -> Tensor: ...

class Vad(torch.nn.Module):
    sample_rate: Incomplete
    trigger_level: Incomplete
    trigger_time: Incomplete
    search_time: Incomplete
    allowed_gap: Incomplete
    pre_trigger_time: Incomplete
    boot_time: Incomplete
    noise_up_time: Incomplete
    noise_down_time: Incomplete
    noise_reduction_amount: Incomplete
    measure_freq: Incomplete
    measure_duration: Incomplete
    measure_smooth_time: Incomplete
    hp_filter_freq: Incomplete
    lp_filter_freq: Incomplete
    hp_lifter_freq: Incomplete
    lp_lifter_freq: Incomplete
    def __init__(
        self,
        sample_rate: int,
        trigger_level: float = 7.0,
        trigger_time: float = 0.25,
        search_time: float = 1.0,
        allowed_gap: float = 0.25,
        pre_trigger_time: float = 0.0,
        boot_time: float = 0.35,
        noise_up_time: float = 0.1,
        noise_down_time: float = 0.01,
        noise_reduction_amount: float = 1.35,
        measure_freq: float = 20.0,
        measure_duration: float | None = None,
        measure_smooth_time: float = 0.4,
        hp_filter_freq: float = 50.0,
        lp_filter_freq: float = 6000.0,
        hp_lifter_freq: float = 150.0,
        lp_lifter_freq: float = 2000.0,
    ) -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...

class SpectralCentroid(torch.nn.Module):
    __constants__: Incomplete
    sample_rate: Incomplete
    n_fft: Incomplete
    win_length: Incomplete
    hop_length: Incomplete
    pad: Incomplete
    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 400,
        win_length: int | None = None,
        hop_length: int | None = None,
        pad: int = 0,
        window_fn: Callable[..., Tensor] = ...,
        wkwargs: dict | None = None,
    ) -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...

class PitchShift(LazyModuleMixin, torch.nn.Module):
    __constants__: Incomplete
    kernel: UninitializedParameter
    width: int
    n_steps: Incomplete
    bins_per_octave: Incomplete
    sample_rate: Incomplete
    n_fft: Incomplete
    win_length: Incomplete
    hop_length: Incomplete
    orig_freq: Incomplete
    gcd: Incomplete
    def __init__(
        self,
        sample_rate: int,
        n_steps: int,
        bins_per_octave: int = 12,
        n_fft: int = 512,
        win_length: int | None = None,
        hop_length: int | None = None,
        window_fn: Callable[..., Tensor] = ...,
        wkwargs: dict | None = None,
    ) -> None: ...
    def initialize_parameters(self, input) -> None: ...
    def forward(self, waveform: Tensor) -> Tensor: ...

class RNNTLoss(torch.nn.Module):
    blank: Incomplete
    clamp: Incomplete
    reduction: Incomplete
    fused_log_softmax: Incomplete
    def __init__(
        self, blank: int = -1, clamp: float = -1.0, reduction: str = "mean", fused_log_softmax: bool = True
    ) -> None: ...
    def forward(self, logits: Tensor, targets: Tensor, logit_lengths: Tensor, target_lengths: Tensor): ...

class Convolve(torch.nn.Module):
    mode: Incomplete
    def __init__(self, mode: str = "full") -> None: ...
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

class FFTConvolve(torch.nn.Module):
    mode: Incomplete
    def __init__(self, mode: str = "full") -> None: ...
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

class Speed(torch.nn.Module):
    orig_freq: Incomplete
    factor: Incomplete
    resampler: Incomplete
    def __init__(self, orig_freq, factor) -> None: ...
    def forward(self, waveform, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class SpeedPerturbation(torch.nn.Module):
    speeders: Incomplete
    def __init__(self, orig_freq: int, factors: Sequence[float]) -> None: ...
    def forward(
        self, waveform: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class AddNoise(torch.nn.Module):
    def forward(
        self, waveform: torch.Tensor, noise: torch.Tensor, snr: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class Preemphasis(torch.nn.Module):
    coeff: Incomplete
    def __init__(self, coeff: float = 0.97) -> None: ...
    def forward(self, waveform: torch.Tensor) -> torch.Tensor: ...

class Deemphasis(torch.nn.Module):
    coeff: Incomplete
    def __init__(self, coeff: float = 0.97) -> None: ...
    def forward(self, waveform: torch.Tensor) -> torch.Tensor: ...
