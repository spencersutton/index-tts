from collections.abc import Sequence

import torch
from torch import Tensor

__all__ = [
    "DB_to_amplitude",
    "DB_to_amplitude",
    "add_noise",
    "amplitude_to_DB",
    "apply_beamforming",
    "compute_deltas",
    "compute_deltas",
    "convolve",
    "create_dct",
    "deemphasis",
    "detect_pitch_frequency",
    "edit_distance",
    "fftconvolve",
    "griffinlim",
    "inverse_spectrogram",
    "linear_fbanks",
    "loudness",
    "mask_along_axis",
    "mask_along_axis_iid",
    "melscale_fbanks",
    "mu_law_decoding",
    "mu_law_encoding",
    "mvdr_weights_rtf",
    "mvdr_weights_souden",
    "phase_vocoder",
    "pitch_shift",
    "preemphasis",
    "psd",
    "resample",
    "rnnt_loss",
    "rtf_evd",
    "rtf_power",
    "sliding_window_cmn",
    "spectral_centroid",
    "spectrogram",
    "speed",
]

def spectrogram(
    waveform: Tensor,
    pad: int,
    window: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    power: float | None,
    normalized: bool | str,
    center: bool = ...,
    pad_mode: str = ...,
    onesided: bool = ...,
    return_complex: bool | None = ...,
) -> Tensor: ...
def inverse_spectrogram(
    spectrogram: Tensor,
    length: int | None,
    pad: int,
    window: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    normalized: bool | str,
    center: bool = ...,
    pad_mode: str = ...,
    onesided: bool = ...,
) -> Tensor: ...
def griffinlim(
    specgram: Tensor,
    window: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    power: float,
    n_iter: int,
    momentum: float,
    length: int | None,
    rand_init: bool,
) -> Tensor: ...
def amplitude_to_DB(
    x: Tensor,
    multiplier: float,
    amin: float,
    db_multiplier: float,
    top_db: float | None = ...,
) -> Tensor: ...
def DB_to_amplitude(x: Tensor, ref: float, power: float) -> Tensor: ...
def melscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    norm: str | None = ...,
    mel_scale: str = ...,
) -> Tensor: ...
def linear_fbanks(n_freqs: int, f_min: float, f_max: float, n_filter: int, sample_rate: int) -> Tensor: ...
def create_dct(n_mfcc: int, n_mels: int, norm: str | None) -> Tensor: ...
def mu_law_encoding(x: Tensor, quantization_channels: int) -> Tensor: ...
def mu_law_decoding(x_mu: Tensor, quantization_channels: int) -> Tensor: ...
def phase_vocoder(complex_specgrams: Tensor, rate: float, phase_advance: Tensor) -> Tensor: ...
def mask_along_axis_iid(
    specgrams: Tensor,
    mask_param: int,
    mask_value: float | Tensor,
    axis: int,
    p: float = ...,
) -> Tensor: ...
def mask_along_axis(
    specgram: Tensor,
    mask_param: int,
    mask_value: float,
    axis: int,
    p: float = ...,
) -> Tensor: ...
def compute_deltas(specgram: Tensor, win_length: int = ..., mode: str = ...) -> Tensor: ...
def detect_pitch_frequency(
    waveform: Tensor,
    sample_rate: int,
    frame_time: float = ...,
    win_length: int = ...,
    freq_low: int = ...,
    freq_high: int = ...,
) -> Tensor: ...
def sliding_window_cmn(
    specgram: Tensor,
    cmn_window: int = ...,
    min_cmn_window: int = ...,
    center: bool = ...,
    norm_vars: bool = ...,
) -> Tensor: ...
def spectral_centroid(
    waveform: Tensor,
    sample_rate: int,
    pad: int,
    window: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> Tensor: ...

_CPU = ...

def resample(
    waveform: Tensor,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = ...,
    rolloff: float = ...,
    resampling_method: str = ...,
    beta: float | None = ...,
) -> Tensor: ...
@torch.jit.unused
def edit_distance(seq1: Sequence, seq2: Sequence) -> int: ...
def loudness(waveform: Tensor, sample_rate: int): ...
def pitch_shift(
    waveform: Tensor,
    sample_rate: int,
    n_steps: int,
    bins_per_octave: int = ...,
    n_fft: int = ...,
    win_length: int | None = ...,
    hop_length: int | None = ...,
    window: Tensor | None = ...,
) -> Tensor: ...

class RnntLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args): ...
    @staticmethod
    def backward(ctx, dy): ...

def psd(
    specgram: Tensor,
    mask: Tensor | None = ...,
    normalize: bool = ...,
    eps: float = ...,
) -> Tensor: ...

rnnt_loss = ...

def mvdr_weights_souden(
    psd_s: Tensor,
    psd_n: Tensor,
    reference_channel: int | Tensor,
    diagonal_loading: bool = ...,
    diag_eps: float = ...,
    eps: float = ...,
) -> Tensor: ...
def mvdr_weights_rtf(
    rtf: Tensor,
    psd_n: Tensor,
    reference_channel: int | Tensor | None = ...,
    diagonal_loading: bool = ...,
    diag_eps: float = ...,
    eps: float = ...,
) -> Tensor: ...
def rtf_evd(psd_s: Tensor) -> Tensor: ...
def rtf_power(
    psd_s: Tensor,
    psd_n: Tensor,
    reference_channel: int | Tensor,
    n_iter: int = ...,
    diagonal_loading: bool = ...,
    diag_eps: float = ...,
) -> Tensor: ...
def apply_beamforming(beamform_weights: Tensor, specgram: Tensor) -> Tensor: ...
def fftconvolve(x: torch.Tensor, y: torch.Tensor, mode: str = ...) -> torch.Tensor: ...
def convolve(x: torch.Tensor, y: torch.Tensor, mode: str = ...) -> torch.Tensor: ...
def add_noise(
    waveform: torch.Tensor,
    noise: torch.Tensor,
    snr: torch.Tensor,
    lengths: torch.Tensor | None = ...,
) -> torch.Tensor: ...
def speed(
    waveform: torch.Tensor,
    orig_freq: int,
    factor: float,
    lengths: torch.Tensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor | None]: ...
def preemphasis(waveform, coeff: float = ...) -> torch.Tensor: ...
def deemphasis(waveform, coeff: float = ...) -> torch.Tensor: ...
def frechet_distance(mu_x, sigma_x, mu_y, sigma_y): ...
