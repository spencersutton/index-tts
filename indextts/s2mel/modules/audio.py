import functools
import math

import torch
from torch import Tensor

N_FFT = 1024  # self.cfg.s2mel["preprocess_params"]["spect_params"]["n_fft"]
WIN_SIZE = 1024  # self.cfg.s2mel["preprocess_params"]["spect_params"]["win_length"]
HOP_SIZE = 256  # self.cfg.s2mel["preprocess_params"]["spect_params"]["hop_length"]
NUM_MELS = 80  # self.cfg.s2mel["preprocess_params"]["spect_params"]["n_mels"]
F_MIN = 0.0  # self.cfg.s2mel["preprocess_params"]["spect_params"].get("fmin", 0)
SAMPLING_RATE = 22050  # self.cfg.s2mel["preprocess_params"]["sr"]
F_MAX = float(SAMPLING_RATE / 2.0)  # self.cfg.s2mel["preprocess_params"]["spect_params"].get("fmax", "None")


def _hz_to_mel_slaney(hz: Tensor) -> Tensor:
    """Convert Hz to mel using Slaney-style formula (librosa default)."""
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0

    hz = torch.clamp(hz, min=0)
    mel = hz / f_sp
    log_region = hz >= min_log_hz
    return torch.where(log_region, min_log_mel + torch.log(hz / min_log_hz) / logstep, mel)


def _mel_to_hz_slaney(mel: Tensor) -> Tensor:
    """Convert mel to Hz using Slaney-style formula (inverse of `_hz_to_mel_slaney`)."""
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0

    mel = torch.clamp(mel, min=0)
    hz = f_sp * mel
    log_region = mel >= min_log_mel
    return torch.where(log_region, min_log_hz * torch.exp(logstep * (mel - min_log_mel)), hz)


def _librosa_mel_filterbank(
    *,
    n_freqs: int,
    n_mels: int,
    sample_rate: int,
    f_min: float,
    f_max: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create a mel filterbank compatible with librosa's `filters.mel` defaults.

    This uses Slaney mel scale and Slaney normalization (area normalization).
    """
    # Compute FFT bin center frequencies.
    freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs, device=device, dtype=torch.float32)

    # Mel frequencies (n_mels + 2) defining the triangular bands.
    min_mel = _hz_to_mel_slaney(torch.tensor([f_min], device=device, dtype=torch.float32))[0]
    max_mel = _hz_to_mel_slaney(torch.tensor([f_max], device=device, dtype=torch.float32))[0]
    mels = torch.linspace(min_mel, max_mel, n_mels + 2, device=device, dtype=torch.float32)
    hz = _mel_to_hz_slaney(mels)

    # Triangular filters.
    fb = torch.zeros((n_mels, n_freqs), device=device, dtype=torch.float32)
    for i in range(n_mels):
        lower = hz[i]
        center = hz[i + 1]
        upper = hz[i + 2]

        # Avoid division by zero for pathological settings.
        up_denom = torch.clamp(center - lower, min=1e-10)
        down_denom = torch.clamp(upper - center, min=1e-10)

        up_slope = (freqs - lower) / up_denom
        down_slope = (upper - freqs) / down_denom
        fb[i] = torch.clamp(torch.minimum(up_slope, down_slope), min=0.0)

    # Slaney-style energy normalization: normalize by the bandwidth of each filter.
    enorm = 2.0 / torch.clamp(hz[2 : n_mels + 2] - hz[:n_mels], min=1e-10)
    fb *= enorm.unsqueeze(1)

    return fb.to(dtype=dtype)


@functools.cache
def get_mel_and_window(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    mel = _librosa_mel_filterbank(
        n_freqs=N_FFT // 2 + 1,
        f_min=F_MIN,
        f_max=F_MAX,
        n_mels=NUM_MELS,
        sample_rate=SAMPLING_RATE,
        device=device,
        dtype=dtype,
    )
    window = torch.hann_window(WIN_SIZE, device=device, dtype=dtype)
    return mel, window


def mel_spectrogram(y: Tensor) -> Tensor:
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        [(N_FFT - HOP_SIZE) // 2, (N_FFT - HOP_SIZE) // 2],
        mode="reflect",
    ).squeeze(1)

    mel_bases, hann_window = get_mel_and_window(y.device, y.dtype)

    spec = torch.view_as_real(
        torch.stft(
            y,
            N_FFT,
            hop_length=HOP_SIZE,
            win_length=WIN_SIZE,
            window=hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    spec = torch.matmul(mel_bases, spec)
    return torch.log(torch.clamp(spec, min=1e-5))
