import functools

import torch
import torchaudio.functional as F
from torch import Tensor


def dynamic_range_compression_torch(x: Tensor, C: float = 1, clip_val: float = 1e-5) -> Tensor:  # noqa: N803
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes: Tensor) -> Tensor:
    return dynamic_range_compression_torch(magnitudes)


N_FFT = 1024
WIN_SIZE = 1024
HOP_SIZE = 256
NUM_MELS = 80
FMIN = 0
SAMPLING_RATE = 22050
FMAX = SAMPLING_RATE / 2.0


@functools.cache
def get_mel_and_window(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    mel = F.melscale_fbanks(
        n_freqs=N_FFT // 2 + 1,
        f_min=FMIN,
        f_max=FMAX,
        n_mels=NUM_MELS,
        sample_rate=SAMPLING_RATE,
        norm="slaney",
        mel_scale="slaney",
    )
    mel = mel.to(device=device, dtype=dtype)
    window = torch.hann_window(WIN_SIZE, device=device, dtype=dtype)
    return mel.T, window


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
    return spectral_normalize_torch(spec)
