from typing import Final

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch import Tensor

N_FFT = 1024
N_MELS = 80
SAMPLING_RATE = 22050


mel: Final = AF.melscale_fbanks(
    n_freqs=N_FFT // 2 + 1,
    f_min=0,
    f_max=SAMPLING_RATE / 2,
    n_mels=N_MELS,
    sample_rate=SAMPLING_RATE,
    norm="slaney",
    mel_scale="slaney",
).mT

window: Final = torch.hann_window(1024)
padding: Final = (N_FFT - (N_FFT // 4)) // 2


def mel_spectrogram(y: Tensor) -> Tensor:
    y = F.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spec = torch.view_as_real(y.stft(N_FFT, window=window, center=False, onesided=True, return_complex=True))
    spec = (spec.square().sum(-1) + 1e-9).sqrt()
    spec = mel @ spec
    return spec.clamp(min=1e-5).log()
