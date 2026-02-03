import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch import Tensor

from indextts.constants import AUDIO_NYQUIST_HZ, AUDIO_SAMPLE_RATE, MEL_BINS, STFT_N_FFT, STFT_N_FREQS

mel = AF.melscale_fbanks(
    n_freqs=STFT_N_FREQS,
    f_min=0.0,
    f_max=float(AUDIO_NYQUIST_HZ),
    n_mels=MEL_BINS,
    sample_rate=AUDIO_SAMPLE_RATE,
    norm="slaney",
    mel_scale="slaney",
).mT

window = torch.hann_window(STFT_N_FFT)


def mel_spectrogram(y: Tensor) -> Tensor:
    padding = STFT_N_FFT * 3 // 8
    y = F.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spec = torch.view_as_real(y.stft(STFT_N_FFT, window=window, center=False, onesided=True, return_complex=True))
    spec = (spec.square().sum(-1) + 1e-9).sqrt()
    spec = mel @ spec
    return spec.clamp(min=1e-5).log()
