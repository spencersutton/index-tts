import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch import Tensor

N_FFT = 1024
SAMPLING_RATE = 22050


mel = AF.melscale_fbanks(
    n_freqs=N_FFT // 2 + 1,
    f_min=0.0,
    f_max=SAMPLING_RATE / 2.0,
    n_mels=80,
    sample_rate=SAMPLING_RATE,
    norm="slaney",
    mel_scale="slaney",
).mT

window = torch.hann_window(1024)


def mel_spectrogram(y: Tensor) -> Tensor:
    padding = (N_FFT - 256) // 2
    y = F.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spec = torch.view_as_real(y.stft(N_FFT, window=window, center=False, onesided=True, return_complex=True))
    spec = (spec.square().sum(-1) + 1e-9).sqrt()
    spec = mel @ spec
    return spec.clamp(min=1e-5).log()
