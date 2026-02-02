import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch import Tensor

mel = AF.melscale_fbanks(
    n_freqs=513, f_min=0.0, f_max=11025.0, n_mels=80, sample_rate=22050, norm="slaney", mel_scale="slaney"
).mT

window = torch.hann_window(1024)


def mel_spectrogram(y: Tensor) -> Tensor:
    y = F.pad(y.unsqueeze(1), (384, 384), mode="reflect").squeeze(1)

    spec = torch.view_as_real(y.stft(1024, window=window, center=False, onesided=True, return_complex=True))
    spec = (spec.square().sum(-1) + 1e-9).sqrt()
    spec = mel @ spec
    return spec.clamp(min=1e-5).log()
