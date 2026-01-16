import torch
from librosa.filters import mel as librosa_mel_fn
from torch import Tensor, nn

from indextts.config import SAMPLING_RATE


def mel_spectrogram(y: Tensor) -> Tensor:
    mel = librosa_mel_fn(sr=SAMPLING_RATE, n_fft=1024, n_mels=80)

    y = nn.functional.pad(y.unsqueeze(1), (384, 384), mode="reflect")
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            1024,
            hop_length=256,
            win_length=1024,
            window=torch.hann_window(1024).to(y.device),
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(torch.from_numpy(mel).float().to(y.device), spec)
    return torch.log(torch.clamp(spec, min=1e-5) * 1)
