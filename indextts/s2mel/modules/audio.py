import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from torch import Tensor


def mel_spectrogram(y: Tensor, sample_rate: int) -> Tensor:
    mel = librosa_mel_fn(sr=sample_rate, n_fft=1024, n_mels=80)

    y = F.pad(y.unsqueeze(1), [384, 384], mode="reflect")
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            1024,
            hop_length=256,
            win_length=1024,
            window=torch.hann_window(1024).to(y.device),
            center=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)

    spec = torch.matmul(torch.from_numpy(mel).float().to(y.device), spec)
    return torch.log(torch.clamp(spec, min=1e-5))
