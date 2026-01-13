import torch
from librosa.filters import mel as librosa_mel_fn
from torch import Tensor, nn


def dynamic_range_compression_torch(x: Tensor) -> Tensor:
    return torch.log(torch.clamp(x, min=1e-5) * 1)


def mel_spectrogram(
    y: Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float | None,
    center: bool = False,
) -> Tensor:
    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)

    y = nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect")
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=torch.hann_window(win_size).to(y.device),
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(torch.from_numpy(mel).float().to(y.device), spec)
    return torch.log(torch.clamp(spec, min=1e-5) * 1)
