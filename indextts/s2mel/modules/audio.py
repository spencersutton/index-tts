import torch
import torchaudio.functional as F
from torch import Tensor


def dynamic_range_compression_torch(x: Tensor, C: float = 1, clip_val: float = 1e-5) -> Tensor:  # noqa: N803
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes: Tensor) -> Tensor:
    return dynamic_range_compression_torch(magnitudes)


mel_basis: dict[str, Tensor] = {}
hann_window: dict[str, Tensor] = {}


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
    if f"{sampling_rate!s}_{fmax!s}_{y.device!s}" not in mel_basis:
        if fmax is None:
            fmax = sampling_rate / 2.0
        # Use torchaudio's melscale_fbanks instead of librosa's mel
        mel = F.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=fmin,
            f_max=fmax,
            n_mels=num_mels,
            sample_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        # Transpose to match librosa's output shape (n_mels, n_fft//2+1)
        mel_basis[f"{sampling_rate}_{fmax}_{y.device}"] = mel.T.to(y.device)
        hann_window[f"{sampling_rate}_{y.device}"] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(sampling_rate) + "_" + str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(
        mel_basis[str(sampling_rate) + "_" + str(fmax) + "_" + str(y.device)],
        spec,
    )
    return spectral_normalize_torch(spec)
