import torch
from librosa.filters import mel as librosa_mel_fn
from torch import Tensor


def _dynamic_range_compression_torch(x: Tensor, C: float = 1, clip_val: float = 1e-5) -> Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * C)


def _spectral_normalize_torch(magnitudes: Tensor) -> Tensor:
    output = _dynamic_range_compression_torch(magnitudes)
    return output


_mel_basis = {}
_hann_window = {}


def mel_spectrogram(
    y: Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float,
    center: bool = False,
) -> Tensor:
    global _mel_basis, _hann_window  # pylint: disable=global-statement
    if f"{sampling_rate!s}_{fmax!s}_{y.device!s}" not in _mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis[str(sampling_rate) + "_" + str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        _hann_window[str(sampling_rate) + "_" + str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=_hann_window[str(sampling_rate) + "_" + str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(_mel_basis[str(sampling_rate) + "_" + str(fmax) + "_" + str(y.device)], spec)
    spec = _spectral_normalize_torch(spec)

    return spec
