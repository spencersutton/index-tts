import torch
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def _dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def _spectral_normalize_torch(magnitudes):
    output = _dynamic_range_compression_torch(magnitudes)
    return output


_mel_basis = {}
_hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
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
