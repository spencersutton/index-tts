import functools

import torch
import torchaudio.functional as F
from torch import Tensor

N_FFT = 1024  # self.cfg.s2mel["preprocess_params"]["spect_params"]["n_fft"]
WIN_SIZE = 1024  # self.cfg.s2mel["preprocess_params"]["spect_params"]["win_length"]
HOP_SIZE = 256  # self.cfg.s2mel["preprocess_params"]["spect_params"]["hop_length"]
NUM_MELS = 80  # self.cfg.s2mel["preprocess_params"]["spect_params"]["n_mels"]
F_MIN = 0.0  # self.cfg.s2mel["preprocess_params"]["spect_params"].get("fmin", 0)
SAMPLING_RATE = 22050  # self.cfg.s2mel["preprocess_params"]["sr"]
F_MAX = float(SAMPLING_RATE / 2.0)  # self.cfg.s2mel["preprocess_params"]["spect_params"].get("fmax", "None")


@functools.cache
def get_mel_and_window(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    mel = F.melscale_fbanks(
        n_freqs=N_FFT // 2 + 1,
        f_min=F_MIN,
        f_max=F_MAX,
        n_mels=NUM_MELS,
        sample_rate=SAMPLING_RATE,
        norm="slaney",
        mel_scale="slaney",
    )

    # Transpose to match librosa's (n_mels, n_freqs).
    mel = mel.to(device=device, dtype=dtype).t()

    window = torch.hann_window(WIN_SIZE, device=device, dtype=dtype)
    return mel, window


def mel_spectrogram(y: Tensor) -> Tensor:
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        [(N_FFT - HOP_SIZE) // 2, (N_FFT - HOP_SIZE) // 2],
        mode="reflect",
    ).squeeze(1)

    mel_bases, hann_window = get_mel_and_window(y.device, y.dtype)

    spec = torch.view_as_real(
        torch.stft(
            y,
            N_FFT,
            hop_length=HOP_SIZE,
            win_length=WIN_SIZE,
            window=hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    spec = torch.matmul(mel_bases, spec)
    return torch.log(torch.clamp(spec, min=1e-5))
