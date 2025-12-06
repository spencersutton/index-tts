
from torch import Tensor

__all__ = [
    "fbank",
    "get_mel_banks",
    "inverse_mel_scale",
    "inverse_mel_scale_scalar",
    "mel_scale",
    "mel_scale_scalar",
    "mfcc",
    "spectrogram",
    "vtln_warp_freq",
    "vtln_warp_mel_freq",
]
EPSILON = ...
MILLISECONDS_TO_SECONDS = ...
HAMMING = ...
HANNING = ...
POVEY = ...
RECTANGULAR = ...
BLACKMAN = ...
WINDOWS = ...

def spectrogram(
    waveform: Tensor,
    blackman_coeff: float = ...,
    channel: int = ...,
    dither: float = ...,
    energy_floor: float = ...,
    frame_length: float = ...,
    frame_shift: float = ...,
    min_duration: float = ...,
    preemphasis_coefficient: float = ...,
    raw_energy: bool = ...,
    remove_dc_offset: bool = ...,
    round_to_power_of_two: bool = ...,
    sample_frequency: float = ...,
    snip_edges: bool = ...,
    subtract_mean: bool = ...,
    window_type: str = ...,
) -> Tensor: ...
def inverse_mel_scale_scalar(mel_freq: float) -> float: ...
def inverse_mel_scale(mel_freq: Tensor) -> Tensor: ...
def mel_scale_scalar(freq: float) -> float: ...
def mel_scale(freq: Tensor) -> Tensor: ...
def vtln_warp_freq(
    vtln_low_cutoff: float,
    vtln_high_cutoff: float,
    low_freq: float,
    high_freq: float,
    vtln_warp_factor: float,
    freq: Tensor,
) -> Tensor: ...
def vtln_warp_mel_freq(
    vtln_low_cutoff: float,
    vtln_high_cutoff: float,
    low_freq,
    high_freq: float,
    vtln_warp_factor: float,
    mel_freq: Tensor,
) -> Tensor: ...
def get_mel_banks(
    num_bins: int,
    window_length_padded: int,
    sample_freq: float,
    low_freq: float,
    high_freq: float,
    vtln_low: float,
    vtln_high: float,
    vtln_warp_factor: float,
) -> tuple[Tensor, Tensor]: ...
def fbank(
    waveform: Tensor,
    blackman_coeff: float = ...,
    channel: int = ...,
    dither: float = ...,
    energy_floor: float = ...,
    frame_length: float = ...,
    frame_shift: float = ...,
    high_freq: float = ...,
    htk_compat: bool = ...,
    low_freq: float = ...,
    min_duration: float = ...,
    num_mel_bins: int = ...,
    preemphasis_coefficient: float = ...,
    raw_energy: bool = ...,
    remove_dc_offset: bool = ...,
    round_to_power_of_two: bool = ...,
    sample_frequency: float = ...,
    snip_edges: bool = ...,
    subtract_mean: bool = ...,
    use_energy: bool = ...,
    use_log_fbank: bool = ...,
    use_power: bool = ...,
    vtln_high: float = ...,
    vtln_low: float = ...,
    vtln_warp: float = ...,
    window_type: str = ...,
) -> Tensor: ...
def mfcc(
    waveform: Tensor,
    blackman_coeff: float = ...,
    cepstral_lifter: float = ...,
    channel: int = ...,
    dither: float = ...,
    energy_floor: float = ...,
    frame_length: float = ...,
    frame_shift: float = ...,
    high_freq: float = ...,
    htk_compat: bool = ...,
    low_freq: float = ...,
    num_ceps: int = ...,
    min_duration: float = ...,
    num_mel_bins: int = ...,
    preemphasis_coefficient: float = ...,
    raw_energy: bool = ...,
    remove_dc_offset: bool = ...,
    round_to_power_of_two: bool = ...,
    sample_frequency: float = ...,
    snip_edges: bool = ...,
    subtract_mean: bool = ...,
    use_energy: bool = ...,
    vtln_high: float = ...,
    vtln_low: float = ...,
    vtln_warp: float = ...,
    window_type: str = ...,
) -> Tensor: ...
