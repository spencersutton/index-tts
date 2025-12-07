from torch import Tensor

def allpass_biquad(waveform: Tensor, sample_rate: int, central_freq: float, Q: float = ...) -> Tensor: ...
def band_biquad(
    waveform: Tensor,
    sample_rate: int,
    central_freq: float,
    Q: float = ...,
    noise: bool = ...,
) -> Tensor: ...
def bandpass_biquad(
    waveform: Tensor,
    sample_rate: int,
    central_freq: float,
    Q: float = ...,
    const_skirt_gain: bool = ...,
) -> Tensor: ...
def bandreject_biquad(waveform: Tensor, sample_rate: int, central_freq: float, Q: float = ...) -> Tensor: ...
def bass_biquad(
    waveform: Tensor,
    sample_rate: int,
    gain: float,
    central_freq: float = ...,
    Q: float = ...,
) -> Tensor: ...
def biquad(
    waveform: Tensor,
    b0: float,
    b1: float,
    b2: float,
    a0: float,
    a1: float,
    a2: float,
) -> Tensor: ...
def contrast(waveform: Tensor, enhancement_amount: float = ...) -> Tensor: ...
def dcshift(waveform: Tensor, shift: float, limiter_gain: float | None = ...) -> Tensor: ...
def deemph_biquad(waveform: Tensor, sample_rate: int) -> Tensor: ...
def dither(waveform: Tensor, density_function: str = ..., noise_shaping: bool = ...) -> Tensor: ...
def equalizer_biquad(
    waveform: Tensor,
    sample_rate: int,
    center_freq: float,
    gain: float,
    Q: float = ...,
) -> Tensor: ...
def filtfilt(waveform: Tensor, a_coeffs: Tensor, b_coeffs: Tensor, clamp: bool = ...) -> Tensor: ...
def flanger(
    waveform: Tensor,
    sample_rate: int,
    delay: float = ...,
    depth: float = ...,
    regen: float = ...,
    width: float = ...,
    speed: float = ...,
    phase: float = ...,
    modulation: str = ...,
    interpolation: str = ...,
) -> Tensor: ...
def gain(waveform: Tensor, gain_db: float = ...) -> Tensor: ...
def highpass_biquad(waveform: Tensor, sample_rate: int, cutoff_freq: float, Q: float = ...) -> Tensor: ...

_lfilter_core_cpu_loop = ...
_lfilter = ...

def lfilter(
    waveform: Tensor,
    a_coeffs: Tensor,
    b_coeffs: Tensor,
    clamp: bool = ...,
    batching: bool = ...,
) -> Tensor: ...
def lowpass_biquad(waveform: Tensor, sample_rate: int, cutoff_freq: float, Q: float = ...) -> Tensor: ...

_overdrive_core_loop_cpu = ...

def overdrive(waveform: Tensor, gain: float = ..., colour: float = ...) -> Tensor: ...
def phaser(
    waveform: Tensor,
    sample_rate: int,
    gain_in: float = ...,
    gain_out: float = ...,
    delay_ms: float = ...,
    decay: float = ...,
    mod_speed: float = ...,
    sinusoidal: bool = ...,
) -> Tensor: ...
def riaa_biquad(waveform: Tensor, sample_rate: int) -> Tensor: ...
def treble_biquad(
    waveform: Tensor,
    sample_rate: int,
    gain: float,
    central_freq: float = ...,
    Q: float = ...,
) -> Tensor: ...
def vad(
    waveform: Tensor,
    sample_rate: int,
    trigger_level: float = ...,
    trigger_time: float = ...,
    search_time: float = ...,
    allowed_gap: float = ...,
    pre_trigger_time: float = ...,
    boot_time: float = ...,
    noise_up_time: float = ...,
    noise_down_time: float = ...,
    noise_reduction_amount: float = ...,
    measure_freq: float = ...,
    measure_duration: float | None = ...,
    measure_smooth_time: float = ...,
    hp_filter_freq: float = ...,
    lp_filter_freq: float = ...,
    hp_lifter_freq: float = ...,
    lp_lifter_freq: float = ...,
) -> Tensor: ...
