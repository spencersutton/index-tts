import numpy as np
from typing import Optional, Union

def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array: ...
def ffmpeg_microphone(
    sampling_rate: int,
    chunk_length_s: float,
    format_for_conversion: str = ...,
    ffmpeg_input_device: Optional[str] = ...,
    ffmpeg_additional_args: Optional[list[str]] = ...,
):  # -> Generator[bytes, Any, None]:

    ...
def ffmpeg_microphone_live(
    sampling_rate: int,
    chunk_length_s: float,
    stream_chunk_s: Optional[int] = ...,
    stride_length_s: Optional[Union[tuple[float, float], float]] = ...,
    format_for_conversion: str = ...,
    ffmpeg_input_device: Optional[str] = ...,
    ffmpeg_additional_args: Optional[list[str]] = ...,
):  # -> Generator[dict[str, bytes | tuple[int, Literal[0]] | bool] | dict[str, bytes | tuple[int, int]], Any, None]:

    ...
def chunk_bytes_iter(
    iterator, chunk_len: int, stride: tuple[int, int], stream: bool = ...
):  # -> Generator[dict[str, Any | tuple[int, Literal[0]] | bool] | dict[str, Any | tuple[int, int]] | dict[str, Any | bytes | tuple[int, int]], Any, None]:

    ...
