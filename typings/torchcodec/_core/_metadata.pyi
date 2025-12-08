import pathlib
from dataclasses import dataclass
from fractions import Fraction

import torch

SPACES = ...

@dataclass
class StreamMetadata:
    duration_seconds_from_header: float | None
    begin_stream_seconds_from_header: float | None
    bit_rate: float | None
    codec: str | None
    stream_index: int
    duration_seconds: float | None
    begin_stream_seconds: float | None

@dataclass
class VideoStreamMetadata(StreamMetadata):
    begin_stream_seconds_from_content: float | None
    end_stream_seconds_from_content: float | None
    width: int | None
    height: int | None
    num_frames_from_header: int | None
    num_frames_from_content: int | None
    average_fps_from_header: float | None
    pixel_aspect_ratio: Fraction | None
    end_stream_seconds: float | None
    num_frames: int | None
    average_fps: float | None

@dataclass
class AudioStreamMetadata(StreamMetadata):
    sample_rate: int | None
    num_channels: int | None
    sample_format: str | None

@dataclass
class ContainerMetadata:
    duration_seconds_from_header: float | None
    bit_rate_from_header: float | None
    best_video_stream_index: int | None
    best_audio_stream_index: int | None
    streams: list[StreamMetadata]
    @property
    def duration_seconds(self) -> float | None: ...
    @property
    def bit_rate(self) -> float | None: ...
    @property
    def best_video_stream(self) -> VideoStreamMetadata: ...
    @property
    def best_audio_stream(self) -> AudioStreamMetadata: ...

def get_container_metadata(decoder: torch.Tensor) -> ContainerMetadata: ...
def get_container_metadata_from_header(
    filename: str | pathlib.Path,
) -> ContainerMetadata: ...
