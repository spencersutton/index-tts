from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import PIL.Image
import torch

from .image_transforms import PaddingMode
from .image_utils import ChannelDimension
from .utils import is_torch_available, is_vision_available

if is_vision_available(): ...
if is_torch_available(): ...
logger = ...
type VideoInput = (
    list[PIL.Image.Image]
    | np.ndarray
    | torch.Tensor
    | list[np.ndarray]
    | list[torch.Tensor]
    | list[list[PIL.Image.Image]]
    | list[list[np.ndarrray]]
    | list[list[torch.Tensor]]
)

@dataclass
class VideoMetadata:
    total_num_frames: int
    fps: float
    duration: float
    video_backend: str
    def __getitem__(self, item):  # -> Any:
        ...

def is_valid_video_frame(frame):  # -> bool:
    ...
def is_valid_video(video):  # -> bool:
    ...
def valid_videos(videos):  # -> bool:
    ...
def is_batched_video(videos):  # -> bool:
    ...
def is_scaled_video(video: np.ndarray) -> bool: ...
def convert_pil_frames_to_video(videos: list[VideoInput]) -> list[np.ndarray | torch.Tensor]: ...
def make_batched_videos(videos) -> list[np.ndarray | torch.Tensor]: ...
def get_video_size(video: np.ndarray, channel_dim: ChannelDimension = ...) -> tuple[int, int]: ...
def get_uniform_frame_indices(
    total_num_frames: int, num_frames: int | None = ...
):  # -> ndarray[tuple[int], dtype[Any]]:

    ...
def default_sample_indices_fn(metadata: VideoMetadata, num_frames=..., fps=..., **kwargs):  # -> _Array1D[Any]:

    ...
def read_video_opencv(video_path: str, sample_indices_fn: Callable, **kwargs):  # -> tuple[NDArray[Any], VideoMetadata]:

    ...
def read_video_decord(
    video_path: str, sample_indices_fn: Callable | None = ..., **kwargs
):  # -> tuple[Any, VideoMetadata]:

    ...
def read_video_pyav(video_path: str, sample_indices_fn: Callable, **kwargs):  # -> tuple[NDArray[Any], VideoMetadata]:

    ...
def read_video_torchvision(video_path: str, sample_indices_fn: Callable, **kwargs):  # -> tuple[Any, VideoMetadata]:

    ...
def read_video_torchcodec(video_path: str, sample_indices_fn: Callable, **kwargs):  # -> tuple[Any, VideoMetadata]:

    ...

VIDEO_DECODERS = ...

def load_video(
    video: str | VideoInput,
    num_frames: int | None = ...,
    fps: float | None = ...,
    backend: str = ...,
    sample_indices_fn: Callable | None = ...,
    **kwargs,
) -> np.array: ...
def convert_to_rgb(
    video: np.array,
    data_format: ChannelDimension | None = ...,
    input_data_format: str | ChannelDimension | None = ...,
) -> np.array: ...
def pad(
    video: np.ndarray,
    padding: int | tuple[int, int] | Iterable[tuple[int, int]],
    mode: PaddingMode = ...,
    constant_values: float | Iterable[float] = ...,
    data_format: str | ChannelDimension | None = ...,
    input_data_format: str | ChannelDimension | None = ...,
) -> np.ndarray: ...
def group_videos_by_shape(
    videos: list[torch.Tensor],
) -> tuple[dict[tuple[int, int], list[torch.Tensor]], dict[int, tuple[tuple[int, int], int]]]: ...
def reorder_videos(
    processed_videos: dict[tuple[int, int], torch.Tensor], grouped_videos_index: dict[int, tuple[int, int]]
) -> list[torch.Tensor]: ...
