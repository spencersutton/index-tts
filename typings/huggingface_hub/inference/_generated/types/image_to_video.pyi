from typing import Any

from .base import BaseInferenceType, dataclass_with_extra

@dataclass_with_extra
class ImageToVideoTargetSize(BaseInferenceType):
    height: int
    width: int

@dataclass_with_extra
class ImageToVideoParameters(BaseInferenceType):
    guidance_scale: float | None = ...
    negative_prompt: str | None = ...
    num_frames: float | None = ...
    num_inference_steps: int | None = ...
    prompt: str | None = ...
    seed: int | None = ...
    target_size: ImageToVideoTargetSize | None = ...

@dataclass_with_extra
class ImageToVideoInput(BaseInferenceType):
    inputs: str
    parameters: ImageToVideoParameters | None = ...

@dataclass_with_extra
class ImageToVideoOutput(BaseInferenceType):
    video: Any
