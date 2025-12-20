from typing import Any

from .base import BaseInferenceType, dataclass_with_extra

@dataclass_with_extra
class ImageToImageTargetSize(BaseInferenceType):
    height: int
    width: int

@dataclass_with_extra
class ImageToImageParameters(BaseInferenceType):
    guidance_scale: float | None = ...
    negative_prompt: str | None = ...
    num_inference_steps: int | None = ...
    prompt: str | None = ...
    target_size: ImageToImageTargetSize | None = ...

@dataclass_with_extra
class ImageToImageInput(BaseInferenceType):
    inputs: str
    parameters: ImageToImageParameters | None = ...

@dataclass_with_extra
class ImageToImageOutput(BaseInferenceType):
    image: Any
