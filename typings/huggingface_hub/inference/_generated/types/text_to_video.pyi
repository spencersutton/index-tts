from typing import Any

from .base import BaseInferenceType, dataclass_with_extra

@dataclass_with_extra
class TextToVideoParameters(BaseInferenceType):
    guidance_scale: float | None = ...
    negative_prompt: list[str] | None = ...
    num_frames: float | None = ...
    num_inference_steps: int | None = ...
    seed: int | None = ...

@dataclass_with_extra
class TextToVideoInput(BaseInferenceType):
    inputs: str
    parameters: TextToVideoParameters | None = ...

@dataclass_with_extra
class TextToVideoOutput(BaseInferenceType):
    video: Any
