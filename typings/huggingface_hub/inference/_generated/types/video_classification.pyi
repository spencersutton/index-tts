from typing import Any, Literal

from .base import BaseInferenceType, dataclass_with_extra

type VideoClassificationOutputTransform = Literal["sigmoid", "softmax", "none"]

@dataclass_with_extra
class VideoClassificationParameters(BaseInferenceType):
    frame_sampling_rate: int | None = ...
    function_to_apply: VideoClassificationOutputTransform | None = ...
    num_frames: int | None = ...
    top_k: int | None = ...

@dataclass_with_extra
class VideoClassificationInput(BaseInferenceType):
    inputs: Any
    parameters: VideoClassificationParameters | None = ...

@dataclass_with_extra
class VideoClassificationOutputElement(BaseInferenceType):
    label: str
    score: float
