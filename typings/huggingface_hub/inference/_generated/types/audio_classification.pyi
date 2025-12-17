from typing import Literal

from .base import BaseInferenceType, dataclass_with_extra

type AudioClassificationOutputTransform = Literal["sigmoid", "softmax", "none"]

@dataclass_with_extra
class AudioClassificationParameters(BaseInferenceType):
    function_to_apply: AudioClassificationOutputTransform | None = ...
    top_k: int | None = ...

@dataclass_with_extra
class AudioClassificationInput(BaseInferenceType):
    inputs: str
    parameters: AudioClassificationParameters | None = ...

@dataclass_with_extra
class AudioClassificationOutputElement(BaseInferenceType):
    label: str
    score: float
