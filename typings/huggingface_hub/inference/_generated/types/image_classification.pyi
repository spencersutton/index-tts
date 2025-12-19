from typing import Literal

from .base import BaseInferenceType, dataclass_with_extra

type ImageClassificationOutputTransform = Literal["sigmoid", "softmax", "none"]

@dataclass_with_extra
class ImageClassificationParameters(BaseInferenceType):
    function_to_apply: ImageClassificationOutputTransform | None = ...
    top_k: int | None = ...

@dataclass_with_extra
class ImageClassificationInput(BaseInferenceType):
    inputs: str
    parameters: ImageClassificationParameters | None = ...

@dataclass_with_extra
class ImageClassificationOutputElement(BaseInferenceType):
    label: str
    score: float
