from typing import Literal

from .base import BaseInferenceType, dataclass_with_extra

type TextClassificationOutputTransform = Literal["sigmoid", "softmax", "none"]

@dataclass_with_extra
class TextClassificationParameters(BaseInferenceType):
    function_to_apply: TextClassificationOutputTransform | None = ...
    top_k: int | None = ...

@dataclass_with_extra
class TextClassificationInput(BaseInferenceType):
    inputs: str
    parameters: TextClassificationParameters | None = ...

@dataclass_with_extra
class TextClassificationOutputElement(BaseInferenceType):
    label: str
    score: float
