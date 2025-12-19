from typing import Any

from .base import BaseInferenceType, dataclass_with_extra

@dataclass_with_extra
class VisualQuestionAnsweringInputData(BaseInferenceType):
    image: Any
    question: str

@dataclass_with_extra
class VisualQuestionAnsweringParameters(BaseInferenceType):
    top_k: int | None = ...

@dataclass_with_extra
class VisualQuestionAnsweringInput(BaseInferenceType):
    inputs: VisualQuestionAnsweringInputData
    parameters: VisualQuestionAnsweringParameters | None = ...

@dataclass_with_extra
class VisualQuestionAnsweringOutputElement(BaseInferenceType):
    score: float
    answer: str | None = ...
