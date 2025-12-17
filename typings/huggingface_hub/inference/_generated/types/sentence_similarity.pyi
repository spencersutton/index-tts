from typing import Any

from .base import BaseInferenceType, dataclass_with_extra

@dataclass_with_extra
class SentenceSimilarityInputData(BaseInferenceType):
    sentences: list[str]
    source_sentence: str

@dataclass_with_extra
class SentenceSimilarityInput(BaseInferenceType):
    inputs: SentenceSimilarityInputData
    parameters: dict[str, Any] | None = ...
