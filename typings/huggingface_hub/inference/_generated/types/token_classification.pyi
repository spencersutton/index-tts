from typing import Literal

from .base import BaseInferenceType, dataclass_with_extra

type TokenClassificationAggregationStrategy = Literal["none", "simple", "first", "average", "max"]

@dataclass_with_extra
class TokenClassificationParameters(BaseInferenceType):
    aggregation_strategy: TokenClassificationAggregationStrategy | None = ...
    ignore_labels: list[str] | None = ...
    stride: int | None = ...

@dataclass_with_extra
class TokenClassificationInput(BaseInferenceType):
    inputs: str
    parameters: TokenClassificationParameters | None = ...

@dataclass_with_extra
class TokenClassificationOutputElement(BaseInferenceType):
    end: int
    score: float
    start: int
    word: str
    entity: str | None = ...
    entity_group: str | None = ...
