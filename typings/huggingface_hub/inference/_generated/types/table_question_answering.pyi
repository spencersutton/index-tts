from typing import Literal

from .base import BaseInferenceType, dataclass_with_extra

@dataclass_with_extra
class TableQuestionAnsweringInputData(BaseInferenceType):
    question: str
    table: dict[str, list[str]]

type Padding = Literal["do_not_pad", "longest", "max_length"]

@dataclass_with_extra
class TableQuestionAnsweringParameters(BaseInferenceType):
    padding: Padding | None = ...
    sequential: bool | None = ...
    truncation: bool | None = ...

@dataclass_with_extra
class TableQuestionAnsweringInput(BaseInferenceType):
    inputs: TableQuestionAnsweringInputData
    parameters: TableQuestionAnsweringParameters | None = ...

@dataclass_with_extra
class TableQuestionAnsweringOutputElement(BaseInferenceType):
    answer: str
    cells: list[str]
    coordinates: list[list[int]]
    aggregator: str | None = ...
