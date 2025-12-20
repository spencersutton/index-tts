from typing import Any

from .base import BaseInferenceType, dataclass_with_extra

@dataclass_with_extra
class DocumentQuestionAnsweringInputData(BaseInferenceType):
    image: Any
    question: str

@dataclass_with_extra
class DocumentQuestionAnsweringParameters(BaseInferenceType):
    doc_stride: int | None = ...
    handle_impossible_answer: bool | None = ...
    lang: str | None = ...
    max_answer_len: int | None = ...
    max_question_len: int | None = ...
    max_seq_len: int | None = ...
    top_k: int | None = ...
    word_boxes: list[list[float] | str] | None = ...

@dataclass_with_extra
class DocumentQuestionAnsweringInput(BaseInferenceType):
    inputs: DocumentQuestionAnsweringInputData
    parameters: DocumentQuestionAnsweringParameters | None = ...

@dataclass_with_extra
class DocumentQuestionAnsweringOutputElement(BaseInferenceType):
    answer: str
    end: int
    score: float
    start: int
