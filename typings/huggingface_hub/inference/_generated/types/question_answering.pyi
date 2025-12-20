from .base import BaseInferenceType, dataclass_with_extra

@dataclass_with_extra
class QuestionAnsweringInputData(BaseInferenceType):
    context: str
    question: str

@dataclass_with_extra
class QuestionAnsweringParameters(BaseInferenceType):
    align_to_words: bool | None = ...
    doc_stride: int | None = ...
    handle_impossible_answer: bool | None = ...
    max_answer_len: int | None = ...
    max_question_len: int | None = ...
    max_seq_len: int | None = ...
    top_k: int | None = ...

@dataclass_with_extra
class QuestionAnsweringInput(BaseInferenceType):
    inputs: QuestionAnsweringInputData
    parameters: QuestionAnsweringParameters | None = ...

@dataclass_with_extra
class QuestionAnsweringOutputElement(BaseInferenceType):
    answer: str
    end: int
    score: float
    start: int
