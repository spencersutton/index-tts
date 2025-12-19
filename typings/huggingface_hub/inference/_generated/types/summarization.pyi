from typing import Any, Literal

from .base import BaseInferenceType, dataclass_with_extra

type SummarizationTruncationStrategy = Literal["do_not_truncate", "longest_first", "only_first", "only_second"]

@dataclass_with_extra
class SummarizationParameters(BaseInferenceType):
    clean_up_tokenization_spaces: bool | None = ...
    generate_parameters: dict[str, Any] | None = ...
    truncation: SummarizationTruncationStrategy | None = ...

@dataclass_with_extra
class SummarizationInput(BaseInferenceType):
    inputs: str
    parameters: SummarizationParameters | None = ...

@dataclass_with_extra
class SummarizationOutput(BaseInferenceType):
    summary_text: str
