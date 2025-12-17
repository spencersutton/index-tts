from typing import Any, Literal

from .base import BaseInferenceType, dataclass_with_extra

type Text2TextGenerationTruncationStrategy = Literal["do_not_truncate", "longest_first", "only_first", "only_second"]

@dataclass_with_extra
class Text2TextGenerationParameters(BaseInferenceType):
    clean_up_tokenization_spaces: bool | None = ...
    generate_parameters: dict[str, Any] | None = ...
    truncation: Text2TextGenerationTruncationStrategy | None = ...

@dataclass_with_extra
class Text2TextGenerationInput(BaseInferenceType):
    inputs: str
    parameters: Text2TextGenerationParameters | None = ...

@dataclass_with_extra
class Text2TextGenerationOutput(BaseInferenceType):
    generated_text: Any
    text2_text_generation_output_generated_text: str | None = ...
