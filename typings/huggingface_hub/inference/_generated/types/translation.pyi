from typing import Any, Literal

from .base import BaseInferenceType, dataclass_with_extra

type TranslationTruncationStrategy = Literal["do_not_truncate", "longest_first", "only_first", "only_second"]

@dataclass_with_extra
class TranslationParameters(BaseInferenceType):
    clean_up_tokenization_spaces: bool | None = ...
    generate_parameters: dict[str, Any] | None = ...
    src_lang: str | None = ...
    tgt_lang: str | None = ...
    truncation: TranslationTruncationStrategy | None = ...

@dataclass_with_extra
class TranslationInput(BaseInferenceType):
    inputs: str
    parameters: TranslationParameters | None = ...

@dataclass_with_extra
class TranslationOutput(BaseInferenceType):
    translation_text: str
