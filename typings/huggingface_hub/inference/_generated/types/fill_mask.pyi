from typing import Any

from .base import BaseInferenceType, dataclass_with_extra

@dataclass_with_extra
class FillMaskParameters(BaseInferenceType):
    targets: list[str] | None = ...
    top_k: int | None = ...

@dataclass_with_extra
class FillMaskInput(BaseInferenceType):
    inputs: str
    parameters: FillMaskParameters | None = ...

@dataclass_with_extra
class FillMaskOutputElement(BaseInferenceType):
    score: float
    sequence: str
    token: int
    token_str: Any
    fill_mask_output_token_str: str | None = ...
