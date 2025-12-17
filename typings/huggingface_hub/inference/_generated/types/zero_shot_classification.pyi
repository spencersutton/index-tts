from .base import BaseInferenceType, dataclass_with_extra

@dataclass_with_extra
class ZeroShotClassificationParameters(BaseInferenceType):
    candidate_labels: list[str]
    hypothesis_template: str | None = ...
    multi_label: bool | None = ...

@dataclass_with_extra
class ZeroShotClassificationInput(BaseInferenceType):
    inputs: str
    parameters: ZeroShotClassificationParameters

@dataclass_with_extra
class ZeroShotClassificationOutputElement(BaseInferenceType):
    label: str
    score: float
