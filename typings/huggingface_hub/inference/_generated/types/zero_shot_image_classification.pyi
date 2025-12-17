from .base import BaseInferenceType, dataclass_with_extra

@dataclass_with_extra
class ZeroShotImageClassificationParameters(BaseInferenceType):
    candidate_labels: list[str]
    hypothesis_template: str | None = ...

@dataclass_with_extra
class ZeroShotImageClassificationInput(BaseInferenceType):
    inputs: str
    parameters: ZeroShotImageClassificationParameters

@dataclass_with_extra
class ZeroShotImageClassificationOutputElement(BaseInferenceType):
    label: str
    score: float
