from .base import BaseInferenceType, dataclass_with_extra

@dataclass_with_extra
class ZeroShotObjectDetectionParameters(BaseInferenceType):
    candidate_labels: list[str]

@dataclass_with_extra
class ZeroShotObjectDetectionInput(BaseInferenceType):
    inputs: str
    parameters: ZeroShotObjectDetectionParameters

@dataclass_with_extra
class ZeroShotObjectDetectionBoundingBox(BaseInferenceType):
    xmax: int
    xmin: int
    ymax: int
    ymin: int

@dataclass_with_extra
class ZeroShotObjectDetectionOutputElement(BaseInferenceType):
    box: ZeroShotObjectDetectionBoundingBox
    label: str
    score: float
