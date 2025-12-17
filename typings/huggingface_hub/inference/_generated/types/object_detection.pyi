from .base import BaseInferenceType, dataclass_with_extra

@dataclass_with_extra
class ObjectDetectionParameters(BaseInferenceType):
    threshold: float | None = ...

@dataclass_with_extra
class ObjectDetectionInput(BaseInferenceType):
    inputs: str
    parameters: ObjectDetectionParameters | None = ...

@dataclass_with_extra
class ObjectDetectionBoundingBox(BaseInferenceType):
    xmax: int
    xmin: int
    ymax: int
    ymin: int

@dataclass_with_extra
class ObjectDetectionOutputElement(BaseInferenceType):
    box: ObjectDetectionBoundingBox
    label: str
    score: float
