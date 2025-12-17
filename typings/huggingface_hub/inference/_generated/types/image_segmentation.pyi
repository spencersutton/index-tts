from typing import Literal

from .base import BaseInferenceType, dataclass_with_extra

type ImageSegmentationSubtask = Literal["instance", "panoptic", "semantic"]

@dataclass_with_extra
class ImageSegmentationParameters(BaseInferenceType):
    mask_threshold: float | None = ...
    overlap_mask_area_threshold: float | None = ...
    subtask: ImageSegmentationSubtask | None = ...
    threshold: float | None = ...

@dataclass_with_extra
class ImageSegmentationInput(BaseInferenceType):
    inputs: str
    parameters: ImageSegmentationParameters | None = ...

@dataclass_with_extra
class ImageSegmentationOutputElement(BaseInferenceType):
    label: str
    mask: str
    score: float | None = ...
