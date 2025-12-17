from typing import Any

from .base import BaseInferenceType, dataclass_with_extra

@dataclass_with_extra
class DepthEstimationInput(BaseInferenceType):
    inputs: Any
    parameters: dict[str, Any] | None = ...

@dataclass_with_extra
class DepthEstimationOutput(BaseInferenceType):
    depth: Any
    predicted_depth: Any
