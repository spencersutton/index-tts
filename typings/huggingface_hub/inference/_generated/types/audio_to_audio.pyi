from typing import Any

from .base import BaseInferenceType, dataclass_with_extra

@dataclass_with_extra
class AudioToAudioInput(BaseInferenceType):
    inputs: Any

@dataclass_with_extra
class AudioToAudioOutputElement(BaseInferenceType):
    blob: Any
    content_type: str
    label: str
