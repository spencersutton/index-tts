from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

"""
Image/Text processor class for GIT
"""

class GitProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

logger = ...

class GitProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor, tokenizer) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[GitProcessorKwargs],
    ) -> BatchFeature: ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    @property
    def model_input_names(self):  # -> list[str]:
        ...

__all__ = ["GitProcessor"]
