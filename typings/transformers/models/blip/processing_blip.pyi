from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput

"""
Processor class for Blip.
"""

class BlipProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class BlipProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor, tokenizer, **kwargs) -> None: ...
    def __call__(
        self,
        images: ImageInput = ...,
        text: str | list[str] | TextInput | PreTokenizedInput | None = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[BlipProcessorKwargs],
    ) -> BatchEncoding: ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...

__all__ = ["BlipProcessor"]
