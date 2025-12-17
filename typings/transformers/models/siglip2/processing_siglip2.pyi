from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

"""
Image/Text processor class for SigLIP2.
"""

class Siglip2ImagesKwargs(ImagesKwargs, total=False):
    max_num_patches: int | None
    patch_size: int | None

class Siglip2ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Siglip2ImagesKwargs
    _defaults = ...

class Siglip2Processor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor, tokenizer) -> None: ...
    def __call__(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]] | None = ...,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[Siglip2ProcessorKwargs],
    ) -> BatchFeature: ...
    def decode(self, *args, **kwargs): ...
    def batch_decode(self, *args, **kwargs): ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...

__all__ = ["Siglip2Processor"]
