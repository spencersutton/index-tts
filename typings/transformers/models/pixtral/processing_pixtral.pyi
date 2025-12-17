from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import is_vision_available

"""
Processor class for Pixtral.
"""
if is_vision_available(): ...
logger = ...

class PixtralProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

def is_url(val) -> bool: ...
def is_image_or_image_url(elem):  # -> bool:
    ...

class PixtralProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor=...,
        tokenizer=...,
        patch_size: int = ...,
        spatial_merge_size: int = ...,
        chat_template=...,
        image_token=...,
        image_break_token=...,
        image_end_token=...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput = ...,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[PixtralProcessorKwargs],
    ) -> BatchFeature: ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...

__all__ = ["PixtralProcessor"]
