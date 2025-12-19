from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput

"""
Image/Text processor class for Chinese-CLIP
"""

class ChineseClipProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class ChineseCLIPProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor=..., tokenizer=..., **kwargs) -> None: ...
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = ...,
        images: ImageInput = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[ChineseClipProcessorKwargs],
    ) -> BatchEncoding: ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...
    @property
    def feature_extractor_class(
        self,
    ):  # -> tuple[Literal['ChineseCLIPImageProcessor'], Literal['ChineseCLIPImageProcessorFast']]:
        ...

__all__ = ["ChineseCLIPProcessor"]
