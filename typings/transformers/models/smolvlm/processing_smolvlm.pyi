from typing import TYPE_CHECKING

from num2words import num2words

from ...image_utils import ImageInput
from ...processing_utils import AllKwargsForChatTemplate, ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
from ...utils import is_num2words_available, is_vision_available
from ...video_utils import VideoInput

"""
Processor class for SmolVLM.
"""
if is_vision_available(): ...
if is_vision_available(): ...
if TYPE_CHECKING: ...
logger = ...
if is_num2words_available(): ...
else:
    num2words = ...
DEFAULT_CHAT_TEMPLATE = ...

def get_image_prompt_string(
    image_rows, image_cols, image_seq_len, fake_token_around_image, image_token, global_image_token
): ...

class SmolVLMImagesKwargs(ImagesKwargs, total=False):
    return_row_col_info: bool | None
    max_image_size: dict[str, int] | None

class SmolVLMProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: SmolVLMImagesKwargs
    _defaults = ...

class SmolVLMProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    video_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor,
        tokenizer,
        video_processor,
        image_seq_len: int = ...,
        chat_template: str | None = ...,
        **kwargs,
    ) -> None: ...
    def process_vision(self, text, images, output_kwargs):  # -> tuple[None, Any] | tuple[list[Any], Any]:
        ...
    def process_video(self, text, videos, output_kwargs):  # -> tuple[None, Any] | tuple[list[Any], Any]:
        ...
    def __call__(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]] = ...,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = ...,
        audio=...,
        videos: VideoInput = ...,
        **kwargs: Unpack[SmolVLMProcessorKwargs],
    ) -> BatchEncoding: ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        chat_template: str | None = ...,
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ) -> str: ...

__all__ = ["SmolVLMProcessor"]
