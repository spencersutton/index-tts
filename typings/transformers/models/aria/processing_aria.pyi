from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils import PreTokenizedInput, TextInput
from ..auto import AutoTokenizer

class AriaProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class AriaProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor=...,
        tokenizer: AutoTokenizer | str = ...,
        chat_template: str | None = ...,
        size_conversion: dict[float | int, int] | None = ...,
    ) -> None: ...
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        images: ImageInput | None = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[AriaProcessorKwargs],
    ) -> BatchFeature: ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...

__all__ = ["AriaProcessor"]
