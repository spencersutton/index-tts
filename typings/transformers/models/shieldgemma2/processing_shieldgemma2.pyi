from collections.abc import Mapping, Sequence

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import Unpack
from ..gemma3.processing_gemma3 import Gemma3Processor, Gemma3ProcessorKwargs

logger = ...
DEFAULT_SHIELDGEMMA2_POLICIES: Mapping[str, str] = ...

class ShieldGemma2ProcessorKwargs(Gemma3ProcessorKwargs, total=False):
    policies: Sequence[str] | None
    custom_policies: Mapping[str, str] | None
    _defaults = ...

class ShieldGemma2Processor(Gemma3Processor):
    def __init__(
        self, image_processor, tokenizer, chat_template=..., image_seq_length=..., policy_definitions=..., **kwargs
    ) -> None: ...
    def __call__(
        self, images: ImageInput = ..., text=..., videos=..., audio=..., **kwargs: Unpack[ShieldGemma2ProcessorKwargs]
    ) -> BatchFeature: ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    @property
    def model_input_names(self):  # -> list[Any]:
        ...

__all__ = ["ShieldGemma2Processor"]
