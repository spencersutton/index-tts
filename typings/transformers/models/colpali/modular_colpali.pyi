import torch
from transformers.models.paligemma.processing_paligemma import PaliGemmaProcessor

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import is_torch_available

if is_torch_available(): ...
logger = ...

class ColPaliProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class ColPaliProcessor(PaliGemmaProcessor):
    def __init__(
        self,
        image_processor=...,
        tokenizer=...,
        chat_template=...,
        visual_prompt_prefix: str = ...,
        query_prefix: str = ...,
    ) -> None: ...
    @property
    def query_augmentation_token(self) -> str: ...
    def __call__(
        self,
        images: ImageInput = ...,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[ColPaliProcessorKwargs],
    ) -> BatchFeature: ...
    def process_images(self, images: ImageInput = ..., **kwargs: Unpack[ColPaliProcessorKwargs]) -> BatchFeature: ...
    def process_queries(
        self, text: TextInput | list[TextInput], **kwargs: Unpack[ColPaliProcessorKwargs]
    ) -> BatchFeature: ...
    def score_retrieval(
        self,
        query_embeddings: torch.Tensor | list[torch.Tensor],
        passage_embeddings: torch.Tensor | list[torch.Tensor],
        batch_size: int = ...,
        output_dtype: torch.dtype | None = ...,
        output_device: torch.device | str = ...,
    ) -> torch.Tensor: ...

__all__ = ["ColPaliProcessor"]
