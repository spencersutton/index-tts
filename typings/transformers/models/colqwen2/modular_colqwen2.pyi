from dataclasses import dataclass

import torch
from transformers.models.colpali.modeling_colpali import ColPaliForRetrieval, ColPaliPreTrainedModel
from transformers.models.colpali.processing_colpali import ColPaliProcessor

from ...cache_utils import Cache
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import ModelOutput, can_return_tuple, is_torch_available
from .configuration_colqwen2 import ColQwen2Config

if is_torch_available(): ...
logger = ...

class ColQwen2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class ColQwen2Processor(ColPaliProcessor):
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(
        self,
        image_processor=...,
        tokenizer=...,
        chat_template=...,
        visual_prompt_prefix: str | None = ...,
        query_prefix: str | None = ...,
        **kwargs,
    ) -> None: ...
    def __call__(
        self,
        images: ImageInput = ...,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[ColQwen2ProcessorKwargs],
    ) -> BatchFeature: ...

class ColQwen2PreTrainedModel(ColPaliPreTrainedModel): ...

@dataclass
class ColQwen2ForRetrievalOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    embeddings: torch.Tensor | None = ...
    past_key_values: list[torch.FloatTensor] | Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

class ColQwen2ForRetrieval(ColPaliForRetrieval):
    _checkpoint_conversion_mapping = ...
    def __init__(self, config: ColQwen2Config) -> None: ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        labels: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        pixel_values: torch.Tensor | None = ...,
        image_grid_thw: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> ColQwen2ForRetrievalOutput: ...

__all__ = ["ColQwen2ForRetrieval", "ColQwen2PreTrainedModel", "ColQwen2Processor"]
