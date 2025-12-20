import torch
from torch import nn
from transformers.models.llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
    LlavaPreTrainedModel,
    TransformersKwargs,
)

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...processing_utils import Unpack
from ...utils.generic import check_model_inputs
from .configuration_aya_vision import AyaVisionConfig

"""PyTorch AyaVision model."""
logger = ...

class AyaVisionMultiModalProjector(nn.Module):
    def __init__(self, config: AyaVisionConfig) -> None: ...
    def forward(self, image_features):  # -> Any:
        ...
    def pixel_shuffle(self, image_features): ...

class AyaVisionPreTrainedModel(LlavaPreTrainedModel):
    _can_compile_fullgraph = ...
    _can_record_outputs = ...

class AyaVisionCausalLMOutputWithPast(LlavaCausalLMOutputWithPast): ...
class AyaVisionModelOutputWithPast(LlavaModelOutputWithPast): ...

class AyaVisionModel(LlavaModel):
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        **kwargs,
    ):  # -> Any:

        ...
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        pixel_values: torch.FloatTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | AyaVisionModelOutputWithPast: ...

class AyaVisionForConditionalGeneration(LlavaForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        image_sizes: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | AyaVisionCausalLMOutputWithPast: ...

__all__ = ["AyaVisionForConditionalGeneration", "AyaVisionModel", "AyaVisionPreTrainedModel"]
