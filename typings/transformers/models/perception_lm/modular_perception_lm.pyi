import torch
from torch import nn

from ...utils import can_return_tuple
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
    LlavaPreTrainedModel,
)
from .configuration_perception_lm import PerceptionLMConfig

"""PyTorch PerceptionLM model."""
logger = ...

class PerceptionLMAdaptiveAvgPooling(nn.Module):
    def __init__(self, pooling_ratio=...) -> None: ...
    def forward(self, hidden_states):  # -> Tensor:
        ...

class PerceptionLMMultiModalProjector(nn.Module):
    def __init__(self, config: PerceptionLMConfig) -> None: ...
    def forward(self, features):  # -> Any:
        ...

class PerceptionLMPreTrainedModel(LlavaPreTrainedModel):
    base_model_prefix = ...

class PerceptionLMModelOutputWithPast(LlavaModelOutputWithPast):
    video_hidden_states: torch.FloatTensor | None = ...

class PerceptionLMCausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    video_hidden_states: torch.FloatTensor | None = ...

class PerceptionLMModel(LlavaModel):
    _checkpoint_conversion_mapping = ...
    def __init__(self, config: PerceptionLMConfig) -> None: ...
    def get_image_features(self, pixel_values: torch.FloatTensor, **kwargs):  # -> Any:

        ...
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor = ...,
        video_features: torch.FloatTensor = ...,
    ):  # -> tuple[Any, Any]:

        ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_values_videos: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **lm_kwargs,
    ) -> tuple | PerceptionLMModelOutputWithPast: ...

class PerceptionLMForConditionalGeneration(LlavaForConditionalGeneration):
    _checkpoint_conversion_mapping = ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        inputs_embeds=...,
        pixel_values=...,
        pixel_values_videos=...,
        attention_mask=...,
        cache_position=...,
        logits_to_keep=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_values_videos: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **lm_kwargs,
    ) -> tuple | PerceptionLMCausalLMOutputWithPast: ...
    def get_image_features(self, **kwargs): ...
    def language_model(self): ...
    def vision_tower(self): ...
    def multi_modal_projector(self): ...

__all__ = ["PerceptionLMForConditionalGeneration", "PerceptionLMModel", "PerceptionLMPreTrainedModel"]
