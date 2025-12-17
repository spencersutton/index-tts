import torch
from torch import nn
from transformers.models.llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
    LlavaPreTrainedModel,
)

from ...cache_utils import Cache
from .configuration_vipllava import VipLlavaConfig

logger = ...

class VipLlavaModelOutputWithPast(LlavaModelOutputWithPast): ...
class VipLlavaCausalLMOutputWithPast(LlavaCausalLMOutputWithPast): ...

class VipLlavaMultiModalProjector(nn.Module):
    def __init__(self, config: VipLlavaConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class VipLlavaPreTrainedModel(LlavaPreTrainedModel): ...

class VipLlavaModel(LlavaModel):
    def get_image_features(
        self, pixel_values: torch.FloatTensor, vision_feature_layers: int | list[int] | None = ...
    ):  # -> Any:

        ...
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        pixel_values: torch.FloatTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        vision_feature_layers: int | list[int] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **lm_kwargs,
    ) -> tuple | VipLlavaModelOutputWithPast: ...

class VipLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    def get_image_features(
        self, pixel_values: torch.FloatTensor, vision_feature_layers: int | list[int] | None = ...
    ):  # -> tuple[Tensor, ...] | list[Any]:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        pixel_values: torch.FloatTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        vision_feature_layers: int | list[int] | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **lm_kwargs,
    ) -> tuple | VipLlavaCausalLMOutputWithPast: ...

__all__ = ["VipLlavaForConditionalGeneration", "VipLlavaModel", "VipLlavaPreTrainedModel"]
