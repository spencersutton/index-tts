from dataclasses import dataclass

import torch
import torch.nn as nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, can_return_tuple
from .configuration_internvl import InternVLConfig, InternVLVisionConfig

@use_kernel_forward_from_hub("RMSNorm")
class InternVLVisionRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class InternVLVisionAttention(nn.Module):
    def __init__(self, config: InternVLVisionConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: torch.Tensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):  # -> tuple[Any, Any] | tuple[Any, None]:
        ...

class InternVLVisionPreTrainedModel(PreTrainedModel):
    config: InternVLVisionConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_sdpa = ...
    _supports_flash_attn = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...

@dataclass
class InternVLVisionModelOutputWithPooling(BaseModelOutputWithPooling): ...

class InternVLVisionPatchEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: ...

class InternVLVisionEmbeddings(nn.Module):
    def __init__(self, config: InternVLVisionConfig) -> None: ...
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor: ...
    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.BoolTensor | None = ...) -> torch.Tensor: ...

class InternVLVisionMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

NORM2FN = ...

class InternVLVisionLayer(GradientCheckpointingLayer):
    def __init__(self, config: InternVLVisionConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, output_attentions: bool = ...
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]: ...

class InternVLVisionEncoder(nn.Module):
    def __init__(self, config: InternVLVisionConfig) -> None: ...
    @can_return_tuple
    def forward(
        self, hidden_states: torch.Tensor, output_attentions: bool = ..., output_hidden_states: bool = ...
    ) -> tuple | BaseModelOutput: ...

class InternVLVisionModel(InternVLVisionPreTrainedModel):
    def __init__(self, config: InternVLVisionConfig) -> None: ...
    def get_input_embeddings(self):  # -> InternVLVisionPatchEmbeddings:
        ...
    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> tuple | InternVLVisionModelOutputWithPooling: ...

class InternVLPreTrainedModel(PreTrainedModel):
    config: InternVLConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...

class InternVLMultiModalProjector(nn.Module):
    def __init__(self, config: InternVLConfig) -> None: ...
    def forward(self, image_features):  # -> Any:
        ...

@dataclass
class InternVLModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: torch.FloatTensor | None = ...

class InternVLModel(InternVLPreTrainedModel):
    _checkpoint_conversion_mapping = ...
    def __init__(self, config: InternVLConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        **kwargs,
    ):  # -> Any:

        ...
    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):  # -> Tensor | Any:

        ...
    @can_return_tuple
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
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | InternVLModelOutputWithPast: ...
    def pixel_shuffle(self, vision_features: torch.Tensor, scale_factor: float = ...):  # -> Tensor:

        ...

@dataclass
class InternVLCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    past_key_values: list[torch.FloatTensor] | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    image_hidden_states: torch.FloatTensor | None = ...

class InternVLForConditionalGeneration(InternVLPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = ...
    _tied_weights_keys = ...
    def __init__(self, config: InternVLConfig) -> None: ...
    def get_input_embeddings(self):  # -> Any:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self) -> nn.Module: ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> Any:
        ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: int | list[int] | None = ...,
        vision_feature_select_strategy: str | None = ...,
        **kwargs,
    ):  # -> Any:
        ...
    @property
    def language_model(self):  # -> Any:
        ...
    @property
    def vision_tower(self):  # -> Any:
        ...
    @property
    def multi_modal_projector(self):  # -> InternVLMultiModalProjector:
        ...
    @can_return_tuple
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
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        image_sizes: torch.Tensor | None = ...,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | InternVLCausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        inputs_embeds=...,
        pixel_values=...,
        attention_mask=...,
        cache_position=...,
        logits_to_keep=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
        ...

__all__ = [
    "InternVLForConditionalGeneration",
    "InternVLModel",
    "InternVLPreTrainedModel",
    "InternVLVisionModel",
    "InternVLVisionPreTrainedModel",
]
