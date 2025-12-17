from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, can_return_tuple
from .configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig

"""PyTorch Siglip model."""

def trunc_normal_tf_(
    tensor: torch.Tensor, mean: float = ..., std: float = ..., a: float = ..., b: float = ...
) -> torch.Tensor: ...
def variance_scaling_(tensor, scale=..., mode=..., distribution=...):  # -> None:
    ...
def lecun_normal_(tensor):  # -> None:
    ...
def default_flax_embed_init(tensor):  # -> None:
    ...

@dataclass
class SiglipVisionModelOutput(ModelOutput):
    image_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
class SiglipTextModelOutput(ModelOutput):
    text_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
class SiglipOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits_per_image: torch.FloatTensor | None = ...
    logits_per_text: torch.FloatTensor | None = ...
    text_embeds: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None: ...
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor: ...
    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=...) -> torch.Tensor: ...

class SiglipTextEmbeddings(nn.Module):
    def __init__(self, config: SiglipTextConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
    ) -> torch.Tensor: ...

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

class SiglipAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = ..., **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class SiglipMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class SiglipEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: SiglipVisionConfig | SiglipTextConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, output_attentions: bool | None = ...
    ) -> tuple[torch.FloatTensor]: ...

class SiglipPreTrainedModel(PreTrainedModel):
    config: SiglipConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _supports_attention_backend = ...

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipConfig) -> None: ...
    @can_return_tuple
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> BaseModelOutput: ...

class SiglipTextTransformer(nn.Module):
    def __init__(self, config: SiglipTextConfig) -> None: ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> BaseModelOutputWithPooling: ...

class SiglipTextModel(SiglipPreTrainedModel):
    config: SiglipTextConfig
    def __init__(self, config: SiglipTextConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> BaseModelOutputWithPooling: ...

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None: ...
    @can_return_tuple
    def forward(
        self,
        pixel_values,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool | None = ...,
    ) -> BaseModelOutputWithPooling: ...

class SiglipMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None: ...
    def forward(self, hidden_state):  # -> Any:
        ...

class SiglipVisionModel(SiglipPreTrainedModel):
    config: SiglipVisionConfig
    main_input_name = ...
    def __init__(self, config: SiglipVisionConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @can_return_tuple
    def forward(
        self,
        pixel_values,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> BaseModelOutputWithPooling: ...

class SiglipModel(SiglipPreTrainedModel):
    config: SiglipConfig
    def __init__(self, config: SiglipConfig) -> None: ...
    def get_text_features(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
    ) -> torch.FloatTensor: ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> torch.FloatTensor: ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        return_loss: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> SiglipOutput: ...

class SiglipForImageClassification(SiglipPreTrainedModel):
    main_input_name = ...
    def __init__(self, config: SiglipConfig) -> None: ...
    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> ImageClassifierOutput: ...

__all__ = [
    "SiglipForImageClassification",
    "SiglipModel",
    "SiglipPreTrainedModel",
    "SiglipTextModel",
    "SiglipVisionModel",
]
