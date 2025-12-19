from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput
from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig

"""PyTorch BLIP model."""
logger = ...

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor: ...
def blip_loss(similarity: torch.Tensor) -> torch.Tensor: ...

@dataclass
class BlipForConditionalGenerationModelOutput(ModelOutput):
    loss: tuple[torch.FloatTensor] | None = ...
    logits: tuple[torch.FloatTensor] | None = ...
    image_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    @property
    def decoder_logits(self):  # -> tuple[FloatTensor] | None:
        ...

@dataclass
class BlipTextVisionModelOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
class BlipImageTextMatchingModelOutput(ModelOutput):
    itm_score: torch.FloatTensor | None = ...
    loss: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    vision_pooler_output: torch.FloatTensor | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    question_embeds: tuple[torch.FloatTensor] | None = ...

@dataclass
class BlipOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits_per_image: torch.FloatTensor | None = ...
    logits_per_text: torch.FloatTensor | None = ...
    text_embeds: torch.FloatTensor | None = ...
    image_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> tuple[Any]: ...

class BlipVisionEmbeddings(nn.Module):
    def __init__(self, config: BlipVisionConfig) -> None: ...
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor: ...
    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding: bool = ...) -> torch.Tensor: ...

class BlipTextEmbeddings(nn.Module):
    def __init__(self, config: BlipTextConfig) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
    ) -> torch.Tensor: ...

class BlipAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class BlipMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class BlipEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: BlipConfig) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, output_attentions: bool | None = ...
    ) -> tuple[torch.FloatTensor]: ...

class BlipPreTrainedModel(PreTrainedModel):
    config: BlipConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...

class BlipEncoder(nn.Module):
    def __init__(self, config: BlipConfig) -> None: ...
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class BlipVisionModel(BlipPreTrainedModel):
    main_input_name = ...
    config: BlipVisionConfig
    def __init__(self, config: BlipVisionConfig) -> None: ...
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...
    def get_input_embeddings(self):  # -> BlipVisionEmbeddings:
        ...

class BlipModel(BlipPreTrainedModel):
    config: BlipConfig
    def __init__(self, config: BlipConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_text_features(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
    ) -> torch.FloatTensor: ...
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> torch.FloatTensor: ...
    def get_multimodal_features(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> torch.FloatTensor: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        return_loss: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> tuple | BlipOutput: ...

class BlipForConditionalGeneration(BlipPreTrainedModel, GenerationMixin):
    config: BlipConfig
    _tied_weights_keys = ...
    main_input_name = ...
    def __init__(self, config: BlipConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.LongTensor | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> tuple | BlipForConditionalGenerationModelOutput: ...
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        **generate_kwargs,
    ) -> torch.LongTensor: ...

class BlipForQuestionAnswering(BlipPreTrainedModel, GenerationMixin):
    config: BlipConfig
    _tied_weights_keys = ...
    def __init__(self, config: BlipConfig) -> None: ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        decoder_input_ids: torch.LongTensor | None = ...,
        decoder_attention_mask: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        labels: torch.LongTensor | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> tuple | BlipTextVisionModelOutput: ...
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = ...,
        interpolate_pos_encoding: bool = ...,
        **generate_kwargs,
    ) -> torch.LongTensor: ...

class BlipForImageTextRetrieval(BlipPreTrainedModel):
    config: BlipConfig
    def __init__(self, config: BlipConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        use_itm_head: bool | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        interpolate_pos_encoding: bool = ...,
    ) -> tuple | BlipTextVisionModelOutput: ...

__all__ = [
    "BlipForConditionalGeneration",
    "BlipForImageTextRetrieval",
    "BlipForQuestionAnswering",
    "BlipModel",
    "BlipPreTrainedModel",
    "BlipTextModel",
    "BlipVisionModel",
]
