from dataclasses import dataclass

import torch
from torch import nn

from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_vilt import ViltConfig

"""PyTorch ViLT model."""
logger = ...

@dataclass
class ViltForImagesAndTextClassificationOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: list[tuple[torch.FloatTensor]] | None = ...
    attentions: list[tuple[torch.FloatTensor]] | None = ...

class ViltEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def visual_embed(
        self, pixel_values, pixel_mask, max_image_length=...
    ):  # -> tuple[Any, Tensor, tuple[Tensor, tuple[Any, Any]]]:
        ...
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        pixel_values,
        pixel_mask,
        inputs_embeds,
        image_embeds,
        image_token_type_idx=...,
    ):  # -> tuple[Tensor, Tensor]:
        ...

class TextEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=...):  # -> Any:
        ...

class ViltPatchEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, pixel_values):  # -> Any:
        ...

class ViltSelfAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states, attention_mask=..., head_mask=..., output_attentions=...
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...

class ViltSelfOutput(nn.Module):
    def __init__(self, config: ViltConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor: ...

class ViltAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(self, hidden_states, attention_mask=..., head_mask=..., output_attentions=...):  # -> Any:
        ...

class ViltIntermediate(nn.Module):
    def __init__(self, config: ViltConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class ViltOutput(nn.Module):
    def __init__(self, config: ViltConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor: ...

class ViltLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, attention_mask=..., head_mask=..., output_attentions=...):  # -> Any:
        ...

class ViltEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | BaseModelOutput:
        ...

class ViltPreTrainedModel(PreTrainedModel):
    config: ViltConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

class ViltModel(ViltPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_mask: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        image_embeds: torch.FloatTensor | None = ...,
        image_token_type_idx: int | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BaseModelOutputWithPooling | tuple[torch.FloatTensor]: ...

class ViltPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class ViltForMaskedLM(ViltPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_mask: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        image_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> MaskedLMOutput | tuple[torch.FloatTensor]: ...

class ViltPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class ViltMLMHead(nn.Module):
    def __init__(self, config, weight=...) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class ViltForQuestionAnswering(ViltPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_mask: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        image_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> SequenceClassifierOutput | tuple[torch.FloatTensor]: ...

class ViltForImageAndTextRetrieval(ViltPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_mask: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        image_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> SequenceClassifierOutput | tuple[torch.FloatTensor]: ...

class ViltForImagesAndTextClassification(ViltPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_mask: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        image_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> ViltForImagesAndTextClassificationOutput | tuple[torch.FloatTensor]: ...

class ViltForTokenClassification(ViltPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        pixel_mask: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        image_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> TokenClassifierOutput | tuple[torch.FloatTensor]: ...

__all__ = [
    "ViltForImageAndTextRetrieval",
    "ViltForImagesAndTextClassification",
    "ViltForMaskedLM",
    "ViltForQuestionAnswering",
    "ViltForTokenClassification",
    "ViltLayer",
    "ViltModel",
    "ViltPreTrainedModel",
]
