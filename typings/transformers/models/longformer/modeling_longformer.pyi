from dataclasses import dataclass

import torch
from torch import nn

from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput
from .configuration_longformer import LongformerConfig

"""PyTorch Longformer model."""
logger = ...

@dataclass
class LongformerBaseModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    global_attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
class LongformerBaseModelOutputWithPooling(ModelOutput):
    last_hidden_state: torch.FloatTensor
    pooler_output: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    global_attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
class LongformerMaskedLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    global_attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
class LongformerQuestionAnsweringModelOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    start_logits: torch.FloatTensor | None = ...
    end_logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    global_attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
class LongformerSequenceClassifierOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    global_attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
class LongformerMultipleChoiceModelOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    global_attentions: tuple[torch.FloatTensor, ...] | None = ...

@dataclass
class LongformerTokenClassifierOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor, ...] | None = ...
    attentions: tuple[torch.FloatTensor, ...] | None = ...
    global_attentions: tuple[torch.FloatTensor, ...] | None = ...

def create_position_ids_from_input_ids(input_ids, padding_idx): ...

class LongformerEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=...):  # -> Any:
        ...
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):  # -> Tensor:

        ...

class LongformerSelfAttention(nn.Module):
    def __init__(self, config, layer_id) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        layer_head_mask=...,
        is_index_masked=...,
        is_index_global_attn=...,
        is_global_attn=...,
        output_attentions=...,
    ):  # -> tuple[Tensor, ...] | tuple[Tensor, Tensor] | tuple[Tensor]:

        ...

class LongformerSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor: ...

class LongformerAttention(nn.Module):
    def __init__(self, config, layer_id=...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        layer_head_mask=...,
        is_index_masked=...,
        is_index_global_attn=...,
        is_global_attn=...,
        output_attentions=...,
    ):  # -> Any:
        ...

class LongformerIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class LongformerOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor: ...

class LongformerLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        layer_head_mask=...,
        is_index_masked=...,
        is_index_global_attn=...,
        is_global_attn=...,
        output_attentions=...,
    ):  # -> Any:
        ...
    def ff_chunk(self, attn_output):  # -> Any:
        ...

class LongformerEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        head_mask=...,
        padding_len=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | LongformerBaseModelOutput:
        ...

class LongformerPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class LongformerLMHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, features, **kwargs):  # -> Any:
        ...

class LongformerPreTrainedModel(PreTrainedModel):
    config: LongformerConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

class LongformerModel(LongformerPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        global_attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | LongformerBaseModelOutputWithPooling: ...

class LongformerForMaskedLM(LongformerPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        global_attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | LongformerMaskedLMOutput: ...

class LongformerForSequenceClassification(LongformerPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        global_attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | LongformerSequenceClassifierOutput: ...

class LongformerClassificationHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, **kwargs):  # -> Any:
        ...

class LongformerForQuestionAnswering(LongformerPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        global_attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        start_positions: torch.Tensor | None = ...,
        end_positions: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | LongformerQuestionAnsweringModelOutput: ...

class LongformerForTokenClassification(LongformerPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        global_attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | LongformerTokenClassifierOutput: ...

class LongformerForMultipleChoice(LongformerPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        global_attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | LongformerMultipleChoiceModelOutput: ...

__all__ = [
    "LongformerForMaskedLM",
    "LongformerForMultipleChoice",
    "LongformerForQuestionAnswering",
    "LongformerForSequenceClassification",
    "LongformerForTokenClassification",
    "LongformerModel",
    "LongformerPreTrainedModel",
    "LongformerSelfAttention",
]
