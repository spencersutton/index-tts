from dataclasses import dataclass

import torch
from torch import nn

from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, ModelOutput, QuestionAnsweringModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import can_return_tuple
from .configuration_splinter import SplinterConfig

"""PyTorch Splinter model."""
logger = ...

class SplinterEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
    ) -> tuple: ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = ...,
    head_mask: torch.Tensor | None = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class SplinterSelfAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor]: ...

class SplinterSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor: ...

class SplinterAttention(nn.Module):
    def __init__(self, config) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor]: ...

class SplinterIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class SplinterOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor: ...

class SplinterLayer(GradientCheckpointingLayer):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor]: ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class SplinterEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    @can_return_tuple
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor] | BaseModelOutput: ...

class SplinterPreTrainedModel(PreTrainedModel):
    config: SplinterConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...

class SplinterModel(SplinterPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class SplinterFullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_act=...) -> None: ...
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: ...

class QuestionAwareSpanSelectionHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, inputs, positions):  # -> tuple[Tensor, Tensor]:
        ...

class SplinterForQuestionAnswering(SplinterPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        question_positions: torch.LongTensor | None = ...,
    ) -> tuple | QuestionAnsweringModelOutput: ...

@dataclass
class SplinterForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    start_logits: torch.FloatTensor | None = ...
    end_logits: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

class SplinterForPreTraining(SplinterPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        question_positions: torch.LongTensor | None = ...,
    ) -> tuple | SplinterForPreTrainingOutput: ...

__all__ = [
    "SplinterForPreTraining",
    "SplinterForQuestionAnswering",
    "SplinterLayer",
    "SplinterModel",
    "SplinterPreTrainedModel",
]
