import torch
from torch import nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_xmod import XmodConfig

"""PyTorch X-MOD model."""
logger = ...

class XmodEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=..., past_key_values_length=...
    ):  # -> Any:
        ...
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):  # -> Tensor:

        ...

class XmodSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=..., layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        past_key_value: Cache | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor]: ...

class XmodSelfOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor: ...

class XmodAttention(nn.Module):
    def __init__(self, config, position_embedding_type=..., layer_idx=...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        past_key_value: Cache | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor]: ...

class XmodIntermediate(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class XmodAdapter(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class XmodOutput(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, lang_ids: torch.Tensor
    ) -> torch.Tensor: ...
    def lang_adapter(self, lang_ids: torch.Tensor, hidden_states: torch.Tensor):  # -> Tensor:
        ...

class XmodLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        lang_ids: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        past_key_value: Cache | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor]: ...
    def feed_forward_chunk(self, attention_output):  # -> Any:
        ...

class XmodEncoder(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        lang_ids: torch.Tensor,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        past_key_values: tuple[tuple[torch.FloatTensor]] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPastAndCrossAttentions: ...

class XmodPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class XmodPreTrainedModel(PreTrainedModel):
    config: XmodConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    def set_default_language(self, language: str):  # -> None:

        ...
    def freeze_embeddings_and_language_adapters(self):  # -> None:

        ...

class XmodModel(XmodPreTrainedModel):
    def __init__(self, config, add_pooling_layer=...) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        lang_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        past_key_values: list[torch.FloatTensor] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPoolingAndCrossAttentions: ...

class XmodForCausalLM(XmodPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        lang_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        past_key_values: tuple[tuple[torch.FloatTensor]] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor] | CausalLMOutputWithCrossAttentions: ...

class XmodForMaskedLM(XmodPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        lang_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.FloatTensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | MaskedLMOutput: ...

class XmodLMHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, features, **kwargs):  # -> Any:
        ...

class XmodForSequenceClassification(XmodPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        lang_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput: ...

class XmodForMultipleChoice(XmodPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        lang_ids: torch.LongTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | MultipleChoiceModelOutput: ...

class XmodForTokenClassification(XmodPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        lang_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | TokenClassifierOutput: ...

class XmodClassificationHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, features, **kwargs):  # -> Any:
        ...

class XmodForQuestionAnswering(XmodPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        lang_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | QuestionAnsweringModelOutput: ...

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=...): ...

__all__ = [
    "XmodForCausalLM",
    "XmodForMaskedLM",
    "XmodForMultipleChoice",
    "XmodForQuestionAnswering",
    "XmodForSequenceClassification",
    "XmodForTokenClassification",
    "XmodModel",
    "XmodPreTrainedModel",
]
