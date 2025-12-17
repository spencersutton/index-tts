import torch
from torch import nn

from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_squeezebert import SqueezeBertConfig

"""PyTorch SqueezeBert model."""
logger = ...

class SqueezeBertEmbeddings(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=...):  # -> Any:
        ...

class MatMulWrapper(nn.Module):
    def __init__(self) -> None: ...
    def forward(self, mat1, mat2):  # -> Tensor:

        ...

class SqueezeBertLayerNorm(nn.LayerNorm):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, x):  # -> Tensor:
        ...

class ConvDropoutLayerNorm(nn.Module):
    def __init__(self, cin, cout, groups, dropout_prob) -> None: ...
    def forward(self, hidden_states, input_tensor):  # -> Any:
        ...

class ConvActivation(nn.Module):
    def __init__(self, cin, cout, groups, act) -> None: ...
    def forward(self, x): ...

class SqueezeBertSelfAttention(nn.Module):
    def __init__(self, config, cin, q_groups=..., k_groups=..., v_groups=...) -> None: ...
    def transpose_for_scores(self, x): ...
    def transpose_key_for_scores(self, x): ...
    def transpose_output(self, x): ...
    def forward(self, hidden_states, attention_mask, output_attentions):  # -> dict[str, Any]:

        ...

class SqueezeBertModule(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states, attention_mask, output_attentions):  # -> dict[str, Any]:
        ...

class SqueezeBertEncoder(nn.Module):
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

class SqueezeBertPooler(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class SqueezeBertPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class SqueezeBertLMPredictionHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class SqueezeBertOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, sequence_output):  # -> Any:
        ...

class SqueezeBertPreTrainedModel(PreTrainedModel):
    config: SqueezeBertConfig
    base_model_prefix = ...

class SqueezeBertModel(SqueezeBertPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPooling: ...

class SqueezeBertForMaskedLM(SqueezeBertPreTrainedModel):
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
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | MaskedLMOutput: ...

class SqueezeBertForSequenceClassification(SqueezeBertPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SequenceClassifierOutput: ...

class SqueezeBertForMultipleChoice(SqueezeBertPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | MultipleChoiceModelOutput: ...

class SqueezeBertForTokenClassification(SqueezeBertPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | TokenClassifierOutput: ...

class SqueezeBertForQuestionAnswering(SqueezeBertPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        start_positions: torch.Tensor | None = ...,
        end_positions: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | QuestionAnsweringModelOutput: ...

__all__ = [
    "SqueezeBertForMaskedLM",
    "SqueezeBertForMultipleChoice",
    "SqueezeBertForQuestionAnswering",
    "SqueezeBertForSequenceClassification",
    "SqueezeBertForTokenClassification",
    "SqueezeBertModel",
    "SqueezeBertModule",
    "SqueezeBertPreTrainedModel",
]
