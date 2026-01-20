from collections.abc import Callable
from dataclasses import dataclass

import torch
from _typeshed import Incomplete
from torch import nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.utils import ModelOutput, auto_docstring

from indextts.util import patch_call

__all__ = [
    "GPT2DoubleHeadsModel",
    "GPT2ForQuestionAnswering",
    "GPT2ForSequenceClassification",
    "GPT2ForTokenClassification",
    "GPT2LMHeadModel",
    "GPT2Model",
    "GPT2PreTrainedModel",
    "load_tf_weights_in_gpt2",
]

def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path): ...

class GPT2Attention(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    split_size: Incomplete
    scale_attn_weights: Incomplete
    is_cross_attention: Incomplete
    scale_attn_by_inverse_layer_idx: Incomplete
    layer_idx: Incomplete
    reorder_and_upcast_attn: Incomplete
    c_attn: Incomplete
    q_attn: Incomplete
    c_proj: Incomplete
    attn_dropout: Incomplete
    resid_dropout: Incomplete
    is_causal: bool
    pruned_heads: Incomplete
    def __init__(self, config, is_cross_attention: bool = False, layer_idx=None) -> None: ...
    def prune_heads(self, heads) -> None: ...
    def forward(
        self,
        hidden_states: tuple[torch.Tensor] | None,
        past_key_value: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], ...]: ...

class GPT2MLP(nn.Module):
    c_fc: Incomplete
    c_proj: Incomplete
    act: Incomplete
    dropout: Incomplete
    def __init__(self, intermediate_size, config) -> None: ...
    def forward(self, hidden_states: tuple[torch.Tensor] | None) -> torch.Tensor: ...

class GPT2Block(nn.Module):
    ln_1: Incomplete
    attn: Incomplete
    ln_2: Incomplete
    crossattention: Incomplete
    ln_cross_attn: Incomplete
    mlp: Incomplete
    def __init__(self, config, layer_idx=None) -> None: ...
    def forward(
        self,
        hidden_states: tuple[torch.Tensor] | None,
        past_key_value: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None: ...

class GPT2SequenceSummary(nn.Module):
    summary_type: Incomplete
    summary: Incomplete
    activation: Callable
    first_dropout: Incomplete
    last_dropout: Incomplete
    def __init__(self, config: GPT2Config) -> None: ...
    def forward(self, hidden_states: torch.Tensor, cls_index: torch.Tensor | None = None) -> torch.Tensor: ...

class GPT2PreTrainedModel(PreTrainedModel):
    config_class = GPT2Config
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix: str
    is_parallelizable: bool
    supports_gradient_checkpointing: bool
    def __init__(self, *inputs, **kwargs) -> None: ...

@dataclass
class GPT2DoubleHeadsModelOutput(ModelOutput):
    loss: torch.Tensor | None = ...
    mc_loss: torch.Tensor | None = ...
    logits: torch.Tensor | None = ...
    mc_logits: torch.Tensor | None = ...
    past_key_values: tuple[tuple[torch.Tensor]] | None = ...
    hidden_states: tuple[torch.Tensor] | None = ...
    attentions: tuple[torch.Tensor] | None = ...

class GPT2Model(GPT2PreTrainedModel):
    embed_dim: Incomplete
    wte: Incomplete
    wpe: Incomplete
    drop: Incomplete
    h: Incomplete
    ln_f: Incomplete
    model_parallel: bool
    device_map: Incomplete
    gradient_checkpointing: bool
    def __init__(self, config) -> None: ...
    first_device: Incomplete
    last_device: Incomplete
    def parallelize(self, device_map=None) -> None: ...
    def deparallelize(self) -> None: ...
    def get_input_embeddings(self): ...
    def set_input_embeddings(self, new_embeddings) -> None: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | Cache | None = None,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...
    @patch_call(forward)
    def __call__(self) -> None: ...

class GPT2LMHeadModel(GPT2PreTrainedModel, GenerationMixin):
    transformer: Incomplete
    lm_head: Incomplete
    model_parallel: bool
    device_map: Incomplete
    def __init__(self, config) -> None: ...
    def parallelize(self, device_map=None) -> None: ...
    def deparallelize(self) -> None: ...
    def get_output_embeddings(self): ...
    def set_output_embeddings(self, new_embeddings) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | CausalLMOutputWithCrossAttentions: ...

class GPT2DoubleHeadsModel(GPT2PreTrainedModel, GenerationMixin):
    transformer: Incomplete
    lm_head: Incomplete
    multiple_choice_head: Incomplete
    model_parallel: bool
    device_map: Incomplete
    def __init__(self, config) -> None: ...
    def parallelize(self, device_map=None) -> None: ...
    def deparallelize(self) -> None: ...
    def get_output_embeddings(self): ...
    def set_output_embeddings(self, new_embeddings) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        mc_token_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        mc_labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | GPT2DoubleHeadsModelOutput: ...

class GPT2ForSequenceClassification(GPT2PreTrainedModel):
    num_labels: Incomplete
    transformer: Incomplete
    score: Incomplete
    model_parallel: bool
    device_map: Incomplete
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | SequenceClassifierOutputWithPast: ...

class GPT2ForTokenClassification(GPT2PreTrainedModel):
    num_labels: Incomplete
    transformer: Incomplete
    dropout: Incomplete
    classifier: Incomplete
    model_parallel: bool
    device_map: Incomplete
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | TokenClassifierOutput: ...

class GPT2ForQuestionAnswering(GPT2PreTrainedModel):
    num_labels: Incomplete
    transformer: Incomplete
    qa_outputs: Incomplete
    model_parallel: bool
    device_map: Incomplete
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        start_positions: torch.Tensor | None = None,
        end_positions: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | QuestionAnsweringModelOutput: ...
