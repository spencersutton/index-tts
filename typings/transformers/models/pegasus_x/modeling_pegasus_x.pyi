from dataclasses import dataclass

import torch
from torch import nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import is_torch_flex_attn_available
from .configuration_pegasus_x import PegasusXConfig

"""PyTorch PEGASUS-X model."""
if is_torch_flex_attn_available(): ...
logger = ...

@dataclass
class DimensionInfo:
    batch_size: int
    seq_len: int
    block_size: int
    num_heads: int
    hidden_dim: int
    dim_per_head: int
    num_blocks: int
    global_len: int
    padded_seq_len: int

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):  # -> Tensor:

    ...

class PegasusXScaledWordEmbedding(nn.Embedding):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float | None = ...
    ) -> None: ...
    def forward(self, input_ids: torch.Tensor):  # -> Tensor:
        ...

class PegasusXSinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_scale: int = ...) -> None: ...
    @torch.no_grad()
    def forward(
        self, input_embeds: torch.Tensor, past_key_values_length: int = ..., position_ids: torch.Tensor | None = ...
    ) -> torch.Tensor: ...

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = ...,
    dropout: float = ...,
    head_mask: torch.Tensor | None = ...,
    **kwargs,
):  # -> tuple[Tensor, Tensor]:
    ...

class PegasusXAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        bias: bool = ...,
        is_causal: bool = ...,
        config: PegasusXConfig | None = ...,
        layer_idx: int | None = ...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = ...,
        past_key_value: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class PegasusXGlobalLocalAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, block_size: int, dropout: float = ..., is_decoder: bool = ...
    ) -> None: ...
    def forward(
        self,
        token_hidden_states: torch.Tensor,
        global_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...
    def compute_global_attention_representations(
        self, global_q, global_k, global_v, local_k, local_v, mask, dim: DimensionInfo
    ):  # -> tuple[Tensor, Tensor]:

        ...
    def compute_local_attention_representations(
        self, global_k, global_v, local_q, local_k, local_v, mask, dim: DimensionInfo
    ):  # -> tuple[Tensor, Tensor]:

        ...

class PegasusXEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, stagger_blocks_this_layer: bool, config: PegasusXConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        global_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = ...,
    ) -> torch.Tensor: ...
    @classmethod
    def pad_local_tokens(cls, hidden_states, attention_mask, block_size):  # -> tuple[Any, Any]:
        ...
    @classmethod
    def unpad_local_tokens(cls, padded_hidden_states, block_size): ...

class PegasusXDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: PegasusXConfig, layer_idx: int | None = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.Tensor | None = ...,
        past_key_value: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> torch.Tensor: ...

class PegasusXPreTrainedModel(PreTrainedModel):
    config: PegasusXConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_flex_attn = ...
    _can_compile_fullgraph = ...

class PegasusXEncoder(PegasusXPreTrainedModel):
    def __init__(self, config: PegasusXConfig, embed_tokens: nn.Embedding | None = ...) -> None: ...
    def resize_position_embeddings(self, new_num_position_embeddings: int):  # -> None:

        ...
    def get_position_embeddings(self) -> nn.Embedding: ...
    def forward(
        self,
        input_ids=...,
        attention_mask=...,
        inputs_embeds=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
    ):  # -> tuple[Any | tuple[tuple[Any, Any], ...] | tuple[Any | Tensor, ...] | tuple[()] | tuple[Any | None, ...], ...] | BaseModelOutput:

        ...

class PegasusXDecoder(PegasusXPreTrainedModel):
    def __init__(self, config: PegasusXConfig, embed_tokens: nn.Embedding | None = ...) -> None: ...
    def forward(
        self,
        input_ids=...,
        attention_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        past_key_values=...,
        inputs_embeds=...,
        use_cache=...,
        output_attentions=...,
        output_hidden_states=...,
        return_dict=...,
        cache_position=...,
    ): ...

class PegasusXModel(PegasusXPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: PegasusXConfig) -> None: ...
    def get_input_embeddings(self):  # -> PegasusXScaledWordEmbedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_encoder(self):  # -> PegasusXEncoder:
        ...
    def get_decoder(self):  # -> PegasusXDecoder:
        ...
    def resize_position_embeddings(self, new_num_position_embeddings: int):  # -> None:

        ...
    def get_position_embeddings(self) -> tuple[nn.Embedding]: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.Tensor | None = ...,
        decoder_attention_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[torch.FloatTensor] | None = ...,
        past_key_values: tuple[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        decoder_inputs_embeds: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | Seq2SeqModelOutput: ...

class PegasusXForConditionalGeneration(PegasusXPreTrainedModel, GenerationMixin):
    base_model_prefix = ...
    _tied_weights_keys = ...
    def __init__(self, config: PegasusXConfig) -> None: ...
    def get_encoder(self):  # -> PegasusXEncoder:
        ...
    def get_decoder(self):  # -> PegasusXDecoder:
        ...
    def resize_position_embeddings(self, new_num_position_embeddings: int):  # -> None:

        ...
    def get_position_embeddings(self) -> tuple[nn.Embedding]: ...
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        decoder_input_ids: torch.Tensor | None = ...,
        decoder_attention_mask: torch.Tensor | None = ...,
        encoder_outputs: tuple[torch.FloatTensor] | None = ...,
        past_key_values: tuple[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        decoder_inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | Seq2SeqLMOutput: ...
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):  # -> Tensor:
        ...

class PegasusXDecoderWrapper(PegasusXPreTrainedModel):
    def __init__(self, config) -> None: ...
    def forward(self, *args, **kwargs):  # -> Any:
        ...

__all__ = ["PegasusXForConditionalGeneration", "PegasusXModel", "PegasusXPreTrainedModel"]
