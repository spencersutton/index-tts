from typing import Any

import torch
from torch import nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import is_flash_attn_available
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import is_torch_flex_attn_available
from .configuration_dbrx import DbrxConfig

"""PyTorch DBRX model."""
if is_torch_flex_attn_available(): ...
if is_flash_attn_available(): ...
logger = ...

class DbrxRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=..., base=..., device=...) -> None: ...
    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=...):  # -> tuple[Tensor, Tensor]:
        ...

def rotate_half(x):  # -> Tensor:

    ...
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=..., unsqueeze_dim=...):  # -> tuple[Any, Any]:

    ...
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...
def load_balancing_loss_func(
    gate_probabilities: torch.Tensor, num_experts: int, top_k: int, attention_mask: torch.Tensor | None
) -> torch.Tensor: ...

class DbrxAttention(nn.Module):
    def __init__(self, config: DbrxConfig, block_idx: int | None = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_value: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]: ...

class DbrxFlashAttention2(DbrxAttention):
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_value: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class DbrxSdpaAttention(DbrxAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_value: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

DBRX_ATTENTION_CLASSES = ...

class DbrxNormAttentionNorm(nn.Module):
    def __init__(self, config: DbrxConfig, block_idx: int | None = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_value: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, Cache | None]: ...

class DbrxRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        moe_num_experts: int,
        moe_top_k: int,
        moe_jitter_eps: float | None,
        moe_normalize_expert_weights: float | None,
    ) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]: ...

class DbrxExpertGLU(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int, moe_num_experts: int, ffn_act_fn: dict) -> None: ...
    def forward(
        self, x: torch.Tensor, expert_w1: torch.Tensor, expert_v1: torch.Tensor, expert_w2: torch.Tensor
    ) -> torch.Tensor: ...

class DbrxExperts(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int, moe_num_experts: int, ffn_act_fn: dict) -> None: ...
    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, top_weights: torch.Tensor, top_experts: torch.LongTensor
    ) -> torch.Tensor: ...

class DbrxFFN(nn.Module):
    def __init__(self, config: DbrxConfig) -> None: ...
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

class DbrxBlock(GradientCheckpointingLayer):
    def __init__(self, config: DbrxConfig, block_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_value: Cache | None = ...,
        output_attentions: bool | None = ...,
        output_router_logits: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Any,
    ) -> (
        tuple[torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, Cache | None]
        | tuple[torch.Tensor, torch.Tensor | None, Cache | None]
        | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
        | tuple[torch.Tensor, Cache | None, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, Cache | None, torch.Tensor | None]
    ): ...

class DbrxPreTrainedModel(PreTrainedModel):
    config: DbrxConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...

class DbrxModel(DbrxPreTrainedModel):
    def __init__(self, config: DbrxConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Embedding: ...
    def set_input_embeddings(self, value: nn.Embedding):  # -> None:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_router_logits: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple | MoeModelOutputWithPast: ...

class DbrxForCausalLM(DbrxPreTrainedModel, GenerationMixin):
    def __init__(self, config: DbrxConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Embedding: ...
    def set_input_embeddings(self, value: nn.Embedding):  # -> None:
        ...
    def get_output_embeddings(self) -> nn.Linear: ...
    def set_output_embeddings(self, new_embeddings: nn.Linear):  # -> None:
        ...
    def set_decoder(self, decoder: DbrxModel):  # -> None:
        ...
    def get_decoder(self) -> DbrxModel: ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_router_logits: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ) -> tuple | MoeCausalLMOutputWithPast: ...

__all__ = ["DbrxForCausalLM", "DbrxModel", "DbrxPreTrainedModel"]
