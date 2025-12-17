from typing import Any, TypedDict

import torch
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, MoeCausalLMOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import can_return_tuple, is_torch_flex_attn_available
from ...utils.import_utils import is_causal_conv1d_available, is_mamba_2_ssm_available
from .configuration_granitemoehybrid import GraniteMoeHybridConfig

if is_mamba_2_ssm_available(): ...
else:
    selective_state_update = ...
if is_causal_conv1d_available(): ...
if is_torch_flex_attn_available(): ...
logger = ...

def rotate_half(x):  # -> Tensor:

    ...
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=..., unsqueeze_dim=...):  # -> tuple[Any, Any]:

    ...
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...
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

class GraniteMoeHybridAttention(nn.Module):
    def __init__(self, config: GraniteMoeHybridConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_value: Cache | None = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class HybridMambaAttentionDynamicCache:
    key_cache = ...
    value_cache = ...
    is_compileable = ...
    def __init__(self, config: GraniteMoeHybridConfig, batch_size, dtype=..., device=...) -> None: ...
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def reorder_cache(self, beam_idx: torch.LongTensor):  # -> None:

        ...
    def get_seq_length(self, layer_idx: int | None = ...) -> int: ...
    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor], tuple[torch.Tensor]]: ...
    @classmethod
    def from_legacy_cache(cls, past_key_values: tuple[tuple[torch.FloatTensor]] | None = ...) -> DynamicCache: ...

def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int): ...
def reshape_into_chunks(input_tensor, pad_size, chunk_size): ...
def segment_sum(input_tensor):  # -> Tensor:

    ...

is_fast_path_available = ...

def apply_mask_to_padding_states(hidden_states, attention_mask): ...

class GraniteMoeHybridMambaLayer(nn.Module):
    def __init__(self, config: GraniteMoeHybridConfig, layer_idx: int) -> None: ...
    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: HybridMambaAttentionDynamicCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        seq_idx: torch.IntTensor | None = ...,
    ):  # -> Any:
        ...
    def torch_forward(
        self,
        input_states,
        cache_params: HybridMambaAttentionDynamicCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
    ):  # -> Any:
        ...
    def forward(
        self,
        hidden_states,
        cache_params: HybridMambaAttentionDynamicCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        seq_idx: torch.IntTensor | None = ...,
        **kwargs,
    ):  # -> Any:
        ...

class GraniteMoeHybridRMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states, gate=...): ...

class GraniteMoeHybridMLP(nn.Module):
    def __init__(self, config: GraniteMoeHybridConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class GraniteFlashAttentionKwargs(TypedDict, total=False):
    cu_seq_lens_q: torch.LongTensor
    cu_seq_lens_k: torch.LongTensor
    max_length_q: int
    max_length_k: int
    seq_idx: torch.IntTensor

class GraniteMoeHybridRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class GraniteMoeHybridParallelExperts(nn.Module):
    def __init__(self, num_experts: int, input_size: int, output_size: int) -> None: ...
    def forward(self, inputs, expert_size):  # -> Tensor:

        ...

class GraniteMoeHybridTopKGating(nn.Module):
    def __init__(self, input_size: int, num_experts: int, top_k: int) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, Any, Tensor, list[Any], Any]:
        ...

class GraniteMoeHybridMoE(nn.Module):
    def __init__(self, config: GraniteMoeHybridConfig) -> None: ...
    def forward(self, layer_input):  # -> tuple[Tensor, Any]:

        ...

class GraniteMoeHybridDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GraniteMoeHybridConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        past_key_value: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        output_router_logits: bool | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[GraniteFlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]: ...

class GraniteMoeHybridPreTrainedModel(PreTrainedModel):
    config: GraniteMoeHybridConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _can_compile_fullgraph = ...
    _is_stateful = ...

class GraniteMoeHybridRotaryEmbedding(nn.Module):
    def __init__(self, config: GraniteMoeHybridConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

class GraniteMoeHybridModel(GraniteMoeHybridPreTrainedModel):
    def __init__(self, config: GraniteMoeHybridConfig) -> None: ...
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_router_logits: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[GraniteFlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast: ...

def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = ...,
    top_k=...,
    attention_mask: torch.Tensor | None = ...,
) -> torch.Tensor | int: ...

class GraniteMoeHybridForCausalLM(GraniteMoeHybridPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config: GraniteMoeHybridConfig) -> None: ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> GraniteMoeHybridModel:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
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
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        cache_position=...,
        position_ids=...,
        use_cache=...,
        **kwargs,
    ):  # -> dict[str, Any]:
        ...

__all__ = ["GraniteMoeHybridForCausalLM", "GraniteMoeHybridModel", "GraniteMoeHybridPreTrainedModel"]
