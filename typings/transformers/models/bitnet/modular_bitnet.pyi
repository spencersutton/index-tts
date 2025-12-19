import torch

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import CausalLMOutputWithPast
from ...processing_utils import Unpack
from ..gemma.modeling_gemma import GemmaMLP
from ..llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM, LlamaModel, LlamaRMSNorm
from .configuration_bitnet import BitNetConfig

"""PyTorch BitNet model."""
logger = ...

class BitNetRMSNorm(LlamaRMSNorm): ...

class BitNetMLP(GemmaMLP):
    def __init__(self, config: BitNetConfig) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class BitNetAttention(LlamaAttention):
    def __init__(self, config: BitNetConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class BitNetDecoderLayer(LlamaDecoderLayer): ...
class BitNetModel(LlamaModel): ...

class BitNetForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ...
    _tp_plan = ...
    _pp_plan = ...
    def forward(self, **super_kwargs) -> CausalLMOutputWithPast: ...

__all__ = ["BitNetForCausalLM", "BitNetModel", "BitNetPreTrainedModel"]
