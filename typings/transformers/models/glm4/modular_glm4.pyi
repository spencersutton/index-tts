import torch

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ..glm.modeling_glm import GlmAttention, GlmForCausalLM, GlmForSequenceClassification, GlmForTokenClassification
from ..phi3.modeling_phi3 import Phi3MLP
from .configuration_glm4 import Glm4Config

logger = ...
_CHECKPOINT_FOR_DOC = ...

class Glm4MLP(Phi3MLP): ...

class Glm4DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Glm4Config, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_value: Cache | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]: ...

class Glm4Attention(GlmAttention): ...

class Glm4ForCausalLM(GlmForCausalLM):
    def forward(self, **super_kwargs: Unpack[TransformersKwargs]) -> tuple | CausalLMOutputWithPast: ...

class Glm4ForSequenceClassification(GlmForSequenceClassification): ...
class Glm4ForTokenClassification(GlmForTokenClassification): ...

__all__ = [
    "Glm4ForCausalLM",
    "Glm4ForSequenceClassification",
    "Glm4ForTokenClassification",
    "Glm4Model",
    "Glm4PreTrainedModel",
]
