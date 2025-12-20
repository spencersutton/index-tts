import torch
from torch import nn

from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import is_torch_flex_attn_available
from ..bart.modeling_bart import BartAttention
from ..time_series_transformer.modeling_time_series_transformer import (
    TimeSeriesFeatureEmbedder,
    TimeSeriesMeanScaler,
    TimeSeriesNOPScaler,
    TimeSeriesSinusoidalPositionalEmbedding,
    TimeSeriesStdScaler,
    TimeSeriesTransformerDecoder,
    TimeSeriesTransformerDecoderLayer,
    TimeSeriesTransformerEncoder,
    TimeSeriesTransformerEncoderLayer,
    TimeSeriesTransformerForPrediction,
    TimeSeriesTransformerModel,
    TimeSeriesValueEmbedding,
)
from .configuration_informer import InformerConfig

"""PyTorch Informer model."""
if is_torch_flex_attn_available(): ...

def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor: ...

class InformerFeatureEmbedder(TimeSeriesFeatureEmbedder): ...
class InformerStdScaler(TimeSeriesStdScaler): ...
class InformerMeanScaler(TimeSeriesMeanScaler): ...
class InformerNOPScaler(TimeSeriesNOPScaler): ...
class InformerSinusoidalPositionalEmbedding(TimeSeriesSinusoidalPositionalEmbedding): ...
class InformerValueEmbedding(TimeSeriesValueEmbedding): ...

class InformerPreTrainedModel(PreTrainedModel):
    config: InformerConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...

class InformerAttention(BartAttention): ...

class InformerProbSparseAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        is_decoder: bool = ...,
        sampling_factor: int = ...,
        bias: bool = ...,
        layer_idx: int | None = ...,
    ) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = ...,
        past_key_value: tuple[torch.Tensor] | None = ...,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class InformerConvLayer(GradientCheckpointingLayer):
    def __init__(self, c_in) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class InformerEncoderLayer(TimeSeriesTransformerEncoderLayer):
    def __init__(self, config: InformerConfig) -> None: ...

class InformerDecoderLayer(TimeSeriesTransformerDecoderLayer):
    def __init__(self, config: InformerConfig, layer_idx: int | None = ...) -> None: ...

class InformerEncoder(TimeSeriesTransformerEncoder):
    def __init__(self, config: InformerConfig) -> None: ...
    def forward(
        self,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...

class InformerDecoder(TimeSeriesTransformerDecoder):
    def __init__(self, config: InformerConfig) -> None: ...

class InformerModel(TimeSeriesTransformerModel, nn.Module):
    def __init__(self, config: InformerConfig) -> None: ...
    def forward(self, **super_kwargs):  # -> None:

        ...

class InformerForPrediction(TimeSeriesTransformerForPrediction, nn.Module):
    def __init__(self, config: InformerConfig) -> None: ...
    def forward(self, **super_kwargs):  # -> None:

        ...

__all__ = ["InformerForPrediction", "InformerModel", "InformerPreTrainedModel"]
