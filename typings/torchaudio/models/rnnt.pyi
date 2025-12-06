from abc import ABC, abstractmethod

import torch

__all__ = ["RNNT", "emformer_rnnt_base", "emformer_rnnt_model"]

class _TimeReduction(torch.nn.Module):
    def __init__(self, stride: int) -> None: ...
    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

class _CustomLSTM(torch.nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, layer_norm: bool = ..., layer_norm_epsilon: float = ...
    ) -> None: ...
    def forward(
        self, input: torch.Tensor, state: list[torch.Tensor] | None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]: ...

class _Transcriber(ABC):
    @abstractmethod
    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    @abstractmethod
    def infer(
        self, input: torch.Tensor, lengths: torch.Tensor, states: list[list[torch.Tensor]] | None
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[torch.Tensor]]]: ...

class _EmformerEncoder(torch.nn.Module, _Transcriber):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        segment_length: int,
        right_context_length: int,
        time_reduction_input_dim: int,
        time_reduction_stride: int,
        transformer_num_heads: int,
        transformer_ffn_dim: int,
        transformer_num_layers: int,
        transformer_left_context_length: int,
        transformer_dropout: float = ...,
        transformer_activation: str = ...,
        transformer_max_memory_size: int = ...,
        transformer_weight_init_scale_strategy: str = ...,
        transformer_tanh_on_mem: bool = ...,
    ) -> None: ...
    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    @torch.jit.export
    def infer(
        self, input: torch.Tensor, lengths: torch.Tensor, states: list[list[torch.Tensor]] | None
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[torch.Tensor]]]: ...

class _Predictor(torch.nn.Module):
    def __init__(
        self,
        num_symbols: int,
        output_dim: int,
        symbol_embedding_dim: int,
        num_lstm_layers: int,
        lstm_hidden_dim: int,
        lstm_layer_norm: bool = ...,
        lstm_layer_norm_epsilon: float = ...,
        lstm_dropout: float = ...,
    ) -> None: ...
    def forward(
        self, input: torch.Tensor, lengths: torch.Tensor, state: list[list[torch.Tensor]] | None = ...
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[torch.Tensor]]]: ...

class _Joiner(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: str = ...) -> None: ...
    def forward(
        self,
        source_encodings: torch.Tensor,
        source_lengths: torch.Tensor,
        target_encodings: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class RNNT(torch.nn.Module):
    def __init__(self, transcriber: _Transcriber, predictor: _Predictor, joiner: _Joiner) -> None: ...
    def forward(
        self,
        sources: torch.Tensor,
        source_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        predictor_state: list[list[torch.Tensor]] | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[torch.Tensor]]]: ...
    @torch.jit.export
    def transcribe_streaming(
        self, sources: torch.Tensor, source_lengths: torch.Tensor, state: list[list[torch.Tensor]] | None
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[torch.Tensor]]]: ...
    @torch.jit.export
    def transcribe(self, sources: torch.Tensor, source_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    @torch.jit.export
    def predict(
        self, targets: torch.Tensor, target_lengths: torch.Tensor, state: list[list[torch.Tensor]] | None
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[torch.Tensor]]]: ...
    @torch.jit.export
    def join(
        self,
        source_encodings: torch.Tensor,
        source_lengths: torch.Tensor,
        target_encodings: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

def emformer_rnnt_model(
    *,
    input_dim: int,
    encoding_dim: int,
    num_symbols: int,
    segment_length: int,
    right_context_length: int,
    time_reduction_input_dim: int,
    time_reduction_stride: int,
    transformer_num_heads: int,
    transformer_ffn_dim: int,
    transformer_num_layers: int,
    transformer_dropout: float,
    transformer_activation: str,
    transformer_left_context_length: int,
    transformer_max_memory_size: int,
    transformer_weight_init_scale_strategy: str,
    transformer_tanh_on_mem: bool,
    symbol_embedding_dim: int,
    num_lstm_layers: int,
    lstm_layer_norm: bool,
    lstm_layer_norm_epsilon: float,
    lstm_dropout: float,
) -> RNNT: ...
def emformer_rnnt_base(num_symbols: int) -> RNNT: ...
