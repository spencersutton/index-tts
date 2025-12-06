
import torch

__all__ = ["Emformer"]

class _EmformerAttention(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dropout: float = ...,
        weight_init_gain: float | None = ...,
        tanh_on_mem: bool = ...,
        negative_inf: float = ...,
    ) -> None: ...
    def forward(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        summary: torch.Tensor,
        mems: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @torch.jit.export
    def infer(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        summary: torch.Tensor,
        mems: torch.Tensor,
        left_context_key: torch.Tensor,
        left_context_val: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

class _EmformerLayer(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        segment_length: int,
        dropout: float = ...,
        activation: str = ...,
        left_context_length: int = ...,
        max_memory_size: int = ...,
        weight_init_gain: float | None = ...,
        tanh_on_mem: bool = ...,
        negative_inf: float = ...,
    ) -> None: ...
    def forward(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        mems: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    @torch.jit.export
    def infer(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        state: list[torch.Tensor] | None,
        mems: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]: ...

class _EmformerImpl(torch.nn.Module):
    def __init__(
        self,
        emformer_layers: torch.nn.ModuleList,
        segment_length: int,
        left_context_length: int = ...,
        right_context_length: int = ...,
        max_memory_size: int = ...,
    ) -> None: ...
    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    @torch.jit.export
    def infer(
        self, input: torch.Tensor, lengths: torch.Tensor, states: list[list[torch.Tensor]] | None = ...
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[torch.Tensor]]]: ...

class Emformer(_EmformerImpl):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        segment_length: int,
        dropout: float = ...,
        activation: str = ...,
        left_context_length: int = ...,
        right_context_length: int = ...,
        max_memory_size: int = ...,
        weight_init_scale_strategy: str | None = ...,
        tanh_on_mem: bool = ...,
        negative_inf: float = ...,
    ) -> None: ...
