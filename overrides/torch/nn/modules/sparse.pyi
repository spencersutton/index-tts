from _typeshed import Incomplete
from torch import Tensor

from indextts.util import patch_call

from .module import Module

__all__ = ["Embedding", "EmbeddingBag"]

class Embedding(Module):
    __constants__: Incomplete
    num_embeddings: int
    embedding_dim: int
    padding_idx: int | None
    max_norm: float | None
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    freeze: bool
    sparse: bool
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Tensor | None = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...
    @classmethod
    def from_pretrained(
        cls,
        embeddings,
        freeze: bool = True,
        padding_idx=None,
        max_norm=None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ): ...
    @patch_call(forward)
    def __call__(self) -> None: ...

class EmbeddingBag(Module):
    __constants__: Incomplete
    num_embeddings: int
    embedding_dim: int
    max_norm: float | None
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    mode: str
    sparse: bool
    include_last_offset: bool
    padding_idx: int | None
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        _weight: Tensor | None = None,
        include_last_offset: bool = False,
        padding_idx: int | None = None,
        device=None,
        dtype=None,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(
        self, input: Tensor, offsets: Tensor | None = None, per_sample_weights: Tensor | None = None
    ) -> Tensor: ...
    def extra_repr(self) -> str: ...
    @classmethod
    def from_pretrained(
        cls,
        embeddings: Tensor,
        freeze: bool = True,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        include_last_offset: bool = False,
        padding_idx: int | None = None,
    ) -> EmbeddingBag: ...
