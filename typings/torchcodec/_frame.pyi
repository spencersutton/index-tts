from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from torch import Tensor

@dataclass
class Frame(Iterable):
    data: Tensor
    pts_seconds: float
    duration_seconds: float
    def __post_init__(self):  # -> None:
        ...
    def __iter__(self) -> Iterator[Tensor | float]: ...

@dataclass
class FrameBatch(Iterable):
    data: Tensor
    pts_seconds: Tensor
    duration_seconds: Tensor
    def __post_init__(self) -> None: ...
    def __iter__(self) -> Iterator[FrameBatch]: ...
    def __getitem__(self, key) -> FrameBatch: ...
    def __len__(self) -> int: ...

@dataclass
class AudioSamples(Iterable):
    data: Tensor
    pts_seconds: float
    duration_seconds: float
    sample_rate: int
    def __post_init__(self) -> None: ...
    def __iter__(self) -> Iterator[Tensor | float]: ...
