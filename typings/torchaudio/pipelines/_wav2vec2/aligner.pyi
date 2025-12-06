from abc import ABC, abstractmethod

from torch import Tensor
from torchaudio.functional import TokenSpan

class ITokenizer(ABC):
    @abstractmethod
    def __call__(self, transcript: list[str]) -> list[list[str]]: ...

class Tokenizer(ITokenizer):
    def __init__(self, dictionary: dict[str, int]) -> None: ...
    def __call__(self, transcript: list[str]) -> list[list[int]]: ...

class IAligner(ABC):
    @abstractmethod
    def __call__(self, emission: Tensor, tokens: list[list[int]]) -> list[list[TokenSpan]]: ...

class Aligner(IAligner):
    def __init__(self, blank) -> None: ...
    def __call__(self, emission: Tensor, tokens: list[list[int]]) -> list[list[TokenSpan]]: ...
