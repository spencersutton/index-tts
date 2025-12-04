from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from torch import Tensor
from torchaudio.models import Tacotron2

class _TextProcessor(ABC):
    @property
    @abstractmethod
    def tokens(self):  # -> None:
        ...
    @abstractmethod
    def __call__(self, texts: str | list[str]) -> tuple[Tensor, Tensor]: ...

class _Vocoder(ABC):
    @property
    @abstractmethod
    def sample_rate(self):  # -> None:
        ...
    @abstractmethod
    def __call__(self, specgrams: Tensor, lengths: Tensor | None = ...) -> tuple[Tensor, Tensor | None]: ...

class Tacotron2TTSBundle(ABC):
    class TextProcessor(_TextProcessor): ...
    class Vocoder(_Vocoder): ...

    @abstractmethod
    def get_text_processor(self, *, dl_kwargs=...) -> TextProcessor: ...
    @abstractmethod
    def get_vocoder(self, *, dl_kwargs=...) -> Vocoder: ...
    @abstractmethod
    def get_tacotron2(self, *, dl_kwargs=...) -> Tacotron2: ...
