import contextvars
import io
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from torch import Tensor

ERROR_REPORTING_INSTRUCTIONS = ...

def create_decoder(
    *,
    source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
    seek_mode: str,
) -> Tensor: ...

_CUDA_BACKEND: contextvars.ContextVar[str] = ...

@contextmanager
def set_cuda_backend(backend: str) -> Generator[None]: ...
