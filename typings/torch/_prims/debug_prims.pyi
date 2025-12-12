import contextlib
from collections.abc import Generator
from typing import Optional
from torch.utils._content_store import ContentStoreReader

LOAD_TENSOR_READER: Optional[ContentStoreReader] = ...

@contextlib.contextmanager
def load_tensor_reader(loc: str) -> Generator[None, None, None]: ...
def register_debug_prims() -> None: ...
