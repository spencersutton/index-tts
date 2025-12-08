from types import TracebackType
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING: ...

def is_available() -> bool: ...

if is_available() and not torch._C._dist_autograd_init(): ...
if is_available(): ...
__all__ = ["context", "is_available"]

class context:
    def __enter__(self) -> int: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
