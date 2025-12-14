import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING: ...
INTERMEDIATE_HOOKS: list[Callable[[str, torch.Tensor], None]] = ...

@contextlib.contextmanager
def intermediate_hook(fn):  # -> Generator[None, Any, None]:
    ...
def run_intermediate_hooks(name, val):  # -> None:
    ...
