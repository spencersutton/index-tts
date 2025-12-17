import contextlib
from collections.abc import Callable

import torch

INTERMEDIATE_HOOKS: list[Callable[[str, torch.Tensor], None]] = ...

@contextlib.contextmanager
def intermediate_hook(fn): ...
def run_intermediate_hooks(name, val): ...
