import torch.nn as nn
from collections.abc import Generator
from typing import Optional
from .contract import _State, contract

class _CheckpointState(_State):
    enable_hook: bool = ...
    _ac_generator: Optional[Generator[None, None, None]]

@contract(_CheckpointState)
def checkpoint(module: nn.Module, **kwargs) -> nn.Module: ...
