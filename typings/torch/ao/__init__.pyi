from types import ModuleType
from typing import TYPE_CHECKING as _TYPE_CHECKING
from torch.ao import nn as nn, ns as ns, pruning as pruning, quantization as quantization

if _TYPE_CHECKING: ...
__all__ = ["nn", "ns", "pruning", "quantization"]

def __getattr__(name: str) -> ModuleType: ...
