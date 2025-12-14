from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

import torch
from torch.nn.modules import Module

if TYPE_CHECKING: ...
__all__ = ["replicate"]
T = TypeVar("T", bound=Module)

def replicate[T: Module](network: T, devices: Sequence[int | torch.device], detach: bool = ...) -> list[T]: ...
