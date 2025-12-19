from collections.abc import Sequence

import torch
from torch.nn.modules import Module

__all__ = ["replicate"]

def replicate[T: Module](network: T, devices: Sequence[int | torch.device], detach: bool = ...) -> list[T]: ...
