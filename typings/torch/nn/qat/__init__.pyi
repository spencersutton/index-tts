"""
QAT Dynamic Modules.

This package is in the process of being deprecated.
Please, use `torch.ao.nn.qat.dynamic` instead.
"""

from torch.nn.qat.modules import *

__all__ = ["Conv1d", "Conv2d", "Conv3d", "Embedding", "EmbeddingBag", "Linear"]
