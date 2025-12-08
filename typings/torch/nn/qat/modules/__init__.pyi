from torch.ao.nn.qat.modules.conv import Conv1d, Conv2d, Conv3d
from torch.ao.nn.qat.modules.embedding_ops import Embedding, EmbeddingBag
from torch.ao.nn.qat.modules.linear import Linear

r"""QAT Modules.

This package is in the process of being deprecated.
Please, use `torch.ao.nn.qat.modules` instead.
"""
__all__ = ["Conv1d", "Conv2d", "Conv3d", "Embedding", "EmbeddingBag", "Linear"]
