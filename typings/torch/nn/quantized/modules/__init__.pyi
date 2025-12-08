from torch.ao.nn.quantized.modules import DeQuantize, Quantize
from torch.ao.nn.quantized.modules.activation import (
    ELU,
    Hardswish,
    LeakyReLU,
    MultiheadAttention,
    PReLU,
    ReLU6,
    Sigmoid,
    Softmax,
)
from torch.ao.nn.quantized.modules.batchnorm import BatchNorm2d, BatchNorm3d
from torch.ao.nn.quantized.modules.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from torch.ao.nn.quantized.modules.dropout import Dropout
from torch.ao.nn.quantized.modules.embedding_ops import Embedding, EmbeddingBag
from torch.ao.nn.quantized.modules.functional_modules import (
    FloatFunctional,
    FXFloatFunctional,
    QFunctional,
)
from torch.ao.nn.quantized.modules.linear import Linear
from torch.ao.nn.quantized.modules.normalization import (
    GroupNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LayerNorm,
)
from torch.ao.nn.quantized.modules.rnn import LSTM

r"""Quantized Modules.

Note::
    The `torch.nn.quantized` namespace is in the process of being deprecated.
    Please, use `torch.ao.nn.quantized` instead.
"""
__all__ = [
    "ELU",
    "LSTM",
    "BatchNorm2d",
    "BatchNorm3d",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "DeQuantize",
    "Dropout",
    "Embedding",
    "EmbeddingBag",
    "FXFloatFunctional",
    "FloatFunctional",
    "GroupNorm",
    "Hardswish",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "LeakyReLU",
    "Linear",
    "MultiheadAttention",
    "PReLU",
    "QFunctional",
    "Quantize",
    "ReLU6",
    "Sigmoid",
    "Softmax",
]
