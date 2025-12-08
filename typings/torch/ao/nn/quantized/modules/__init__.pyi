import torch

from .activation import (
    ELU,
    Hardswish,
    LeakyReLU,
    MultiheadAttention,
    PReLU,
    ReLU6,
    Sigmoid,
    Softmax,
)
from .batchnorm import BatchNorm2d, BatchNorm3d
from .conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from .dropout import Dropout
from .embedding_ops import Embedding, EmbeddingBag
from .functional_modules import FloatFunctional, FXFloatFunctional, QFunctional
from .linear import Linear
from .normalization import (
    GroupNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LayerNorm,
)
from .rnn import LSTM

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

class Quantize(torch.nn.Module):
    scale: torch.Tensor
    zero_point: torch.Tensor
    def __init__(self, scale, zero_point, dtype, factory_kwargs=...) -> None: ...
    def forward(self, X) -> Tensor: ...
    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=...) -> Quantize: ...
    def extra_repr(self) -> str: ...

class DeQuantize(torch.nn.Module):
    def forward(self, Xq): ...
    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=...) -> DeQuantize: ...
