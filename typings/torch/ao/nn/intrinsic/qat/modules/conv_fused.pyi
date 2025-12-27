from typing import ClassVar

import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
from torch import nn

__all__ = [
    "ConvBn1d",
    "ConvBn2d",
    "ConvBn3d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBnReLU3d",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "freeze_bn_stats",
    "update_bn_stats",
]
_BN_CLASS_MAP = ...

class _ConvBnNd(nn.modules.conv._ConvNd, nni._FusedModule):
    _version = ...
    _FLOAT_MODULE: ClassVar[type[nn.modules.conv._ConvNd]]
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
        eps=...,
        momentum=...,
        freeze_bn=...,
        qconfig=...,
        dim=...,
    ) -> None: ...
    def reset_running_stats(self) -> None: ...
    def reset_bn_parameters(self) -> None: ...
    def reset_parameters(self) -> None: ...
    def update_bn_stats(self) -> Self: ...
    def freeze_bn_stats(self) -> Self: ...
    def extra_repr(self) -> str: ...
    def forward(self, input) -> Tensor | Any: ...
    def train(self, mode=...) -> Self:
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self:
        """
        Create a qat module from a float module or qparams_dict

        Args: `mod` a float module, either produced by torch.ao.quantization utilities
        or directly from user
        """
    def to_float(self): ...

class ConvBn1d(_ConvBnNd, nn.Conv1d):
    """
    A ConvBn1d module is a module fused from Conv1d and BatchNorm1d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv1d` and
    :class:`torch.nn.BatchNorm1d`.

    Similar to :class:`torch.nn.Conv1d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_BN_MODULE: ClassVar[type[nn.BatchNorm1d]] = ...
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
    _FLOAT_MODULE: ClassVar[type[nn.Module]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv1d]] = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        dilation=...,
        groups=...,
        bias=...,
        padding_mode=...,
        eps=...,
        momentum=...,
        freeze_bn=...,
        qconfig=...,
    ) -> None: ...

class ConvBnReLU1d(ConvBn1d):
    """
    A ConvBnReLU1d module is a module fused from Conv1d, BatchNorm1d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv1d` and
    :class:`torch.nn.BatchNorm1d` and :class:`torch.nn.ReLU`.

    Similar to `torch.nn.Conv1d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE: ClassVar[type[nn.Module]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv1d]] = ...
    _FLOAT_BN_MODULE: ClassVar[type[nn.BatchNorm1d]] = ...
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
    _FUSED_FLOAT_MODULE: ClassVar[type[nn.Module] | None] = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        dilation=...,
        groups=...,
        bias=...,
        padding_mode=...,
        eps=...,
        momentum=...,
        freeze_bn=...,
        qconfig=...,
    ) -> None: ...
    def forward(self, input) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self: ...

class ConvReLU1d(nnqat.Conv1d, nni._FusedModule):
    """
    A ConvReLU1d module is a fused module of Conv1d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv1d` and
    :class:`~torch.nn.BatchNorm1d`.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE: ClassVar[type[nni.ConvReLU1d]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv1d]] = ...
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None] = ...
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        dilation=...,
        groups=...,
        bias=...,
        padding_mode=...,
        qconfig=...,
    ) -> None: ...
    def forward(self, input) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...

class ConvBn2d(_ConvBnNd, nn.Conv2d):
    """
    A ConvBn2d module is a module fused from Conv2d and BatchNorm2d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.

    Similar to :class:`torch.nn.Conv2d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE: ClassVar[type[nni.ConvBn2d]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv2d]] = ...
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None] = ...
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        dilation=...,
        groups=...,
        bias=...,
        padding_mode=...,
        eps=...,
        momentum=...,
        freeze_bn=...,
        qconfig=...,
    ) -> None: ...

class ConvBnReLU2d(ConvBn2d):
    """
    A ConvBnReLU2d module is a module fused from Conv2d, BatchNorm2d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d` and :class:`torch.nn.ReLU`.

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE: ClassVar[type[nni.ConvBnReLU2d]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv2d]] = ...
    _FLOAT_BN_MODULE: ClassVar[type[nn.BatchNorm2d]] = ...
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
    _FUSED_FLOAT_MODULE: ClassVar[type[nni.ConvReLU2d] | None] = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        dilation=...,
        groups=...,
        bias=...,
        padding_mode=...,
        eps=...,
        momentum=...,
        freeze_bn=...,
        qconfig=...,
    ) -> None: ...
    def forward(self, input) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self: ...

class ConvReLU2d(nnqat.Conv2d, nni._FusedModule):
    """
    A ConvReLU2d module is a fused module of Conv2d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv2d` and
    :class:`~torch.nn.BatchNorm2d`.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE: ClassVar[type[nn.Module]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv2d]] = ...
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None] = ...
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        dilation=...,
        groups=...,
        bias=...,
        padding_mode=...,
        qconfig=...,
    ) -> None: ...
    def forward(self, input) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...

class ConvBn3d(_ConvBnNd, nn.Conv3d):
    """
    A ConvBn3d module is a module fused from Conv3d and BatchNorm3d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv3d` and
    :class:`torch.nn.BatchNorm3d`.

    Similar to :class:`torch.nn.Conv3d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE: ClassVar[type[nni.ConvBn3d]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv3d]] = ...
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None] = ...
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        dilation=...,
        groups=...,
        bias=...,
        padding_mode=...,
        eps=...,
        momentum=...,
        freeze_bn=...,
        qconfig=...,
    ) -> None: ...

class ConvBnReLU3d(ConvBn3d):
    """
    A ConvBnReLU3d module is a module fused from Conv3d, BatchNorm3d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv3d` and
    :class:`torch.nn.BatchNorm3d` and :class:`torch.nn.ReLU`.

    Similar to `torch.nn.Conv3d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE: ClassVar[type[nni.ConvBnReLU3d]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv3d]] = ...
    _FLOAT_BN_MODULE: ClassVar[type[nn.BatchNorm3d]] = ...
    _FLOAT_RELU_MODULE: ClassVar[type[nn.ReLU] | None] = ...
    _FUSED_FLOAT_MODULE: ClassVar[type[nni.ConvReLU3d] | None] = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        dilation=...,
        groups=...,
        bias=...,
        padding_mode=...,
        eps=...,
        momentum=...,
        freeze_bn=...,
        qconfig=...,
    ) -> None: ...
    def forward(self, input) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self: ...

class ConvReLU3d(nnqat.Conv3d, nni._FusedModule):
    """
    A ConvReLU3d module is a fused module of Conv3d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv3d` and
    :class:`~torch.nn.BatchNorm3d`.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE: ClassVar[type[nni.ConvReLU3d]] = ...
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv3d]] = ...
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None] = ...
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = ...
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=...,
        padding=...,
        dilation=...,
        groups=...,
        bias=...,
        padding_mode=...,
        qconfig=...,
    ) -> None: ...
    def forward(self, input) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...): ...

def update_bn_stats(mod) -> None: ...
def freeze_bn_stats(mod) -> None: ...
