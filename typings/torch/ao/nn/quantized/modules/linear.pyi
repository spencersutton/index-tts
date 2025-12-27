import torch

from .utils import WeightedQuantizedModule

__all__ = ["Linear", "LinearPackedParams"]

class LinearPackedParams(torch.nn.Module):
    _version = ...
    def __init__(self, dtype=...) -> None: ...
    @torch.jit.export
    def set_weight_bias(self, weight: torch.Tensor, bias: torch.Tensor | None) -> None: ...
    def forward(self, x): ...

class Linear(WeightedQuantizedModule):
    r"""
    A quantized linear module with quantized tensor as inputs and outputs.
    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear for documentation.

    Similar to :class:`~torch.nn.Linear`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module of
                         shape :math:`(\text{out\_features}, \text{in\_features})`.
        bias (Tensor): the non-learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized to zero.
        scale: `scale` parameter of output Quantized Tensor, type: double
        zero_point: `zero_point` parameter for output Quantized Tensor, type: long

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> m = nn.quantized.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> # xdoctest: +SKIP
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, torch.quint8)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    _version = ...
    _FLOAT_MODULE = ...
    def __init__(self, in_features, out_features, bias_=..., dtype=...) -> None: ...
    def extra_repr(self) -> str: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def weight(self) -> Any: ...
    def bias(self) -> Any: ...
    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=...) -> Self:
        """
        Create a quantized module from an observed float module

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by the user
            use_precomputed_fake_quant (bool): if True, the module will reuse min/max
                          values from the precomputed fake quant module.
        """
    @classmethod
    def from_reference(cls, ref_qlinear, output_scale, output_zero_point) -> Self:
        """
        Create a (fbgemm/qnnpack) quantized module from a reference quantized module

        Args:
            ref_qlinear (Module): a reference quantized linear module, either produced by torch.ao.quantization
                          utilities or provided by the user
            output_scale (float): scale for output Tensor
            output_zero_point (int): zero point for output Tensor
        """
