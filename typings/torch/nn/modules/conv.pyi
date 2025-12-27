from typing import Literal
from warnings import deprecated

from torch import Tensor
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.parameter import UninitializedParameter

from indextts.util import patch_call

from .lazy import LazyModuleMixin
from .module import Module

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
]
convolution_notes = ...

class _ConvNd(Module):
    __constants__ = ...
    __annotations__ = ...
    in_channels: int
    _reversed_padding_repeated_twice: list[int]
    out_channels: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: str | tuple[int, ...]
    dilation: tuple[int, ...]
    transposed: bool
    output_padding: tuple[int, ...]
    groups: int
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"]
    weight: Tensor
    bias: Tensor | None
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: str | tuple[int, ...],
        dilation: tuple[int, ...],
        transposed: bool,
        output_padding: tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"],
        device=...,
        dtype=...,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def extra_repr(self) -> str: ...
    def __setstate__(self, state) -> None: ...

class Conv1d(_ConvNd):
    r"""
    Applies a 1D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})` can be
    precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.


    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {'valid', 'same'} or a tuple of ints giving the
      amount of implicit padding applied on both sides.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\frac{\text{out\_channels}}{\text{in\_channels}}`).

    Note:
        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also known as a "depthwise convolution".

        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
        a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments
        :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})`.
    Note:
        In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.

    Note:
        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
        the input so the output has the shape as the input. However, this mode
        doesn't support any stride values other than 1.

    Note:
        This module supports complex data types i.e. ``complex32, complex64, complex128``.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``



    Shape:
        - Input: :math:`(N, C_{in}, L_{in})` or :math:`(C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out})`, where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels},
            \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``, then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`

    Examples::

        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    __doc__ = ...
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = ...,
        padding: str | _size_1_t = ...,
        dilation: _size_1_t = ...,
        groups: int = ...,
        bias: bool = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    @patch_call(forward)
    def __call__(self, input: ...) -> None: ...

class Conv2d(_ConvNd):
    r"""
    Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.


    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {'valid', 'same'} or an int / a tuple of ints giving the
      amount of implicit padding applied on both sides.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.


    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\frac{\text{out\_channels}}{\text{in\_channels}}`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Note:
        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also known as a "depthwise convolution".

        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
        a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments
        :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})`.

    Note:
        In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.

    Note:
        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
        the input so the output has the shape as the input. However, this mode
        doesn't support any stride values other than 1.

    Note:
        This module supports complex data types i.e. ``complex32, complex64, complex128``.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``


    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples:

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    __doc__ = ...
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = ...,
        padding: str | _size_2_t = ...,
        dilation: _size_2_t = ...,
        groups: int = ...,
        bias: bool = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    @patch_call(forward)
    def __call__(self, input: ...) -> None: ...

class Conv3d(_ConvNd):
    r"""
    Applies a 3D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, D, H, W)`
    and output :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` can be precisely described as:

    .. math::
        out(N_i, C_{out_j}) = bias(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) \star input(N_i, k)

    where :math:`\star` is the valid 3D `cross-correlation`_ operator


    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {'valid', 'same'} or a tuple of ints giving the
      amount of implicit padding applied on both sides.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.


    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\frac{\text{out\_channels}}{\text{in\_channels}}`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Note:
        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also known as a "depthwise convolution".

        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
        a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments
        :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})`.

    Note:
        In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.

    Note:
        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
        the input so the output has the shape as the input. However, this mode
        doesn't support any stride values other than 1.

    Note:
        This module supports complex data types i.e. ``complex32, complex64, complex128``.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all six sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``


    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` or :math:`(C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` or :math:`(C_{out}, D_{out}, H_{out}, W_{out})`,
          where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0]
                    \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1]
                    \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2]
                    \times (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]}, \text{kernel\_size[2]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    __doc__ = ...
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = ...,
        padding: str | _size_3_t = ...,
        dilation: _size_3_t = ...,
        groups: int = ...,
        bias: bool = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

class _ConvTransposeNd(_ConvNd):
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
        device=...,
        dtype=...,
    ) -> None: ...

class ConvTranspose1d(_ConvTransposeNd):
    r"""
    Applies a 1D transposed convolution operator over an input image
    composed of several input planes.

    This module can be seen as the gradient of Conv1d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation as it does
    not compute a true inverse of convolution). For more information, see the visualizations
    `here`_ and the `Deconvolutional Networks`_ paper.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero padding on both
      sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\frac{\text{out\_channels}}{\text{in\_channels}}`).

    Note:
        The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv1d` and a :class:`~torch.nn.ConvTranspose1d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv1d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Note:
        In some circumstances when using the CUDA backend with CuDNN, this operator
        may select a nondeterministic algorithm to increase performance. If this is
        undesirable, you can try to make the operation deterministic (potentially at
        a performance cost) by setting ``torch.backends.cudnn.deterministic =
        True``.
        Please see the notes on :doc:`/notes/randomness` for background.


    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1


    Shape:
        - Input: :math:`(N, C_{in}, L_{in})` or :math:`(C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out})`, where

          .. math::
              L_{out} = (L_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{dilation}
                        \times (\text{kernel\_size} - 1) + \text{output\_padding} + 1

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{in\_channels}, \frac{\text{out\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{out} * \text{kernel\_size}}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels).
                         If :attr:`bias` is ``True``, then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{out} * \text{kernel\_size}}`

    .. _`here`:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    .. _`Deconvolutional Networks`:
        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf
    """

    __doc__ = ...
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = ...,
        padding: _size_1_t = ...,
        output_padding: _size_1_t = ...,
        groups: int = ...,
        bias: bool = ...,
        dilation: _size_1_t = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor, output_size: list[int] | None = ...) -> Tensor: ...
    @patch_call(forward)
    def __call__(self) -> None: ...

class ConvTranspose2d(_ConvTransposeNd):
    r"""
    Applies a 2D transposed convolution operator over an input image
    composed of several input planes.

    This module can be seen as the gradient of Conv2d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation as it does
    not compute a true inverse of convolution). For more information, see the visualizations
    `here`_ and the `Deconvolutional Networks`_ paper.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    * :attr:`stride` controls the stride for the cross-correlation. When stride > 1, ConvTranspose2d inserts zeros between input
      elements along the spatial dimensions before applying the convolution kernel. This zero-insertion operation is the standard
      behavior of transposed convolutions, which can increase the spatial resolution and is equivalent to a learnable
      upsampling operation.

    * :attr:`padding` controls the amount of implicit zero padding on both
      sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\frac{\text{out\_channels}}{\text{in\_channels}}`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimensions
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Note:
        The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv2d` and a :class:`~torch.nn.ConvTranspose2d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv2d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Note:
        In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1


    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`, where

        .. math::
              H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) + \text{output\_padding}[0] + 1
        .. math::
              W_{out} = (W_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) + \text{output\_padding}[1] + 1

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{in\_channels}, \frac{\text{out\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels)
                         If :attr:`bias` is ``True``, then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12)
        >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])

    .. _`here`:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    .. _`Deconvolutional Networks`:
        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf
    """

    __doc__ = ...
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = ...,
        padding: _size_2_t = ...,
        output_padding: _size_2_t = ...,
        groups: int = ...,
        bias: bool = ...,
        dilation: _size_2_t = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor, output_size: list[int] | None = ...) -> Tensor:
        """
        Performs the forward pass.

        Attributes:
            input (Tensor): The input tensor.
            output_size (list[int], optional): A list of integers representing
                the size of the output tensor. Default is None.
        """

class ConvTranspose3d(_ConvTransposeNd):
    r"""
    Applies a 3D transposed convolution operator over an input image composed of several input
    planes.
    The transposed convolution operator multiplies each input value element-wise by a learnable kernel,
    and sums over the outputs from all input feature planes.

    This module can be seen as the gradient of Conv3d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation as it does
    not compute a true inverse of convolution). For more information, see the visualizations
    `here`_ and the `Deconvolutional Networks`_ paper.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero padding on both
      sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\frac{\text{out\_channels}}{\text{in\_channels}}`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimensions
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Note:
        The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv3d` and a :class:`~torch.nn.ConvTranspose3d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv3d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Note:
        In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1


    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` or :math:`(C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` or
          :math:`(C_{out}, D_{out}, H_{out}, W_{out})`, where

        .. math::
              D_{out} = (D_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) + \text{output\_padding}[0] + 1
        .. math::
              H_{out} = (H_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) + \text{output\_padding}[1] + 1
        .. math::
              W_{out} = (W_{in} - 1) \times \text{stride}[2] - 2 \times \text{padding}[2] + \text{dilation}[2]
                        \times (\text{kernel\_size}[2] - 1) + \text{output\_padding}[2] + 1


    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{in\_channels}, \frac{\text{out\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]}, \text{kernel\_size[2]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels)
                         If :attr:`bias` is ``True``, then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.ConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = m(input)

    .. _`here`:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    .. _`Deconvolutional Networks`:
        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf
    """

    __doc__ = ...
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = ...,
        padding: _size_3_t = ...,
        output_padding: _size_3_t = ...,
        groups: int = ...,
        bias: bool = ...,
        dilation: _size_3_t = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, input: Tensor, output_size: list[int] | None = ...) -> Tensor: ...

class _ConvTransposeMixin(_ConvTransposeNd):
    @deprecated(
        "`_ConvTransposeMixin` is a deprecated internal class. Please consider using public APIs.",
        category=FutureWarning,
    )
    def __init__(self, *args, **kwargs) -> None: ...

class _LazyConvXdMixin(LazyModuleMixin):
    groups: int
    transposed: bool
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, ...]
    weight: UninitializedParameter
    bias: UninitializedParameter
    def reset_parameters(self) -> None: ...
    def initialize_parameters(self, input: Tensor, *args, **kwargs) -> None: ...

class LazyConv1d(_LazyConvXdMixin, Conv1d):
    """
    A :class:`torch.nn.Conv1d` module with lazy initialization of the ``in_channels`` argument.

    The ``in_channels`` argument of the :class:`Conv1d` is inferred from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``

    .. seealso:: :class:`torch.nn.Conv1d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
    """

    cls_to_become = ...
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = ...,
        padding: _size_1_t = ...,
        dilation: _size_1_t = ...,
        groups: int = ...,
        bias: bool = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...

class LazyConv2d(_LazyConvXdMixin, Conv2d):
    """
    A :class:`torch.nn.Conv2d` module with lazy initialization of the ``in_channels`` argument.

    The ``in_channels`` argument of the :class:`Conv2d` that is inferred from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``

    .. seealso:: :class:`torch.nn.Conv2d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
    """

    cls_to_become = ...
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = ...,
        padding: _size_2_t = ...,
        dilation: _size_2_t = ...,
        groups: int = ...,
        bias: bool = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...

class LazyConv3d(_LazyConvXdMixin, Conv3d):
    """
    A :class:`torch.nn.Conv3d` module with lazy initialization of the ``in_channels`` argument.

    The ``in_channels`` argument of the :class:`Conv3d` that is inferred from
    the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``

    .. seealso:: :class:`torch.nn.Conv3d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
    """

    cls_to_become = ...
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = ...,
        padding: _size_3_t = ...,
        dilation: _size_3_t = ...,
        groups: int = ...,
        bias: bool = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...

class LazyConvTranspose1d(_LazyConvXdMixin, ConvTranspose1d):
    """
    A :class:`torch.nn.ConvTranspose1d` module with lazy initialization of the ``in_channels`` argument.

    The ``in_channels`` argument of the :class:`ConvTranspose1d` that is inferred from
    the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    .. seealso:: :class:`torch.nn.ConvTranspose1d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
    """

    cls_to_become = ...
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = ...,
        padding: _size_1_t = ...,
        output_padding: _size_1_t = ...,
        groups: int = ...,
        bias: bool = ...,
        dilation: _size_1_t = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...

class LazyConvTranspose2d(_LazyConvXdMixin, ConvTranspose2d):
    """
    A :class:`torch.nn.ConvTranspose2d` module with lazy initialization of the ``in_channels`` argument.

    The ``in_channels`` argument of the :class:`ConvTranspose2d` is inferred from
    the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    .. seealso:: :class:`torch.nn.ConvTranspose2d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
    """

    cls_to_become = ...
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = ...,
        padding: _size_2_t = ...,
        output_padding: _size_2_t = ...,
        groups: int = ...,
        bias: bool = ...,
        dilation: int = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...

class LazyConvTranspose3d(_LazyConvXdMixin, ConvTranspose3d):
    """
    A :class:`torch.nn.ConvTranspose3d` module with lazy initialization of the ``in_channels`` argument.

    The ``in_channels`` argument of the :class:`ConvTranspose3d` is inferred from
    the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight` and `bias`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    .. seealso:: :class:`torch.nn.ConvTranspose3d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`
    """

    cls_to_become = ...
    def __init__(
        self,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = ...,
        padding: _size_3_t = ...,
        output_padding: _size_3_t = ...,
        groups: int = ...,
        bias: bool = ...,
        dilation: _size_3_t = ...,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = ...,
        device=...,
        dtype=...,
    ) -> None: ...
