from collections.abc import Sequence
from typing import Literal

from torch import Tensor
from torch.types import _int, _size

def adaptive_avg_pool2d(input: Tensor, output_size: _int | _size) -> Tensor: ...
def adaptive_avg_pool3d(input: Tensor, output_size: _int | _size) -> Tensor: ...
def adaptive_max_pool2d(input: Tensor, output_size: _int | _size) -> tuple[Tensor, Tensor]: ...
def adaptive_max_pool3d(input: Tensor, output_size: _int | _size) -> tuple[Tensor, Tensor]: ...
def avg_pool2d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size | None = ...,
    padding: _int | _size = ...,
    ceil_mode: bool = ...,
    count_include_pad: bool = ...,
    divisor_override: int | None = ...,
) -> Tensor:
    r"""
    avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor

    Applies 2D average-pooling operation in :math:`kH \times kW` regions by step size
    :math:`sH \times sW` steps. The number of output features is equal to the number of
    input planes.

    See :class:`~torch.nn.AvgPool2d` for details and output shape.

    Args:
        input: input tensor :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
        kernel_size: size of the pooling region. Can be a single number, a single-element tuple or a
          tuple `(kH, kW)`
        stride: stride of the pooling operation. Can be a single number, a single-element tuple or a
          tuple `(sH, sW)`. Default: :attr:`kernel_size`
        padding: implicit zero paddings on both sides of the input. Can be a
          single number, a single-element tuple or a tuple `(padH, padW)`.
          Should be at most half of effective kernel size, that
          is :math:`((kernelSize - 1) * dilation + 1) / 2`. Default: 0
        ceil_mode: when True, will use `ceil` instead of `floor` in the formula
            to compute the output shape. Default: ``False``
        count_include_pad: when True, will include the zero-padding in the
            averaging calculation. Default: ``True``
        divisor_override: if specified, it will be used as divisor, otherwise
             size of the pooling region will be used. Default: None
    """

def avg_pool3d(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size | None = ...,
    padding: _int | _size = ...,
    ceil_mode: bool = ...,
    count_include_pad: bool = ...,
    divisor_override: int | None = ...,
) -> Tensor:
    r"""
    avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor

    Applies 3D average-pooling operation in :math:`kT \times kH \times kW` regions by step
    size :math:`sT \times sH \times sW` steps. The number of output features is equal to
    :math:`\lfloor\frac{\text{input planes}}{sT}\rfloor`.

    See :class:`~torch.nn.AvgPool3d` for details and output shape.

    Args:
        input: input tensor :math:`(\text{minibatch} , \text{in\_channels} , iT \times iH , iW)`
        kernel_size: size of the pooling region. Can be a single number or a
          tuple `(kT, kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a
          tuple `(sT, sH, sW)`. Default: :attr:`kernel_size`
        padding: implicit zero paddings on both sides of the input. Can be a
          single number or a tuple `(padT, padH, padW)`. Should be at most half
          of effective kernel size, that is :math:`((kernelSize - 1) * dilation + 1) / 2`.
          Default: 0
        ceil_mode: when True, will use `ceil` instead of `floor` in the formula
            to compute the output shape
        count_include_pad: when True, will include the zero-padding in the
            averaging calculation
        divisor_override: if specified, it will be used as divisor, otherwise
            size of the pooling region will be used. Default: None
    """

def binary_cross_entropy(
    input: Tensor, target: Tensor, weight: Tensor | None = ..., reduction: str = ...
) -> Tensor: ...
def col2im(
    input: Tensor,
    output_size: _int | _size,
    kernel_size: _int | _size,
    dilation: _int | _size,
    stride: _int | _size | None = ...,
    padding: _int | _size = ...,
) -> Tensor: ...
def cross_entropy_loss(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = ...,
    reduction: str = ...,
    ignore_index: int = ...,
    label_smoothing: float = ...,
) -> Tensor: ...
def elu(input: Tensor, alpha: float = ..., scale: float = ..., input_scale: float = ...) -> Tensor: ...
def elu_(input: Tensor, alpha: float = ...) -> Tensor:
    """
    elu_(input, alpha=1.) -> Tensor

    In-place version of :func:`~elu`.
    """

def fractional_max_pool2d(
    input: Tensor, kernel_size: _int | _size, output_size: _int | _size, _random_samples: Tensor
) -> tuple[Tensor, Tensor]: ...
def fractional_max_pool3d(
    input: Tensor, kernel_size: _int | _size, output_size: _int | _size, _random_samples: Tensor
) -> tuple[Tensor, Tensor]: ...
def gelu(input: Tensor, approximate: str = ...) -> Tensor:
    r"""
    gelu(input, approximate = 'none') -> Tensor

    When the approximate argument is 'none', it applies element-wise the function
    :math:`\text{GELU}(x) = x * \Phi(x)`

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    When the approximate argument is 'tanh', Gelu is estimated with

    .. math::
        \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))

    See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_.
    """

def glu(input: Tensor, dim: int = ...) -> Tensor: ...
def hardsigmoid(input: Tensor, *, out: Tensor | None = ...) -> Tensor: ...
def hardsigmoid_(input: Tensor) -> Tensor: ...
def hardswish(input: Tensor) -> Tensor: ...
def hardswish_(input: Tensor) -> Tensor: ...
def hardtanh(input: Tensor, min_val: float = ..., max_val: float = ..., *, out: Tensor | None = ...) -> Tensor: ...
def hardtanh_(input: Tensor, min_val: float = ..., max_val: float = ...) -> Tensor:
    """
    hardtanh_(input, min_val=-1., max_val=1.) -> Tensor

    In-place version of :func:`~hardtanh`.
    """

def huber_loss(input: Tensor, target: Tensor, reduction: str = ..., delta: float = ...) -> Tensor: ...
def leaky_relu(input: Tensor, negative_slope: float = ..., *, out: Tensor | None = ...) -> Tensor: ...
def leaky_relu_(input: Tensor, negative_slope: float = ...) -> Tensor:
    """
    leaky_relu_(input, negative_slope=0.01) -> Tensor

    In-place version of :func:`~leaky_relu`.
    """

def linear(input: Tensor, weight: Tensor, bias: Tensor | None = ...) -> Tensor:
    r"""
    linear(input, weight, bias=None) -> Tensor

    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This operation supports 2-D :attr:`weight` with :ref:`sparse layout<sparse-docs>`


    .. warning::
        Sparse support is a beta feature and some layout(s)/dtype/device combinations may not be supported,
        or may not have autograd support. If you notice missing functionality please
        open a feature request.

    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Shape:

        - Input: :math:`(*, in\_features)` where `*` means any number of
          additional dimensions, including none
        - Weight: :math:`(out\_features, in\_features)` or :math:`(in\_features)`
        - Bias: :math:`(out\_features)` or :math:`()`
        - Output: :math:`(*, out\_features)` or :math:`(*)`, based on the shape of the weight
    """

def log_sigmoid(input: Tensor) -> Tensor:
    r"""
    logsigmoid(input) -> Tensor

    Applies element-wise :math:`\text{LogSigmoid}(x_i) = \log \left(\frac{1}{1 + \exp(-x_i)}\right)`

    See :class:`~torch.nn.LogSigmoid` for more details.
    """

def max_pool2d_with_indices(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size | None = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: bool = ...,
) -> tuple[Tensor, Tensor]: ...
def max_pool3d_with_indices(
    input: Tensor,
    kernel_size: _int | _size,
    stride: _int | _size | None = ...,
    padding: _int | _size = ...,
    dilation: _int | _size = ...,
    ceil_mode: bool = ...,
) -> tuple[Tensor, Tensor]: ...
def max_unpool2d(input: Tensor, indices: Tensor, output_size: Sequence[int] | None) -> Tensor: ...
def max_unpool3d(
    input: Tensor, indices: Tensor, output_size: Sequence[int] | None, stride: _int | _size, padding: _int | _size
) -> Tensor: ...
def one_hot(tensor: Tensor, num_classes: int = ...) -> Tensor:
    """
    one_hot(tensor, num_classes=-1) -> LongTensor

    Takes LongTensor with index values of shape ``(*)`` and returns a tensor
    of shape ``(*, num_classes)`` that have zeros everywhere except where the
    index of last dimension matches the corresponding value of the input tensor,
    in which case it will be 1.

    See also `One-hot on Wikipedia`_ .

    .. _One-hot on Wikipedia:
        https://en.wikipedia.org/wiki/One-hot

    Arguments:
        tensor (LongTensor): class values of any shape.
        num_classes (int, optional):  Total number of classes. If set to -1, the number
            of classes will be inferred as one greater than the largest class
            value in the input tensor. Default: -1

    Returns:
        LongTensor that has one more dimension with 1 values at the
        index of last dimension indicated by the input, and 0 everywhere
        else.

    Examples:
        >>> F.one_hot(torch.arange(0, 5) % 3)
        tensor([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]])
        >>> F.one_hot(torch.arange(0, 5) % 3, num_classes=5)
        tensor([[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0]])
        >>> F.one_hot(torch.arange(0, 6).view(3,2) % 3)
        tensor([[[1, 0, 0],
                 [0, 1, 0]],
                [[0, 0, 1],
                 [1, 0, 0]],
                [[0, 1, 0],
                 [0, 0, 1]]])
    """

def pad(input: Tensor, pad: Sequence[int], mode: str = ..., value: float | None = ...) -> Tensor: ...
def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = ...,
    dropout_p: float = ...,
    is_causal: bool = ...,
    scale: float | None = ...,
    enable_gqa: bool = ...,
) -> Tensor:
    r"""
    scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> Tensor:

    Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed,
    and applying dropout if a probability greater than 0.0 is specified. The optional scale argument can only be
    specified as a keyword argument.

    .. code-block:: python

        # Efficient implementation equivalent to the following:
        def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
            L, S = query.size(-2), key.size(-2)
            scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
            attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
            if is_causal:
                assert attn_mask is None
                temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                else:
                    attn_bias = attn_mask + attn_bias

            if enable_gqa:
                key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
                value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

            attn_weight = query @ key.transpose(-2, -1) * scale_factor
            attn_weight += attn_bias
            attn_weight = torch.softmax(attn_weight, dim=-1)
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            return attn_weight @ value

    .. warning::
        This function is beta and subject to change.

    .. warning::
        This function always applies dropout according to the specified ``dropout_p`` argument.
        To disable dropout during evaluation, be sure to pass a value of ``0.0`` when the module
        that makes the function call is not in training mode.

        For example:

        .. code-block:: python

            class MyModel(nn.Module):
                def __init__(self, p=0.5):
                    super().__init__()
                    self.p = p

                def forward(self, ...):
                    return F.scaled_dot_product_attention(...,
                        dropout_p=(self.p if self.training else 0.0))

    Note:

        There are currently three supported implementations of scaled dot product attention:

            - `FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning`_
            - `Memory-Efficient Attention`_
            - A PyTorch implementation defined in C++ matching the above formulation

        The function may call optimized kernels for improved performance when using the CUDA backend.
        For all other backends, the PyTorch implementation will be used.

        All implementations are enabled by default. Scaled dot product attention attempts to automatically select the
        most optimal implementation based on the inputs. In order to provide more fine-grained control over what implementation
        is used, the following functions are provided for enabling and disabling implementations.
        The context manager is the preferred mechanism:

            - :func:`torch.nn.attention.sdpa_kernel`: A context manager used to enable or disable any of the implementations.
            - :func:`torch.backends.cuda.enable_flash_sdp`: Globally enables or disables FlashAttention.
            - :func:`torch.backends.cuda.enable_mem_efficient_sdp`: Globally enables or disables  Memory-Efficient Attention.
            - :func:`torch.backends.cuda.enable_math_sdp`: Globally enables or disables  the PyTorch C++ implementation.

        Each of the fused kernels has specific input limitations. If the user requires the use of a specific fused implementation,
        disable the PyTorch C++ implementation using :func:`torch.nn.attention.sdpa_kernel`.
        In the event that a fused implementation is not available, a warning will be raised with the
        reasons why the fused implementation cannot run.

        Due to the nature of fusing floating point operations, the output of this function may be different
        depending on what backend kernel is chosen.
        The c++ implementation supports torch.float64 and can be used when higher precision is required.
        For math backend, all intermediates are kept in torch.float if inputs are in torch.half or torch.bfloat16.
    For more information please see :doc:`/notes/numerical_accuracy`

        Grouped Query Attention (GQA) is an experimental feature. It currently works only for Flash_attention
        and math kernel on CUDA tensor, and does not support Nested tensor.
        Constraints for GQA:

            - number_of_heads_query % number_of_heads_key_value == 0 and,
            - number_of_heads_key == number_of_heads_value

    Note:

        In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.

    Args:
        query (Tensor): Query tensor; shape :math:`(N, ..., Hq, L, E)`.
        key (Tensor): Key tensor; shape :math:`(N, ..., H, S, E)`.
        value (Tensor): Value tensor; shape :math:`(N, ..., H, S, Ev)`.
        attn_mask (optional Tensor): Attention mask; shape must be broadcastable to the shape of attention weights,
            which is :math:`(N,..., L, S)`. Two types of masks are supported.
            A boolean mask where a value of True indicates that the element *should* take part in attention.
            A float mask of the same type as query, key, value that is added to the attention score.
        dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied
        is_causal (bool): If set to true, the attention masking is a lower triangular matrix when the mask is a
            square matrix. The attention masking has the form of the upper left causal bias due to the alignment
            (see :class:`torch.nn.attention.bias.CausalBias`) when the mask is a non-square matrix.
            An error is thrown if both attn_mask and is_causal are set.
        scale (optional float, keyword-only): Scaling factor applied prior to softmax. If None, the default value is set
            to :math:`\frac{1}{\sqrt{E}}`.
        enable_gqa (bool): If set to True, Grouped Query Attention (GQA) is enabled, by default it is set to False.

    Returns:
        output (Tensor): Attention output; shape :math:`(N, ..., Hq, L, Ev)`.

    Shape legend:
        - :math:`N: \text{Batch size} ... : \text{Any number of other batch dimensions (optional)}`
        - :math:`S: \text{Source sequence length}`
        - :math:`L: \text{Target sequence length}`
        - :math:`E: \text{Embedding dimension of the query and key}`
        - :math:`Ev: \text{Embedding dimension of the value}`
        - :math:`Hq: \text{Number of heads of query}`
        - :math:`H: \text{Number of heads of key and value}`

    Examples:

        >>> # Optionally use the context manager to ensure one of the fused kernels is run
        >>> query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        >>> key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        >>> value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        >>> with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        >>>     F.scaled_dot_product_attention(query,key,value)


        >>> # Sample for GQA for llama3
        >>> query = torch.rand(32, 32, 128, 64, dtype=torch.float16, device="cuda")
        >>> key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        >>> value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        >>> with sdpa_kernel(backends=[SDPBackend.MATH]):
        >>>     F.scaled_dot_product_attention(query,key,value,enable_gqa=True)


    .. _FlashAttention-2\: Faster Attention with Better Parallelism and Work Partitioning:
        https://arxiv.org/abs/2307.08691
    .. _Memory-Efficient Attention:
        https://github.com/facebookresearch/xformers
    .. _Grouped-Query Attention:
        https://arxiv.org/pdf/2305.13245
    """

def softplus(input: Tensor, beta: float = ..., threshold: float = ...) -> Tensor:
    r"""
    softplus(input, beta=1, threshold=20) -> Tensor

    Applies element-wise, the function :math:`\text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))`.

    For numerical stability the implementation reverts to the linear function
    when :math:`input \times \beta > threshold`.

    See :class:`~torch.nn.Softplus` for more details.
    """

def softshrink(input: Tensor, lambd: float = ...) -> Tensor:
    """
    softshrink(input, lambd=0.5) -> Tensor

    Applies the soft shrinkage function elementwise

    See :class:`~torch.nn.Softshrink` for more details.
    """

def mkldnn_linear(input: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor: ...
def mkldnn_reorder_conv2d_weight(
    self: Tensor, padding: list, stride: list, dilatation: list, groups: int
) -> Tensor: ...
def mkldnn_reorder_conv3d_weight(
    self: Tensor, padding: list, stride: list, dilatation: list, groups: int
) -> Tensor: ...
def mkldnn_prelu(input: Tensor, weight: Tensor) -> Tensor: ...
def pad_sequence(
    sequences: list[Tensor] | tuple[Tensor, ...],
    batch_first: bool = ...,
    padding_value: float = ...,
    padding_side: Literal["left", "right"] = ...,
) -> Tensor: ...
def upsample_nearest1d(
    input: Tensor, output_size: Sequence[int] | None, scale_factors: Sequence[float] | None
) -> Tensor: ...
def upsample_nearest2d(
    input: Tensor, output_size: Sequence[int] | None, scale_factors: Sequence[float] | None
) -> Tensor: ...
def upsample_nearest3d(
    input: Tensor, output_size: Sequence[int] | None, scale_factors: Sequence[float] | None
) -> Tensor: ...
def upsample_linear1d(
    input: Tensor, output_size: Sequence[int] | None, align_corners: bool, scale_factors: Sequence[float] | None
) -> Tensor: ...
def upsample_bilinear2d(
    input: Tensor, output_size: Sequence[int] | None, align_corners: bool, scale_factors: Sequence[float] | None
) -> Tensor: ...
def upsample_trilinear3d(
    input: Tensor, output_size: Sequence[int] | None, align_corners: bool, scale_factors: Sequence[float] | None
) -> Tensor: ...
def upsample_bicubic2d(
    input: Tensor, output_size: Sequence[int] | None, align_corners: bool, scale_factors: Sequence[float] | None
) -> Tensor: ...
def flatten_dense_tensors(tensors: list[Tensor]) -> Tensor: ...
def unflatten_dense_tensors(flat: Tensor, tensors: list[Tensor]) -> list[Tensor]: ...
