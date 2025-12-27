"""
The torch package contains data structures for multi-dimensional
tensors and defines mathematical operations over these tensors.
Additionally, it provides many utilities for efficient serialization of
Tensors and arbitrary types, and other useful utilities.

It has a CUDA counterpart, that enables you to run your tensor computations
on an NVIDIA GPU with compute capability >= 3.0.
"""

import builtins
import functools
import os
import platform
import sys
from collections.abc import Callable as Callable
from typing import TYPE_CHECKING, Any as Any, TypeIs as _TypeIs, overload as overload

import torch
from torch import (
    _C as _C,
    _VF as _VF,
    __config__ as __config__,
    __future__ as __future__,
    _awaits as _awaits,
    _dynamo as _dynamo,
    _inductor as _inductor,
    _library as _library,
    _ops as _ops,
    _subclasses as _subclasses,
    accelerator as accelerator,
    amp as amp,
    ao as ao,
    autograd as autograd,
    backends as backends,
    compiler as compiler,
    cpu as cpu,
    cuda as cuda,
    distributed as distributed,
    distributions as distributions,
    export as export,
    fft as fft,
    func as func,
    functional as functional,
    futures as futures,
    fx as fx,
    hub as hub,
    jit as jit,
    library as library,
    linalg as linalg,
    masked as masked,
    mps as mps,
    mtia as mtia,
    multiprocessing as multiprocessing,
    nested as nested,
    nn as nn,
    onnx as onnx,
    optim as optim,
    overrides as overrides,
    profiler as profiler,
    quantization as quantization,
    quasirandom as quasirandom,
    random as random,
    return_types as return_types,
    serialization as serialization,
    sparse as sparse,
    special as special,
    storage as storage,
    testing as testing,
    types as types,
    utils as utils,
    version as version,
    xpu as xpu,
)
from torch._C import (
    BoolTensor,
    ByteTensor,
    DispatchKey,
    DoubleTensor,
    FloatTensor,
    IntTensor,
    LongTensor,
    ShortTensor,
    Size,
    bfloat16,
    bool,
    device,
    dtype,
    finfo,
    float,
    float16,
    float32,
    float64,
    half,
    iinfo,
    int,
    int16,
    int32,
    int64,
    layout,
    long,
    memory_format,
    qscheme,
    short,
    uint8,
)
from torch._C._VariableFunctions import (
    __and__,
    __lshift__,
    __or__,
    __rshift__,
    __xor__,
    abs,
    abs_,
    absolute,
    acos,
    acos_,
    acosh,
    acosh_,
    adaptive_avg_pool1d,
    adaptive_max_pool1d,
    add,
    addbmm,
    addcdiv,
    addcmul,
    addmm,
    addmv,
    addmv_,
    addr,
    adjoint,
    affine_grid_generator,
    alias_copy,
    all,
    allclose,
    alpha_dropout,
    alpha_dropout_,
    amax,
    amin,
    aminmax,
    angle,
    any,
    arange,
    arccos,
    arccos_,
    arccosh,
    arccosh_,
    arcsin,
    arcsin_,
    arcsinh,
    arcsinh_,
    arctan,
    arctan2,
    arctan_,
    arctanh,
    arctanh_,
    argmax,
    argmin,
    argsort,
    argwhere,
    as_strided,
    as_strided_,
    as_strided_copy,
    as_strided_scatter,
    as_tensor,
    asarray,
    asin,
    asin_,
    asinh,
    asinh_,
    atan,
    atan2,
    atan_,
    atanh,
    atanh_,
    avg_pool1d,
    baddbmm,
    bartlett_window,
    batch_norm,
    batch_norm_backward_elemt,
    batch_norm_backward_reduce,
    batch_norm_elemt,
    batch_norm_gather_stats,
    batch_norm_gather_stats_with_counts,
    batch_norm_stats,
    batch_norm_update_stats,
    bernoulli,
    bilinear,
    binary_cross_entropy_with_logits,
    bincount,
    binomial,
    bitwise_and,
    bitwise_left_shift,
    bitwise_not,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    blackman_window,
    bmm,
    broadcast_to,
    bucketize,
    can_cast,
    cat,
    ccol_indices_copy,
    ceil,
    ceil_,
    celu,
    celu_,
    channel_shuffle,
    cholesky,
    cholesky_inverse,
    cholesky_solve,
    choose_qparams_optimized,
    chunk,
    clamp,
    clamp_,
    clamp_max,
    clamp_max_,
    clamp_min,
    clamp_min_,
    clip,
    clip_,
    clone,
    col_indices_copy,
    column_stack,
    combinations,
    complex,
    concat,
    concatenate,
    conj,
    conj_physical,
    conj_physical_,
    constant_pad_nd,
    conv1d,
    conv2d,
    conv3d,
    conv_tbc,
    conv_transpose1d,
    conv_transpose2d,
    conv_transpose3d,
    convolution,
    copysign,
    corrcoef,
    cos,
    cos_,
    cosh,
    cosh_,
    cosine_embedding_loss,
    cosine_similarity,
    count_nonzero,
    cov,
    cross,
    crow_indices_copy,
    ctc_loss,
    cudnn_affine_grid_generator,
    cudnn_batch_norm,
    cudnn_convolution,
    cudnn_convolution_add_relu,
    cudnn_convolution_relu,
    cudnn_convolution_transpose,
    cudnn_grid_sampler,
    cudnn_is_acceptable,
    cummax,
    cummin,
    cumprod,
    cumsum,
    cumulative_trapezoid,
    deg2rad,
    deg2rad_,
    dequantize,
    det,
    detach,
    detach_,
    detach_copy,
    diag,
    diag_embed,
    diagflat,
    diagonal,
    diagonal_copy,
    diagonal_scatter,
    diff,
    digamma,
    dist,
    div,
    divide,
    dot,
    dropout,
    dropout_,
    dsmm,
    dsplit,
    dstack,
    embedding,
    embedding_bag,
    embedding_renorm_,
    empty,
    empty_like,
    empty_permuted,
    empty_quantized,
    empty_strided,
    eq,
    equal,
    erf,
    erf_,
    erfc,
    erfc_,
    erfinv,
    exp,
    exp2,
    exp2_,
    exp_,
    expand_copy,
    expm1,
    expm1_,
    eye,
    fake_quantize_per_channel_affine,
    fake_quantize_per_tensor_affine,
    fbgemm_linear_fp16_weight,
    fbgemm_linear_fp16_weight_fp32_activation,
    fbgemm_linear_int8_weight,
    fbgemm_linear_int8_weight_fp32_activation,
    fbgemm_linear_quantize_weight,
    fbgemm_pack_gemm_matrix_fp16,
    fbgemm_pack_quantized_matrix,
    feature_alpha_dropout,
    feature_alpha_dropout_,
    feature_dropout,
    feature_dropout_,
    fill,
    fill_,
    fix,
    fix_,
    flatten,
    flip,
    fliplr,
    flipud,
    float_power,
    floor,
    floor_,
    floor_divide,
    fmax,
    fmin,
    fmod,
    frac,
    frac_,
    frexp,
    frobenius_norm,
    from_file,
    from_numpy,
    frombuffer,
    full,
    full_like,
    fused_moving_avg_obs_fake_quant,
    gather,
    gcd,
    gcd_,
    ge,
    geqrf,
    ger,
    get_default_dtype,
    get_num_interop_threads,
    get_num_threads,
    gradient,
    greater,
    greater_equal,
    grid_sampler,
    grid_sampler_2d,
    grid_sampler_3d,
    group_norm,
    gru,
    gru_cell,
    gt,
    hamming_window,
    hann_window,
    hardshrink,
    hash_tensor,
    heaviside,
    hinge_embedding_loss,
    histc,
    histogram,
    histogramdd,
    hsmm,
    hsplit,
    hspmm,
    hstack,
    hypot,
    i0,
    i0_,
    igamma,
    igammac,
    imag,
    index_add,
    index_copy,
    index_fill,
    index_put,
    index_put_,
    index_reduce,
    index_select,
    indices_copy,
    init_num_threads,
    inner,
    instance_norm,
    int_repr,
    inverse,
    is_complex,
    is_conj,
    is_distributed,
    is_floating_point,
    is_grad_enabled,
    is_inference,
    is_inference_mode_enabled,
    is_neg,
    is_nonzero,
    is_same_size,
    is_signed,
    is_vulkan_available,
    isclose,
    isfinite,
    isin,
    isinf,
    isnan,
    isneginf,
    isposinf,
    isreal,
    istft,
    kaiser_window,
    kl_div,
    kron,
    kthvalue,
    layer_norm,
    lcm,
    lcm_,
    ldexp,
    ldexp_,
    le,
    lerp,
    less,
    less_equal,
    lgamma,
    linspace,
    log,
    log1p,
    log1p_,
    log2,
    log2_,
    log10,
    log10_,
    log_,
    log_softmax,
    logaddexp,
    logaddexp2,
    logcumsumexp,
    logdet,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    logit,
    logit_,
    logspace,
    logsumexp,
    lstm,
    lstm_cell,
    lt,
    lu_solve,
    lu_unpack,
    margin_ranking_loss,
    masked_fill,
    masked_scatter,
    masked_select,
    matmul,
    matrix_exp,
    matrix_power,
    max,
    max_pool1d,
    max_pool1d_with_indices,
    max_pool2d,
    max_pool3d,
    maximum,
    mean,
    median,
    min,
    minimum,
    miopen_batch_norm,
    miopen_convolution,
    miopen_convolution_add_relu,
    miopen_convolution_relu,
    miopen_convolution_transpose,
    miopen_depthwise_convolution,
    miopen_rnn,
    mkldnn_adaptive_avg_pool2d,
    mkldnn_convolution,
    mkldnn_linear_backward_weights,
    mkldnn_max_pool2d,
    mkldnn_max_pool3d,
    mkldnn_rnn_layer,
    mm,
    mode,
    moveaxis,
    movedim,
    msort,
    mul,
    multinomial,
    multiply,
    mv,
    mvlgamma,
    nan_to_num,
    nan_to_num_,
    nanmean,
    nanmedian,
    nanquantile,
    nansum,
    narrow,
    narrow_copy,
    native_batch_norm,
    native_channel_shuffle,
    native_dropout,
    native_group_norm,
    native_layer_norm,
    native_norm,
    ne,
    neg,
    neg_,
    negative,
    negative_,
    nextafter,
    nonzero,
    nonzero_static,
    norm_except_dim,
    normal,
    not_equal,
    nuclear_norm,
    numel,
    ones,
    ones_like,
    orgqr,
    ormqr,
    outer,
    pairwise_distance,
    pdist,
    permute,
    permute_copy,
    pinverse,
    pixel_shuffle,
    pixel_unshuffle,
    poisson,
    poisson_nll_loss,
    polar,
    polygamma,
    positive,
    pow,
    prelu,
    prod,
    promote_types,
    put,
    q_per_channel_axis,
    q_per_channel_scales,
    q_per_channel_zero_points,
    q_scale,
    q_zero_point,
    qr,
    quantile,
    quantize_per_channel,
    quantize_per_tensor,
    quantize_per_tensor_dynamic,
    quantized_batch_norm,
    quantized_gru_cell,
    quantized_lstm_cell,
    quantized_max_pool1d,
    quantized_max_pool2d,
    quantized_max_pool3d,
    quantized_rnn_relu_cell,
    quantized_rnn_tanh_cell,
    rad2deg,
    rad2deg_,
    rand,
    rand_like,
    randint,
    randint_like,
    randn,
    randn_like,
    randperm,
    range,
    ravel,
    real,
    reciprocal,
    reciprocal_,
    relu,
    relu_,
    remainder,
    renorm,
    repeat_interleave,
    reshape,
    resize_as_,
    resize_as_sparse_,
    resolve_conj,
    resolve_neg,
    result_type,
    rms_norm,
    rnn_relu,
    rnn_relu_cell,
    rnn_tanh,
    rnn_tanh_cell,
    roll,
    rot90,
    round,
    round_,
    row_indices_copy,
    row_stack,
    rrelu,
    rrelu_,
    rsqrt,
    rsqrt_,
    rsub,
    saddmm,
    scalar_tensor,
    scatter,
    scatter_add,
    scatter_reduce,
    searchsorted,
    segment_reduce,
    select,
    select_copy,
    select_scatter,
    selu,
    selu_,
    set_flush_denormal,
    set_num_interop_threads,
    set_num_threads,
    sgn,
    sigmoid,
    sigmoid_,
    sign,
    signbit,
    sin,
    sin_,
    sinc,
    sinc_,
    sinh,
    sinh_,
    slice_copy,
    slice_inverse,
    slice_scatter,
    slogdet,
    smm,
    softmax,
    sort,
    sparse_bsc_tensor,
    sparse_bsr_tensor,
    sparse_compressed_tensor,
    sparse_coo_tensor,
    sparse_csc_tensor,
    sparse_csr_tensor,
    split_copy,
    split_with_sizes,
    split_with_sizes_copy,
    spmm,
    sqrt,
    sqrt_,
    square,
    square_,
    squeeze,
    squeeze_copy,
    sspaddmm,
    stack,
    std,
    std_mean,
    sub,
    subtract,
    sum,
    svd,
    swapaxes,
    swapdims,
    sym_constrain_range,
    sym_constrain_range_for_size,
    t,
    t_copy,
    take,
    take_along_dim,
    tan,
    tan_,
    tanh,
    tanh_,
    tensor,
    tensor_split,
    threshold,
    threshold_,
    tile,
    topk,
    trace,
    transpose,
    transpose_copy,
    trapezoid,
    trapz,
    triangular_solve,
    tril,
    tril_indices,
    triplet_margin_loss,
    triu,
    triu_indices,
    true_divide,
    trunc,
    trunc_,
    unbind,
    unbind_copy,
    unflatten,
    unfold_copy,
    unique_dim,
    unsafe_chunk,
    unsafe_split,
    unsafe_split_with_sizes,
    unsqueeze,
    unsqueeze_copy,
    values_copy,
    vander,
    var,
    var_mean,
    vdot,
    view_as_complex,
    view_as_complex_copy,
    view_as_real,
    view_as_real_copy,
    view_copy,
    vsplit,
    vstack,
    where,
    xlogy,
    xlogy_,
    zero_,
    zeros,
    zeros_like,
)
from torch._classes import classes as classes
from torch._higher_order_ops import cond as cond, while_loop as while_loop
from torch._lobpcg import lobpcg as lobpcg
from torch._ops import ops as ops
from torch._tensor import Tensor
from torch._tensor_str import set_printoptions
from torch._utils import classproperty
from torch._utils_internal import USE_RTLD_GLOBAL_WITH_LIBTORCH
from torch.amp import GradScaler, autocast
from torch.autograd import (
    enable_grad as enable_grad,
    inference_mode as inference_mode,
    no_grad as no_grad,
    set_grad_enabled as set_grad_enabled,
)
from torch.func import vmap as vmap
from torch.functional import *
from torch.random import get_rng_state, initial_seed, manual_seed, seed, set_rng_state
from torch.serialization import load, save
from torch.signal import windows as windows
from torch.storage import TypedStorage, UntypedStorage, _LegacyStorage
from torch.torch_version import __version__ as __version__
from torch.types import Device, IntLikeType

__all__ = [
    "BoolStorage",
    "BoolTensor",
    "ByteStorage",
    "ByteTensor",
    "CharStorage",
    "CharTensor",
    "DispatchKey",
    "DoubleStorage",
    "DoubleTensor",
    "FloatStorage",
    "FloatTensor",
    "GradScaler",
    "IntStorage",
    "IntTensor",
    "LongStorage",
    "LongTensor",
    "ShortStorage",
    "ShortTensor",
    "Size",
    "SymBool",
    "SymFloat",
    "SymInt",
    "Tensor",
    "TypedStorage",
    "UntypedStorage",
    "__and__",
    "__lshift__",
    "__or__",
    "__rshift__",
    "__xor__",
    "abs",
    "abs_",
    "absolute",
    "acos",
    "acos_",
    "acosh",
    "acosh_",
    "adaptive_avg_pool1d",
    "adaptive_max_pool1d",
    "add",
    "addbmm",
    "addcdiv",
    "addcmul",
    "addmm",
    "addmv",
    "addmv_",
    "addr",
    "adjoint",
    "affine_grid_generator",
    "alias_copy",
    "all",
    "allclose",
    "alpha_dropout",
    "alpha_dropout_",
    "amax",
    "amin",
    "aminmax",
    "angle",
    "any",
    "arange",
    "arccos",
    "arccos_",
    "arccosh",
    "arccosh_",
    "arcsin",
    "arcsin_",
    "arcsinh",
    "arcsinh_",
    "arctan",
    "arctan2",
    "arctan_",
    "arctanh",
    "arctanh_",
    "are_deterministic_algorithms_enabled",
    "argmax",
    "argmin",
    "argsort",
    "argwhere",
    "as_strided",
    "as_strided_",
    "as_strided_copy",
    "as_strided_scatter",
    "as_tensor",
    "asarray",
    "asin",
    "asin_",
    "asinh",
    "asinh_",
    "atan",
    "atan2",
    "atan_",
    "atanh",
    "atanh_",
    "autocast",
    "avg_pool1d",
    "baddbmm",
    "bartlett_window",
    "batch_norm",
    "batch_norm_backward_elemt",
    "batch_norm_backward_reduce",
    "batch_norm_elemt",
    "batch_norm_gather_stats",
    "batch_norm_gather_stats_with_counts",
    "batch_norm_stats",
    "batch_norm_update_stats",
    "bernoulli",
    "bfloat16",
    "bilinear",
    "binary_cross_entropy_with_logits",
    "bincount",
    "binomial",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "blackman_window",
    "bmm",
    "bool",
    "broadcast_to",
    "bucketize",
    "can_cast",
    "cat",
    "ccol_indices_copy",
    "ceil",
    "ceil_",
    "celu",
    "celu_",
    "channel_shuffle",
    "cholesky",
    "cholesky_inverse",
    "cholesky_solve",
    "choose_qparams_optimized",
    "chunk",
    "chunk",
    "clamp",
    "clamp_",
    "clamp_max",
    "clamp_max_",
    "clamp_min",
    "clamp_min_",
    "clip",
    "clip_",
    "clone",
    "col_indices_copy",
    "column_stack",
    "combinations",
    "compile",
    "complex",
    "concat",
    "concatenate",
    "cond",
    "conj",
    "conj_physical",
    "conj_physical_",
    "constant_pad_nd",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_tbc",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "convolution",
    "copysign",
    "corrcoef",
    "cos",
    "cos_",
    "cosh",
    "cosh_",
    "cosine_embedding_loss",
    "cosine_similarity",
    "count_nonzero",
    "cov",
    "cross",
    "crow_indices_copy",
    "ctc_loss",
    "cudnn_affine_grid_generator",
    "cudnn_batch_norm",
    "cudnn_convolution",
    "cudnn_convolution_add_relu",
    "cudnn_convolution_relu",
    "cudnn_convolution_transpose",
    "cudnn_grid_sampler",
    "cudnn_is_acceptable",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "cumulative_trapezoid",
    "deg2rad",
    "deg2rad_",
    "dequantize",
    "det",
    "detach",
    "detach_",
    "detach_copy",
    "device",
    "device",
    "diag",
    "diag_embed",
    "diagflat",
    "diagonal",
    "diagonal_copy",
    "diagonal_scatter",
    "diff",
    "digamma",
    "dist",
    "div",
    "divide",
    "dot",
    "dropout",
    "dropout_",
    "dsmm",
    "dsplit",
    "dstack",
    "dtype",
    "dtype",
    "embedding",
    "embedding_bag",
    "embedding_renorm_",
    "empty",
    "empty_like",
    "empty_permuted",
    "empty_quantized",
    "empty_strided",
    "enable_grad",
    "eq",
    "equal",
    "erf",
    "erf_",
    "erfc",
    "erfc_",
    "erfinv",
    "exp",
    "exp2",
    "exp2_",
    "exp_",
    "expand_copy",
    "expm1",
    "expm1_",
    "export",
    "eye",
    "fake_quantize_per_channel_affine",
    "fake_quantize_per_tensor_affine",
    "fbgemm_linear_fp16_weight",
    "fbgemm_linear_fp16_weight_fp32_activation",
    "fbgemm_linear_int8_weight",
    "fbgemm_linear_int8_weight_fp32_activation",
    "fbgemm_linear_quantize_weight",
    "fbgemm_pack_gemm_matrix_fp16",
    "fbgemm_pack_quantized_matrix",
    "feature_alpha_dropout",
    "feature_alpha_dropout_",
    "feature_dropout",
    "feature_dropout_",
    "fill",
    "fill_",
    "finfo",
    "fix",
    "fix_",
    "flatten",
    "flip",
    "fliplr",
    "flipud",
    "float",
    "float16",
    "float32",
    "float64",
    "float_power",
    "floor",
    "floor_",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "frac",
    "frac_",
    "frexp",
    "frobenius_norm",
    "from_file",
    "from_numpy",
    "frombuffer",
    "full",
    "full_like",
    "fused_moving_avg_obs_fake_quant",
    "gather",
    "gcd",
    "gcd_",
    "ge",
    "geqrf",
    "ger",
    "get_default_device",
    "get_default_dtype",
    "get_deterministic_debug_mode",
    "get_device_module",
    "get_float32_matmul_precision",
    "get_num_interop_threads",
    "get_num_threads",
    "get_rng_state",
    "gradient",
    "greater",
    "greater_equal",
    "grid_sampler",
    "grid_sampler_2d",
    "grid_sampler_3d",
    "group_norm",
    "gru",
    "gru_cell",
    "gt",
    "half",
    "hamming_window",
    "hann_window",
    "hardshrink",
    "hash_tensor",
    "heaviside",
    "hinge_embedding_loss",
    "histc",
    "histogram",
    "histogramdd",
    "hsmm",
    "hsplit",
    "hspmm",
    "hstack",
    "hypot",
    "i0",
    "i0_",
    "igamma",
    "igammac",
    "iinfo",
    "imag",
    "index_add",
    "index_copy",
    "index_fill",
    "index_put",
    "index_put_",
    "index_reduce",
    "index_select",
    "indices_copy",
    "inference_mode",
    "init_num_threads",
    "initial_seed",
    "inner",
    "instance_norm",
    "int",
    "int16",
    "int32",
    "int64",
    "int_repr",
    "inverse",
    "is_complex",
    "is_conj",
    "is_deterministic_algorithms_warn_only_enabled",
    "is_distributed",
    "is_floating_point",
    "is_grad_enabled",
    "is_inference",
    "is_inference_mode_enabled",
    "is_neg",
    "is_nonzero",
    "is_same_size",
    "is_signed",
    "is_storage",
    "is_tensor",
    "is_vulkan_available",
    "is_warn_always_enabled",
    "isclose",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "isreal",
    "istft",
    "kaiser_window",
    "kl_div",
    "kron",
    "kthvalue",
    "layer_norm",
    "layout",
    "lcm",
    "lcm_",
    "ldexp",
    "ldexp_",
    "le",
    "lerp",
    "less",
    "less_equal",
    "lgamma",
    "linspace",
    "load",
    "lobpcg",
    "log",
    "log1p",
    "log1p_",
    "log2",
    "log2_",
    "log10",
    "log10_",
    "log_",
    "log_softmax",
    "logaddexp",
    "logaddexp2",
    "logcumsumexp",
    "logdet",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logit",
    "logit_",
    "logspace",
    "logsumexp",
    "long",
    "lstm",
    "lstm_cell",
    "lt",
    "lu_solve",
    "lu_unpack",
    "manual_seed",
    "margin_ranking_loss",
    "masked_fill",
    "masked_scatter",
    "masked_select",
    "matmul",
    "matmul",
    "matrix_exp",
    "matrix_power",
    "max",
    "max_pool1d",
    "max_pool1d_with_indices",
    "max_pool2d",
    "max_pool3d",
    "maximum",
    "mean",
    "median",
    "memory_format",
    "min",
    "minimum",
    "miopen_batch_norm",
    "miopen_convolution",
    "miopen_convolution_add_relu",
    "miopen_convolution_relu",
    "miopen_convolution_transpose",
    "miopen_depthwise_convolution",
    "miopen_rnn",
    "mkldnn_adaptive_avg_pool2d",
    "mkldnn_convolution",
    "mkldnn_linear_backward_weights",
    "mkldnn_max_pool2d",
    "mkldnn_max_pool3d",
    "mkldnn_rnn_layer",
    "mm",
    "mode",
    "moveaxis",
    "movedim",
    "msort",
    "mul",
    "multinomial",
    "multiply",
    "mv",
    "mvlgamma",
    "nan_to_num",
    "nan_to_num_",
    "nanmean",
    "nanmedian",
    "nanquantile",
    "nansum",
    "narrow",
    "narrow_copy",
    "native_batch_norm",
    "native_channel_shuffle",
    "native_dropout",
    "native_group_norm",
    "native_layer_norm",
    "native_norm",
    "ne",
    "neg",
    "neg_",
    "negative",
    "negative_",
    "nextafter",
    "no_grad",
    "nonzero",
    "nonzero_static",
    "norm_except_dim",
    "normal",
    "not_equal",
    "nuclear_norm",
    "numel",
    "ones",
    "ones_like",
    "orgqr",
    "ormqr",
    "outer",
    "pairwise_distance",
    "pdist",
    "permute",
    "permute_copy",
    "pinverse",
    "pixel_shuffle",
    "pixel_unshuffle",
    "poisson",
    "poisson_nll_loss",
    "polar",
    "polygamma",
    "positive",
    "pow",
    "prelu",
    "prod",
    "promote_types",
    "put",
    "q_per_channel_axis",
    "q_per_channel_scales",
    "q_per_channel_zero_points",
    "q_scale",
    "q_zero_point",
    "qr",
    "qscheme",
    "quantile",
    "quantize_per_channel",
    "quantize_per_tensor",
    "quantize_per_tensor_dynamic",
    "quantized_batch_norm",
    "quantized_gru_cell",
    "quantized_lstm_cell",
    "quantized_max_pool1d",
    "quantized_max_pool2d",
    "quantized_max_pool3d",
    "quantized_rnn_relu_cell",
    "quantized_rnn_tanh_cell",
    "rad2deg",
    "rad2deg_",
    "rand",
    "rand",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn",
    "randn_like",
    "randperm",
    "range",
    "ravel",
    "real",
    "reciprocal",
    "reciprocal_",
    "relu",
    "relu_",
    "remainder",
    "renorm",
    "repeat_interleave",
    "reshape",
    "resize_as_",
    "resize_as_sparse_",
    "resolve_conj",
    "resolve_neg",
    "result_type",
    "rms_norm",
    "rnn_relu",
    "rnn_relu_cell",
    "rnn_tanh",
    "rnn_tanh_cell",
    "roll",
    "rot90",
    "round",
    "round_",
    "row_indices_copy",
    "row_stack",
    "rrelu",
    "rrelu_",
    "rsqrt",
    "rsqrt_",
    "rsub",
    "saddmm",
    "save",
    "scalar_tensor",
    "scatter",
    "scatter_add",
    "scatter_reduce",
    "searchsorted",
    "seed",
    "segment_reduce",
    "select",
    "select_copy",
    "select_scatter",
    "selu",
    "selu_",
    "set_default_device",
    "set_default_tensor_type",
    "set_deterministic_debug_mode",
    "set_float32_matmul_precision",
    "set_flush_denormal",
    "set_num_interop_threads",
    "set_num_threads",
    "set_printoptions",
    "set_rng_state",
    "set_warn_always",
    "sgn",
    "short",
    "sigmoid",
    "sigmoid_",
    "sign",
    "signbit",
    "sin",
    "sin_",
    "sinc",
    "sinc_",
    "sinh",
    "sinh_",
    "slice_copy",
    "slice_inverse",
    "slice_scatter",
    "slogdet",
    "smm",
    "softmax",
    "sort",
    "sparse_bsc_tensor",
    "sparse_bsr_tensor",
    "sparse_compressed_tensor",
    "sparse_coo_tensor",
    "sparse_csc_tensor",
    "sparse_csr_tensor",
    "split",
    "split_copy",
    "split_with_sizes",
    "split_with_sizes_copy",
    "spmm",
    "sqrt",
    "sqrt_",
    "square",
    "square_",
    "squeeze",
    "squeeze_copy",
    "sspaddmm",
    "stack",
    "stack",
    "std",
    "std_mean",
    "sub",
    "subtract",
    "sum",
    "svd",
    "swapaxes",
    "swapdims",
    "sym_constrain_range",
    "sym_constrain_range_for_size",
    "sym_float",
    "sym_fresh_size",
    "sym_int",
    "sym_ite",
    "sym_max",
    "sym_min",
    "sym_not",
    "sym_sum",
    "t",
    "t_copy",
    "take",
    "take_along_dim",
    "tan",
    "tan_",
    "tanh",
    "tanh_",
    "tensor",
    "tensor_split",
    "threshold",
    "threshold_",
    "tile",
    "topk",
    "trace",
    "transpose",
    "transpose_copy",
    "trapezoid",
    "trapz",
    "triangular_solve",
    "tril",
    "tril_indices",
    "triplet_margin_loss",
    "triu",
    "triu_indices",
    "true_divide",
    "trunc",
    "trunc_",
    "typename",
    "uint8",
    "unbind",
    "unbind_copy",
    "unflatten",
    "unfold_copy",
    "unique_dim",
    "unravel_index",
    "unsafe_chunk",
    "unsafe_split",
    "unsafe_split_with_sizes",
    "unsqueeze",
    "unsqueeze_copy",
    "use_deterministic_algorithms",
    "values_copy",
    "vander",
    "var",
    "var_mean",
    "vdot",
    "view_as_complex",
    "view_as_complex_copy",
    "view_as_real",
    "view_as_real_copy",
    "view_copy",
    "vmap",
    "void",
    "vsplit",
    "vstack",
    "where",
    "xlogy",
    "xlogy_",
    "zero_",
    "zeros",
    "zeros_like",
]
if sys.platform == "win32": ...
if (USE_RTLD_GLOBAL_WITH_LIBTORCH or os.getenv("TORCH_USE_RTLD_GLOBAL")) and (platform.system() != "Windows"):
    old_flags = ...

class SymInt:
    """
    Like an int (including magic methods), but redirects all operations on the
    wrapped node. This is used in particular to symbolically record operations
    in the symbolic shape workflow.
    """
    def __init__(self, node) -> None: ...
    def __bool__(self) -> bool: ...
    def __int__(self) -> int: ...
    def __index__(self) -> int: ...
    def __round__(self, ndigits=...) -> Self: ...
    def __truediv__(self, other) -> SymFloat | Any | _NotImplementedType: ...
    def __rtruediv__(self, other) -> SymFloat | Any | _NotImplementedType: ...
    def __floordiv__(self, other) -> Any | SymFloat | float | _NotImplementedType: ...
    def __rfloordiv__(self, other) -> Any | SymFloat | float | _NotImplementedType: ...
    def __pow__(self, other) -> _NotImplementedType | SymFloat | Any | SymInt: ...
    def __rpow__(self, other) -> _NotImplementedType | SymFloat | Any | SymInt: ...
    def __eq__(self, other: object) -> builtins.bool: ...
    def __lt__(self, other) -> builtins.bool: ...
    def __gt__(self, other) -> builtins.bool: ...
    def __le__(self, other) -> builtins.bool: ...
    def __ge__(self, other) -> builtins.bool: ...
    def __add__(self, other) -> SymInt: ...
    def __radd__(self, other) -> SymInt: ...
    def __rmul__(self, other) -> SymInt: ...
    def __mod__(self, other: IntLikeType) -> SymInt: ...
    def __mul__(self, other) -> SymInt: ...
    def __pow_by_natural__(self, other) -> SymInt: ...
    def __rpow_by_natural__(self, other) -> SymInt: ...
    def __int_truediv__(self, other) -> SymFloat: ...
    def __rint_truediv__(self, other) -> SymFloat: ...
    def __int_floordiv__(self, other) -> SymFloat: ...
    def __rint_floordiv__(self, other) -> SymFloat: ...
    def __sym_max__(self, other): ...
    def __sym_min__(self, other): ...
    def __sym_float__(self): ...
    def __neg__(self): ...
    def __sub__(self, other: IntLikeType) -> SymInt: ...
    def __rsub__(self, other: IntLikeType) -> SymInt: ...
    def __and__(self, other) -> SymInt: ...
    def __or__(self, other) -> SymInt: ...
    def __hash__(self) -> builtins.int: ...
    def as_integer_ratio(self) -> tuple[SymInt, builtins.int]:
        """Represent this int as an exact integer ratio"""
    def bit_length(self) -> builtins.int: ...
    def conjugate(self) -> SymInt: ...

class SymFloat:
    """
    Like a float (including magic methods), but redirects all operations on the
    wrapped node. This is used in particular to symbolically record operations
    in the symbolic shape workflow.
    """
    def __init__(self, node) -> None: ...
    def __truediv__(self, other) -> _NotImplementedType | SymFloat: ...
    def __rtruediv__(self, other) -> _NotImplementedType | SymFloat: ...
    def __floordiv__(self, other) -> _NotImplementedType | Any | SymFloat | float: ...
    def __rfloordiv__(self, other) -> _NotImplementedType | Any | SymFloat | float: ...
    def __bool__(self) -> bool: ...
    def __float__(self) -> float: ...
    def __pow__(self, other) -> _NotImplementedType | SymFloat: ...
    def __rpow__(self, other) -> _NotImplementedType | SymFloat: ...
    def __eq__(self, other: object) -> builtins.bool: ...
    def __lt__(self, other) -> builtins.bool: ...
    def __gt__(self, other) -> builtins.bool: ...
    def __le__(self, other) -> builtins.bool: ...
    def __ge__(self, other) -> builtins.bool: ...
    def __float_pow__(self, other) -> SymFloat: ...
    def __rfloat_pow__(self, other) -> SymFloat: ...
    def __float_truediv__(self, other) -> SymFloat: ...
    def __rfloat_truediv__(self, other) -> SymFloat: ...
    def __trunc__(self): ...
    def __sym_max__(self, other): ...
    def __sym_min__(self, other): ...
    def __sym_int__(self): ...
    def is_integer(self):
        """Return True if the float is an integer."""
    def as_integer_ratio(self) -> tuple[builtins.int, builtins.int]:
        """Represent this float as an exact integer ratio"""
    def __hash__(self) -> int: ...
    def conjugate(self) -> SymFloat:
        """Returns the complex conjugate of the float."""
    def hex(self) -> str:
        """Returns the hexadecimal representation of the float."""

class SymBool:
    """
    Like a bool (including magic methods), but redirects all operations on the
    wrapped node. This is used in particular to symbolically record operations
    in the symbolic shape workflow.

    Unlike regular bools, regular boolean operators will force extra guards instead
    of symbolically evaluate.  Use the bitwise operators instead to handle this.
    """
    def __init__(self, node) -> None: ...
    def __bool__(self) -> bool: ...
    def __int__(self) -> int: ...
    def __and__(self, other) -> SymBool: ...
    def __or__(self, other) -> SymBool: ...
    def __sym_not__(self) -> SymBool: ...
    def __sym_ite__(self, then_val, else_val): ...
    def __eq__(self, other) -> builtins.bool: ...
    def __hash__(self) -> int: ...

def sym_not(a) -> Any | bool:
    """
    SymInt-aware utility for logical negation.

    Args:
        a (SymBool or bool): Object to negate
    """

def sym_float(a) -> Any | SymFloat | float:
    """
    SymInt-aware utility for float casting.

    Args:
        a (SymInt, SymFloat, or object): Object to cast
    """

def sym_int(a) -> Any | SymInt | int:
    """
    SymInt-aware utility for int casting.

    Args:
        a (SymInt, SymFloat, or object): Object to cast
    """

def sym_max(a, b) -> Any | float:
    """
    SymInt-aware utility for max which avoids branching on a < b.
    Unlike builtins.max(), this only works for int/float, and it always
    promotes to float if any argument is float (unlike builtins.max, which
    will faithfully preserve the type of the input argument).
    """

def sym_min(a, b) -> Any | float:
    """SymInt-aware utility for min()."""

def sym_sum(args) -> Any | int | float | bool | SymInt | SymFloat | SymBool:
    """
    N-ary add which is faster to compute for long lists than iterated binary
    addition.  Only does something special for integers.
    """

sym_sqrt = ...

def sym_ite(b, t, f) -> Any:
    """SymInt-aware utility for ternary operator (``t if b else f``.)"""

def sym_fresh_size(expr) -> Number: ...

if not TYPE_CHECKING: ...

def typename(obj: Any, /) -> str:
    """
    String representation of the type of an object.

    This function returns a fully qualified string representation of an object's type.
    Args:
        obj (object): The object whose type to represent
    Returns:
        str: the type of the object `o`
    Example:
        >>> x = torch.tensor([1, 2, 3])
        >>> torch.typename(x)
        'torch.LongTensor'
        >>> torch.typename(torch.nn.Parameter)
        'torch.nn.parameter.Parameter'
    """

def is_tensor(obj: Any, /) -> _TypeIs[torch.Tensor]:
    """
    Returns True if `obj` is a PyTorch tensor.

    Note that this function is simply doing ``isinstance(obj, Tensor)``.
    Using that ``isinstance`` check is better for type checking with mypy,
    and more explicit - so it's recommended to use that instead of
    ``is_tensor``.

    Args:
        obj (object): Object to test
    Example::

        >>> x = torch.tensor([1, 2, 3])
        >>> torch.is_tensor(x)
        True
    """

def is_storage(obj: Any, /) -> _TypeIs[TypedStorage | UntypedStorage]:
    """
    Returns True if `obj` is a PyTorch storage object.

    Args:
        obj (Object): Object to test
    """

_GLOBAL_DEVICE_CONTEXT = ...

def get_default_device() -> torch.device:
    """Gets the default ``torch.Tensor`` to be allocated on ``device``"""

def set_default_device(device: Device) -> None:
    """
    Sets the default ``torch.Tensor`` to be allocated on ``device``.  This
    does not affect factory function calls which are called with an explicit
    ``device`` argument.  Factory calls will be performed as if they
    were passed ``device`` as an argument.

    To only temporarily change the default device instead of setting it
    globally, use ``with torch.device(device):`` instead.

    The default device is initially ``cpu``.  If you set the default tensor
    device to another device (e.g., ``cuda``) without a device index, tensors
    will be allocated on whatever the current device for the device type,
    even after :func:`torch.cuda.set_device` is called.

    .. warning::

        This function imposes a slight performance cost on every Python
        call to the torch API (not just factory functions).  If this
        is causing problems for you, please comment on
        https://github.com/pytorch/pytorch/issues/92701

    .. note::

        This doesn't affect functions that create tensors that share the same memory as the input, like:
        :func:`torch.from_numpy` and :func:`torch.frombuffer`

    Args:
        device (device or string): the device to set as default

    Example::

        >>> # xdoctest: +SKIP("requires cuda, changes global state")
        >>> torch.get_default_device()
        device(type='cpu')
        >>> torch.set_default_device('cuda')  # current device is 0
        >>> torch.get_default_device()
        device(type='cuda', index=0)
        >>> torch.set_default_device('cuda')
        >>> torch.cuda.set_device('cuda:1')  # current device is 1
        >>> torch.get_default_device()
        device(type='cuda', index=1)
        >>> torch.set_default_device('cuda:1')
        >>> torch.get_default_device()
        device(type='cuda', index=1)
    """

def set_default_tensor_type(t: type[torch.Tensor] | str, /) -> None:
    """
    .. warning::

        This function is deprecated as of PyTorch 2.1, please use :func:`torch.set_default_dtype()` and
        :func:`torch.set_default_device()` as alternatives.

    Sets the default ``torch.Tensor`` type to floating point tensor type
    ``t``. This type will also be used as default floating point type for
    type inference in :func:`torch.tensor`.

    The default floating point tensor type is initially ``torch.FloatTensor``.

    Args:
        t (type or string): the floating point tensor type or its name

    Example::

        >>> # xdoctest: +SKIP("Other tests may have changed the default type. Can we reset it?")
        >>> torch.tensor([1.2, 3]).dtype    # initial default for floating point is torch.float32
        torch.float32
        >>> torch.set_default_tensor_type(torch.DoubleTensor)
        >>> torch.tensor([1.2, 3]).dtype    # a new floating point tensor
        torch.float64
    """

def set_default_dtype(d: torch.dtype, /) -> None:
    """
    Sets the default floating point dtype to :attr:`d`. Supports floating point dtype
    as inputs. Other dtypes will cause torch to raise an exception.

    When PyTorch is initialized its default floating point dtype is torch.float32,
    and the intent of set_default_dtype(torch.float64) is to facilitate NumPy-like
    type inference. The default floating point dtype is used to:

    1. Implicitly determine the default complex dtype. When the default floating type is float16,
       the default complex dtype is complex32. For float32, the default complex dtype is complex64.
       For float64, it is complex128. For bfloat16, an exception will be raised because
       there is no corresponding complex type for bfloat16.
    2. Infer the dtype for tensors constructed using Python floats or complex Python
       numbers. See examples below.
    3. Determine the result of type promotion between bool and integer tensors and
       Python floats and complex Python numbers.

    Args:
        d (:class:`torch.dtype`): the floating point dtype to make the default.

    Example:
        >>> # xdoctest: +SKIP("Other tests may have changed the default type. Can we reset it?")
        >>> # initial default for floating point is torch.float32
        >>> # Python floats are interpreted as float32
        >>> torch.tensor([1.2, 3]).dtype
        torch.float32
        >>> # initial default for floating point is torch.complex64
        >>> # Complex Python numbers are interpreted as complex64
        >>> torch.tensor([1.2, 3j]).dtype
        torch.complex64

        >>> torch.set_default_dtype(torch.float64)
        >>> # Python floats are now interpreted as float64
        >>> torch.tensor([1.2, 3]).dtype  # a new floating point tensor
        torch.float64
        >>> # Complex Python numbers are now interpreted as complex128
        >>> torch.tensor([1.2, 3j]).dtype  # a new complex tensor
        torch.complex128

        >>> torch.set_default_dtype(torch.float16)
        >>> # Python floats are now interpreted as float16
        >>> torch.tensor([1.2, 3]).dtype  # a new floating point tensor
        torch.float16
        >>> # Complex Python numbers are now interpreted as complex128
        >>> torch.tensor([1.2, 3j]).dtype  # a new complex tensor
        torch.complex32
    """

def use_deterministic_algorithms(mode: builtins.bool, *, warn_only: builtins.bool = ...) -> None:
    """
    Sets whether PyTorch operations must use "deterministic"
    algorithms. That is, algorithms which, given the same input, and when
    run on the same software and hardware, always produce the same output.
    When enabled, operations will use deterministic algorithms when available,
    and if only nondeterministic algorithms are available they will throw a
    :class:`RuntimeError` when called.

    .. note:: This setting alone is not always enough to make an application
        reproducible. Refer to :ref:`reproducibility` for more information.

    .. note:: :func:`torch.set_deterministic_debug_mode` offers an alternative
        interface for this feature.

    The following normally-nondeterministic operations will act
    deterministically when ``mode=True``:

        * :class:`torch.nn.Conv1d` when called on CUDA tensor
        * :class:`torch.nn.Conv2d` when called on CUDA tensor
        * :class:`torch.nn.Conv3d` when called on CUDA tensor
        * :class:`torch.nn.ConvTranspose1d` when called on CUDA tensor
        * :class:`torch.nn.ConvTranspose2d` when called on CUDA tensor
        * :class:`torch.nn.ConvTranspose3d` when called on CUDA tensor
        * :class:`torch.nn.ReplicationPad1d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReplicationPad2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReplicationPad3d` when attempting to differentiate a CUDA tensor
        * :func:`torch.bmm` when called on sparse-dense CUDA tensors
        * :func:`torch.Tensor.__getitem__` when attempting to differentiate a CPU tensor
          and the index is a list of tensors
        * :func:`torch.Tensor.index_put` with ``accumulate=False``
        * :func:`torch.Tensor.index_put` with ``accumulate=True`` when called on a CPU
          tensor
        * :func:`torch.Tensor.put_` with ``accumulate=True`` when called on a CPU
          tensor
        * :func:`torch.Tensor.scatter_add_` when called on a CUDA tensor
        * :func:`torch.gather` when called on a CUDA tensor that requires grad
        * :func:`torch.index_add` when called on CUDA tensor
        * :func:`torch.index_select` when attempting to differentiate a CUDA tensor
        * :func:`torch.repeat_interleave` when attempting to differentiate a CUDA tensor
        * :func:`torch.Tensor.index_copy` when called on a CPU or CUDA tensor
        * :func:`torch.Tensor.scatter` when `src` type is Tensor and called on CUDA tensor
        * :func:`torch.Tensor.scatter_reduce` when ``reduce='sum'`` or ``reduce='mean'`` and called on CUDA tensor

    The following normally-nondeterministic operations will throw a
    :class:`RuntimeError` when ``mode=True``:

        * :class:`torch.nn.AvgPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.AdaptiveAvgPool2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.AdaptiveAvgPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.MaxPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.AdaptiveMaxPool2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.FractionalMaxPool2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.FractionalMaxPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.MaxUnpool1d`
        * :class:`torch.nn.MaxUnpool2d`
        * :class:`torch.nn.MaxUnpool3d`
        * :func:`torch.nn.functional.interpolate` when attempting to differentiate a CUDA tensor
          and one of the following modes is used:

          - ``linear``
          - ``bilinear``
          - ``bicubic``
          - ``trilinear``

        * :class:`torch.nn.ReflectionPad1d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReflectionPad2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReflectionPad3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.NLLLoss` when called on a CUDA tensor
        * :class:`torch.nn.CTCLoss` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.EmbeddingBag` when attempting to differentiate a CUDA tensor when
          ``mode='max'``
        * :func:`torch.Tensor.put_` when ``accumulate=False``
        * :func:`torch.Tensor.put_` when ``accumulate=True`` and called on a CUDA tensor
        * :func:`torch.histc` when called on a CUDA tensor
        * :func:`torch.bincount` when called on a CUDA tensor and ``weights``
          tensor is given
        * :func:`torch.kthvalue` with called on a CUDA tensor
        * :func:`torch.median` with indices output when called on a CUDA tensor
        * :func:`torch.nn.functional.grid_sample` when attempting to differentiate a CUDA tensor
        * :func:`torch.cumsum` when called on a CUDA tensor when dtype is floating point or complex
        * :func:`torch.Tensor.scatter_reduce` when ``reduce='prod'`` and called on CUDA tensor
        * :func:`torch.Tensor.resize_` when called with a quantized tensor

    In addition, several operations fill uninitialized memory when this setting
    is turned on and when
    :attr:`torch.utils.deterministic.fill_uninitialized_memory` is turned on.
    See the documentation for that attribute for more information.

    A handful of CUDA operations are nondeterministic if the CUDA version is
    10.2 or greater, unless the environment variable ``CUBLAS_WORKSPACE_CONFIG=:4096:8``
    or ``CUBLAS_WORKSPACE_CONFIG=:16:8`` is set. See the CUDA documentation for more
    details: `<https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility>`_
    If one of these environment variable configurations is not set, a :class:`RuntimeError`
    will be raised from these operations when called with CUDA tensors:

        * :func:`torch.mm`
        * :func:`torch.mv`
        * :func:`torch.bmm`

    Note that deterministic operations tend to have worse performance than
    nondeterministic operations.

    .. note::

        This flag does not detect or prevent nondeterministic behavior caused
        by calling an inplace operation on a tensor with an internal memory
        overlap or by giving such a tensor as the :attr:`out` argument for an
        operation. In these cases, multiple writes of different data may target
        a single memory location, and the order of writes is not guaranteed.

    Args:
        mode (:class:`bool`): If True, makes potentially nondeterministic
            operations switch to a deterministic algorithm or throw a runtime
            error. If False, allows nondeterministic operations.

    Keyword args:
        warn_only (:class:`bool`, optional): If True, operations that do not
            have a deterministic implementation will throw a warning instead of
            an error. Default: ``False``

    Example::

        >>> # xdoctest: +SKIP
        >>> torch.use_deterministic_algorithms(True)

        # Forward mode nondeterministic error
        >>> torch.randn(10, device='cuda').kthvalue(1)
        ...
        RuntimeError: kthvalue CUDA does not have a deterministic implementation...

        # Backward mode nondeterministic error
        >>> torch.nn.AvgPool3d(1)(torch.randn(3, 4, 5, 6, requires_grad=True).cuda()).sum().backward()
        ...
        RuntimeError: avg_pool3d_backward_cuda does not have a deterministic implementation...
    """

def are_deterministic_algorithms_enabled() -> builtins.bool:
    """
    Returns True if the global deterministic flag is turned on. Refer to
    :func:`torch.use_deterministic_algorithms` documentation for more details.
    """

def is_deterministic_algorithms_warn_only_enabled() -> builtins.bool:
    """
    Returns True if the global deterministic flag is set to warn only.
    Refer to :func:`torch.use_deterministic_algorithms` documentation for more
    details.
    """

def set_deterministic_debug_mode(debug_mode: builtins.int | str) -> None:
    """
    Sets the debug mode for deterministic operations.

    .. note:: This is an alternative interface for
        :func:`torch.use_deterministic_algorithms`. Refer to that function's
        documentation for details about affected operations.

    Args:
        debug_mode(str or int): If "default" or 0, don't error or warn on
            nondeterministic operations. If "warn" or 1, warn on
            nondeterministic operations. If "error" or 2, error on
            nondeterministic operations.
    """

def get_deterministic_debug_mode() -> builtins.int:
    """
    Returns the current value of the debug mode for deterministic
    operations. Refer to :func:`torch.set_deterministic_debug_mode`
    documentation for more details.
    """

def get_float32_matmul_precision() -> str:
    """
    Returns the current value of float32 matrix multiplication precision. Refer to
    :func:`torch.set_float32_matmul_precision` documentation for more details.
    """

def set_float32_matmul_precision(precision: str) -> None:
    """
    Sets the internal precision of float32 matrix multiplications.

    Running float32 matrix multiplications in lower precision may significantly increase
    performance, and in some programs the loss of precision has a negligible impact.

    Supports three settings:

        * "highest", float32 matrix multiplications use the float32 datatype (24 mantissa
          bits with 23 bits explicitly stored) for internal computations.
        * "high", float32 matrix multiplications either use the TensorFloat32 datatype (10
          mantissa bits explicitly stored) or treat each float32 number as the sum of two bfloat16 numbers
          (approximately 16 mantissa bits with 14 bits explicitly stored), if the appropriate fast matrix multiplication
          algorithms are available.  Otherwise float32 matrix multiplications are computed
          as if the precision is "highest".  See below for more information on the bfloat16
          approach.
        * "medium", float32 matrix multiplications use the bfloat16 datatype (8 mantissa
          bits with 7 bits explicitly stored) for internal computations, if a fast matrix multiplication algorithm
          using that datatype internally is available. Otherwise float32
          matrix multiplications are computed as if the precision is "high".

    When using "high" precision, float32 multiplications may use a bfloat16-based algorithm
    that is more complicated than simply truncating to some smaller number mantissa bits
    (e.g. 10 for TensorFloat32, 7 for bfloat16 explicitly stored).  Refer to [Henry2019]_ for a complete
    description of this algorithm.  To briefly explain here, the first step is to realize
    that we can perfectly encode a single float32 number as the sum of three bfloat16
    numbers (because float32 has 23 mantissa bits while bfloat16 has 7 explicitly stored, and both have the
    same number of exponent bits).  This means that the product of two float32 numbers can
    be exactly given by the sum of nine products of bfloat16 numbers.  We can then trade
    accuracy for speed by dropping some of these products.  The "high" precision algorithm
    specifically keeps only the three most significant products, which conveniently excludes
    all of the products involving the last 8 mantissa bits of either input.  This means that
    we can represent our inputs as the sum of two bfloat16 numbers rather than three.
    Because bfloat16 fused-multiply-add (FMA) instructions are typically >10x faster than
    float32 ones, it's faster to do three multiplications and 2 additions with bfloat16
    precision than it is to do a single multiplication with float32 precision.

    .. [Henry2019] http://arxiv.org/abs/1904.06376

    .. note::

        This does not change the output dtype of float32 matrix multiplications,
        it controls how the internal computation of the matrix multiplication is performed.

    .. note::

        This does not change the precision of convolution operations. Other flags,
        like `torch.backends.cudnn.allow_tf32`, may control the precision of convolution
        operations.

    .. note::

        This flag currently only affects one native device type: CUDA.
        If "high" or "medium" are set then the TensorFloat32 datatype will be used
        when computing float32 matrix multiplications, equivalent to setting
        `torch.backends.cuda.matmul.allow_tf32 = True`. When "highest" (the default)
        is set then the float32 datatype is used for internal computations, equivalent
        to setting `torch.backends.cuda.matmul.allow_tf32 = False`.

    Args:
        precision(str): can be set to "highest" (default), "high", or "medium" (see above).
    """

def set_warn_always(b: builtins.bool, /) -> None:
    """
    When this flag is False (default) then some PyTorch warnings may only
    appear once per process. This helps avoid excessive warning information.
    Setting it to True causes these warnings to always appear, which may be
    helpful when debugging.

    Args:
        b (:class:`bool`): If True, force warnings to always be emitted
                           If False, set to the default behaviour
    """

def is_warn_always_enabled() -> builtins.bool:
    """
    Returns True if the global warn_always flag is turned on. Refer to
    :func:`torch.set_warn_always` documentation for more details.
    """

newaxis: None = ...

class ByteStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class DoubleStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class FloatStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class HalfStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class LongStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class IntStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class ShortStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class CharStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class BoolStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class BFloat16Storage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class ComplexDoubleStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class ComplexFloatStorage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class QUInt8Storage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class QInt8Storage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class QInt32Storage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class QUInt4x2Storage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

class QUInt2x4Storage(_LegacyStorage):
    @classproperty
    def dtype(self): ...

_storage_classes: set[type[TypedStorage | UntypedStorage]] = ...
_tensor_classes: set[type[torch.Tensor]] = ...
_segment_reduce = ...
PRIVATE_OPS = ...

def compiled_with_cxx11_abi() -> builtins.bool:
    """Returns whether PyTorch was built with _GLIBCXX_USE_CXX11_ABI=1"""

legacy_contiguous_format = ...
quantized_lstm = ...
quantized_gru = ...

class _TorchCompileInductorWrapper:
    compiler_name = ...
    def __init__(self, mode, options, dynamic) -> None: ...
    def __eq__(self, other) -> bool: ...
    def apply_mode(self, mode: str | None) -> None: ...
    def apply_options(self, options: dict[str, Any] | None) -> None: ...
    def __call__(self, model_, inputs_) -> Callable[[list[object]], Sequence[Tensor]] | str | list[str] | Weights: ...
    def get_compiler_config(self) -> dict[str, Any]: ...
    def reset(self) -> None: ...

class _TorchCompileWrapper:
    def __init__(self, backend, mode, options, dynamic) -> None: ...
    def __eq__(self, other) -> bool: ...
    def __call__(self, model_, inputs_) -> CompiledFn: ...
    def reset(self) -> None: ...

@overload
def compile[T: Callable](
    model: T,
    *,
    fullgraph: builtins.bool = ...,
    dynamic: builtins.bool | None = ...,
    backend: str | Callable[..., Any] = ...,
    mode: str | None = ...,
    options: dict[str, str | builtins.int | builtins.bool | Callable[..., Any]] | None = ...,
    disable: builtins.bool = ...,
) -> T:
    """
    Optimizes given model/function using TorchDynamo and specified backend.
    If you are compiling an :class:`torch.nn.Module`, you can also use :meth:`torch.nn.Module.compile`
    to compile the module inplace without changing its structure.

    Concretely, for every frame executed within the compiled region, we will attempt
    to compile it and cache the compiled result on the code object for future
    use.  A single frame may be compiled multiple times if previous compiled
    results are not applicable for subsequent calls (this is called a "guard
    failure), you can use TORCH_LOGS=guards to debug these situations.
    Multiple compiled results can be associated with a frame up to
    ``torch._dynamo.config.recompile_limit``, which defaults to 8; at which
    point we will fall back to eager.  Note that compile caches are per
    *code object*, not frame; if you dynamically create multiple copies of a
    function, they will all share the same code cache.

    Args:
       model (Callable or None): Module/function to optimize
       fullgraph (bool): If False (default), torch.compile attempts to discover compilable regions
        in the function that it will optimize. If True, then we require that the entire function be
        capturable into a single graph. If this is not possible (that is, if there are graph breaks),
        then this will raise an error.
       dynamic (bool or None): Use dynamic shape tracing.  When this is True, we will up-front attempt
        to generate a kernel that is as dynamic as possible to avoid recompilations when
        sizes change.  This may not always work as some operations/optimizations will
        force specialization; use TORCH_LOGS=dynamic to debug overspecialization.
        When this is False, we will NEVER generate dynamic kernels, we will always specialize.
        By default (None), we automatically detect if dynamism has occurred and compile a more
        dynamic kernel upon recompile.
       backend (str or Callable): backend to be used

        - "inductor" is the default backend, which is a good balance between performance and overhead

        - Non experimental in-tree backends can be seen with `torch._dynamo.list_backends()`

        - Experimental or debug in-tree backends can be seen with `torch._dynamo.list_backends(None)`

        - To register an out-of-tree custom backend:
          https://pytorch.org/docs/main/torch.compiler_custom_backends.html#registering-custom-backends
       mode (str): Can be either "default", "reduce-overhead", "max-autotune" or "max-autotune-no-cudagraphs"

        - "default" is the default mode, which is a good balance between performance and overhead

        - "reduce-overhead" is a mode that reduces the overhead of python with CUDA graphs,
          useful for small batches.  Reduction of overhead can come at the cost of more memory
          usage, as we will cache the workspace memory required for the invocation so that we
          do not have to reallocate it on subsequent runs.  Reduction of overhead is not guaranteed
          to work; today, we only reduce overhead for CUDA only graphs which do not mutate inputs.
          There are other circumstances where CUDA graphs are not applicable; use TORCH_LOG=perf_hints
          to debug.

        - "max-autotune" is a mode that leverages Triton or template based matrix multiplications
          on supported devices and Triton based convolutions on GPU.
          It enables CUDA graphs by default on GPU.

        - "max-autotune-no-cudagraphs" is a mode similar to "max-autotune" but without CUDA graphs

        - To see the exact configs that each mode sets you can call `torch._inductor.list_mode_options()`

       options (dict): A dictionary of options to pass to the backend. Some notable ones to try out are

        - `epilogue_fusion` which fuses pointwise ops into templates. Requires `max_autotune` to also be set

        - `max_autotune` which will profile to pick the best matmul configuration

        - `fallback_random` which is useful when debugging accuracy issues

        - `shape_padding` which pads matrix shapes to better align loads on GPUs especially for tensor cores

        - `triton.cudagraphs` which will reduce the overhead of python with CUDA graphs

        - `trace.enabled` which is the most useful debugging flag to turn on

        - `trace.graph_diagram` which will show you a picture of your graph after fusion

        - `guard_filter_fn` that controls which dynamo guards are saved with compilations.
          This is an unsafe feature and there is no backward compatibility guarantee provided
          for dynamo guards as data types.
          For stable helper functions to use, see the documentations in `torch.compiler`, for example:
          - `torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe`
          - `torch.compiler.skip_guard_on_all_nn_modules_unsafe`
          - `torch.compiler.keep_tensor_guards_unsafe`

        - For inductor you can see the full list of configs that it supports by calling `torch._inductor.list_options()`
       disable (bool): Turn torch.compile() into a no-op for testing

    Example::

        @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
        def foo(x):
            return torch.sin(x) + torch.cos(x)
    """

@overload
def compile[T: Callable](
    model: None = ...,
    *,
    fullgraph: builtins.bool = ...,
    dynamic: builtins.bool | None = ...,
    backend: str | Callable[..., Any] = ...,
    mode: str | None = ...,
    options: dict[str, str | builtins.int | builtins.bool | Callable[..., Any]] | None = ...,
    disable: builtins.bool = ...,
) -> T:
    """
    Optimizes given model/function using TorchDynamo and specified backend.
    If you are compiling an :class:`torch.nn.Module`, you can also use :meth:`torch.nn.Module.compile`
    to compile the module inplace without changing its structure.

    Concretely, for every frame executed within the compiled region, we will attempt
    to compile it and cache the compiled result on the code object for future
    use.  A single frame may be compiled multiple times if previous compiled
    results are not applicable for subsequent calls (this is called a "guard
    failure), you can use TORCH_LOGS=guards to debug these situations.
    Multiple compiled results can be associated with a frame up to
    ``torch._dynamo.config.recompile_limit``, which defaults to 8; at which
    point we will fall back to eager.  Note that compile caches are per
    *code object*, not frame; if you dynamically create multiple copies of a
    function, they will all share the same code cache.

    Args:
       model (Callable or None): Module/function to optimize
       fullgraph (bool): If False (default), torch.compile attempts to discover compilable regions
        in the function that it will optimize. If True, then we require that the entire function be
        capturable into a single graph. If this is not possible (that is, if there are graph breaks),
        then this will raise an error.
       dynamic (bool or None): Use dynamic shape tracing.  When this is True, we will up-front attempt
        to generate a kernel that is as dynamic as possible to avoid recompilations when
        sizes change.  This may not always work as some operations/optimizations will
        force specialization; use TORCH_LOGS=dynamic to debug overspecialization.
        When this is False, we will NEVER generate dynamic kernels, we will always specialize.
        By default (None), we automatically detect if dynamism has occurred and compile a more
        dynamic kernel upon recompile.
       backend (str or Callable): backend to be used

        - "inductor" is the default backend, which is a good balance between performance and overhead

        - Non experimental in-tree backends can be seen with `torch._dynamo.list_backends()`

        - Experimental or debug in-tree backends can be seen with `torch._dynamo.list_backends(None)`

        - To register an out-of-tree custom backend:
          https://pytorch.org/docs/main/torch.compiler_custom_backends.html#registering-custom-backends
       mode (str): Can be either "default", "reduce-overhead", "max-autotune" or "max-autotune-no-cudagraphs"

        - "default" is the default mode, which is a good balance between performance and overhead

        - "reduce-overhead" is a mode that reduces the overhead of python with CUDA graphs,
          useful for small batches.  Reduction of overhead can come at the cost of more memory
          usage, as we will cache the workspace memory required for the invocation so that we
          do not have to reallocate it on subsequent runs.  Reduction of overhead is not guaranteed
          to work; today, we only reduce overhead for CUDA only graphs which do not mutate inputs.
          There are other circumstances where CUDA graphs are not applicable; use TORCH_LOG=perf_hints
          to debug.

        - "max-autotune" is a mode that leverages Triton or template based matrix multiplications
          on supported devices and Triton based convolutions on GPU.
          It enables CUDA graphs by default on GPU.

        - "max-autotune-no-cudagraphs" is a mode similar to "max-autotune" but without CUDA graphs

        - To see the exact configs that each mode sets you can call `torch._inductor.list_mode_options()`

       options (dict): A dictionary of options to pass to the backend. Some notable ones to try out are

        - `epilogue_fusion` which fuses pointwise ops into templates. Requires `max_autotune` to also be set

        - `max_autotune` which will profile to pick the best matmul configuration

        - `fallback_random` which is useful when debugging accuracy issues

        - `shape_padding` which pads matrix shapes to better align loads on GPUs especially for tensor cores

        - `triton.cudagraphs` which will reduce the overhead of python with CUDA graphs

        - `trace.enabled` which is the most useful debugging flag to turn on

        - `trace.graph_diagram` which will show you a picture of your graph after fusion

        - `guard_filter_fn` that controls which dynamo guards are saved with compilations.
          This is an unsafe feature and there is no backward compatibility guarantee provided
          for dynamo guards as data types.
          For stable helper functions to use, see the documentations in `torch.compiler`, for example:
          - `torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe`
          - `torch.compiler.skip_guard_on_all_nn_modules_unsafe`
          - `torch.compiler.keep_tensor_guards_unsafe`

        - For inductor you can see the full list of configs that it supports by calling `torch._inductor.list_options()`
       disable (bool): Turn torch.compile() into a no-op for testing

    Example::

        @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
        def foo(x):
            return torch.sin(x) + torch.cos(x)
    """

def compile[T: Callable](
    model: T | None = ...,
    *,
    fullgraph: builtins.bool = ...,
    dynamic: builtins.bool | None = ...,
    backend: str | Callable[..., Any] = ...,
    mode: str | None = ...,
    options: dict[str, str | builtins.int | builtins.bool | Callable[..., Any]] | None = ...,
    disable: builtins.bool = ...,
) -> T:
    """
    Optimizes given model/function using TorchDynamo and specified backend.
    If you are compiling an :class:`torch.nn.Module`, you can also use :meth:`torch.nn.Module.compile`
    to compile the module inplace without changing its structure.

    Concretely, for every frame executed within the compiled region, we will attempt
    to compile it and cache the compiled result on the code object for future
    use.  A single frame may be compiled multiple times if previous compiled
    results are not applicable for subsequent calls (this is called a "guard
    failure), you can use TORCH_LOGS=guards to debug these situations.
    Multiple compiled results can be associated with a frame up to
    ``torch._dynamo.config.recompile_limit``, which defaults to 8; at which
    point we will fall back to eager.  Note that compile caches are per
    *code object*, not frame; if you dynamically create multiple copies of a
    function, they will all share the same code cache.

    Args:
       model (Callable or None): Module/function to optimize
       fullgraph (bool): If False (default), torch.compile attempts to discover compilable regions
        in the function that it will optimize. If True, then we require that the entire function be
        capturable into a single graph. If this is not possible (that is, if there are graph breaks),
        then this will raise an error.
       dynamic (bool or None): Use dynamic shape tracing.  When this is True, we will up-front attempt
        to generate a kernel that is as dynamic as possible to avoid recompilations when
        sizes change.  This may not always work as some operations/optimizations will
        force specialization; use TORCH_LOGS=dynamic to debug overspecialization.
        When this is False, we will NEVER generate dynamic kernels, we will always specialize.
        By default (None), we automatically detect if dynamism has occurred and compile a more
        dynamic kernel upon recompile.
       backend (str or Callable): backend to be used

        - "inductor" is the default backend, which is a good balance between performance and overhead

        - Non experimental in-tree backends can be seen with `torch._dynamo.list_backends()`

        - Experimental or debug in-tree backends can be seen with `torch._dynamo.list_backends(None)`

        - To register an out-of-tree custom backend:
          https://pytorch.org/docs/main/torch.compiler_custom_backends.html#registering-custom-backends
       mode (str): Can be either "default", "reduce-overhead", "max-autotune" or "max-autotune-no-cudagraphs"

        - "default" is the default mode, which is a good balance between performance and overhead

        - "reduce-overhead" is a mode that reduces the overhead of python with CUDA graphs,
          useful for small batches.  Reduction of overhead can come at the cost of more memory
          usage, as we will cache the workspace memory required for the invocation so that we
          do not have to reallocate it on subsequent runs.  Reduction of overhead is not guaranteed
          to work; today, we only reduce overhead for CUDA only graphs which do not mutate inputs.
          There are other circumstances where CUDA graphs are not applicable; use TORCH_LOG=perf_hints
          to debug.

        - "max-autotune" is a mode that leverages Triton or template based matrix multiplications
          on supported devices and Triton based convolutions on GPU.
          It enables CUDA graphs by default on GPU.

        - "max-autotune-no-cudagraphs" is a mode similar to "max-autotune" but without CUDA graphs

        - To see the exact configs that each mode sets you can call `torch._inductor.list_mode_options()`

       options (dict): A dictionary of options to pass to the backend. Some notable ones to try out are

        - `epilogue_fusion` which fuses pointwise ops into templates. Requires `max_autotune` to also be set

        - `max_autotune` which will profile to pick the best matmul configuration

        - `fallback_random` which is useful when debugging accuracy issues

        - `shape_padding` which pads matrix shapes to better align loads on GPUs especially for tensor cores

        - `triton.cudagraphs` which will reduce the overhead of python with CUDA graphs

        - `trace.enabled` which is the most useful debugging flag to turn on

        - `trace.graph_diagram` which will show you a picture of your graph after fusion

        - `guard_filter_fn` that controls which dynamo guards are saved with compilations.
          This is an unsafe feature and there is no backward compatibility guarantee provided
          for dynamo guards as data types.
          For stable helper functions to use, see the documentations in `torch.compiler`, for example:
          - `torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe`
          - `torch.compiler.skip_guard_on_all_nn_modules_unsafe`
          - `torch.compiler.keep_tensor_guards_unsafe`

        - For inductor you can see the full list of configs that it supports by calling `torch._inductor.list_options()`
       disable (bool): Turn torch.compile() into a no-op for testing

    Example::

        @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
        def foo(x):
            return torch.sin(x) + torch.cos(x)
    """

if not TYPE_CHECKING: ...
if "TORCH_CUDA_SANITIZER" in os.environ: ...

class _TritonLibrary:
    lib = ...
    ops_table: dict[tuple[str, str], Callable] = ...
    @classmethod
    def registerOp(cls, op_key, full_schema, op_impl, dispatch_key) -> Callable[..., Any]: ...

_deprecated_attrs = ...

@functools.cache
def get_device_module(device: torch.device | str | None = ...) -> Any:
    """
    Returns the module associated with a given device(e.g., torch.device('cuda'), "mtia:0", "xpu", ...).
    If no device is given, return the module for the current accelerator or CPU if none is present.
    """

if _is_device_backend_autoload_enabled(): ...
