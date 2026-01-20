import builtins
import functools
from collections.abc import Callable, Generator
from math import e, inf, nan, pi
from typing import Any, TypeIs, overload

import torch
from _typeshed import Incomplete
from torch import _inductor as _inductor
from torch import distributed as distributed
from torch import export as export
from torch import nn as nn
from torch._C import *
from torch._C._VariableFunctions import *
from torch._higher_order_ops import cond as cond
from torch._lobpcg import lobpcg as lobpcg
from torch._tensor import Tensor as Tensor
from torch._tensor_str import set_printoptions as set_printoptions
from torch._utils import classproperty
from torch.amp import GradScaler as GradScaler
from torch.amp import autocast as autocast
from torch.autograd import enable_grad as enable_grad
from torch.autograd import inference_mode as inference_mode
from torch.autograd import no_grad as no_grad
from torch.func import vmap as vmap
from torch.functional import *
from torch.random import get_rng_state as get_rng_state
from torch.random import initial_seed as initial_seed
from torch.random import manual_seed as manual_seed
from torch.random import seed as seed
from torch.random import set_rng_state as set_rng_state
from torch.serialization import load as load
from torch.serialization import save as save
from torch.storage import TypedStorage as TypedStorage
from torch.storage import UntypedStorage as UntypedStorage
from torch.storage import _LegacyStorage
from torch.types import Device, IntLikeType

__all__ = [
    "AVG",
    "SUM",
    "AcceleratorError",
    "AggregationType",
    "AliasDb",
    "AnyType",
    "Argument",
    "ArgumentSpec",
    "AwaitType",
    "BenchmarkConfig",
    "BenchmarkExecutionStats",
    "Block",
    "BoolStorage",
    "BoolTensor",
    "BoolType",
    "BufferDict",
    "ByteStorage",
    "ByteTensor",
    "CallStack",
    "Capsule",
    "CharStorage",
    "CharTensor",
    "ClassType",
    "Code",
    "CompilationUnit",
    "CompleteArgumentSpec",
    "ComplexType",
    "ConcreteModuleType",
    "ConcreteModuleTypeBuilder",
    "DeepCopyMemoTable",
    "DeserializationStorageContext",
    "DeviceObjType",
    "DictType",
    "DisableTorchFunction",
    "DisableTorchFunctionSubclass",
    "DispatchKey",
    "DispatchKeySet",
    "DoubleStorage",
    "DoubleTensor",
    "EnumType",
    "ErrorReport",
    "Event",
    "ExcludeDispatchKeyGuard",
    "ExecutionPlan",
    "FatalError",
    "FileCheck",
    "FloatStorage",
    "FloatTensor",
    "FloatType",
    "FunctionSchema",
    "Future",
    "FutureType",
    "Generator",
    "GradScaler",
    "Gradient",
    "Graph",
    "GraphExecutorState",
    "IODescriptor",
    "InferredType",
    "IntStorage",
    "IntTensor",
    "IntType",
    "InterfaceType",
    "JITException",
    "ListType",
    "LiteScriptModule",
    "LockingLogger",
    "LongStorage",
    "LongTensor",
    "ModuleDict",
    "Node",
    "NoneType",
    "NoopLogger",
    "NumberType",
    "OperatorInfo",
    "OptionalType",
    "OutOfMemoryError",
    "ParameterDict",
    "PyObjectType",
    "PyTorchFileReader",
    "PyTorchFileWriter",
    "RRefType",
    "ScriptClass",
    "ScriptClassFunction",
    "ScriptDict",
    "ScriptDictIterator",
    "ScriptDictKeyIterator",
    "ScriptFunction",
    "ScriptList",
    "ScriptListIterator",
    "ScriptMethod",
    "ScriptModule",
    "ScriptModuleSerializer",
    "ScriptObject",
    "ScriptObjectProperty",
    "SerializationStorageContext",
    "ShortStorage",
    "ShortTensor",
    "Size",
    "StaticModule",
    "Stream",
    "StreamObjType",
    "StringType",
    "SymBool",
    "SymBoolType",
    "SymFloat",
    "SymInt",
    "SymIntType",
    "Tag",
    "Tensor",
    "TensorType",
    "ThroughputBenchmark",
    "TracingState",
    "TupleType",
    "Type",
    "TypedStorage",
    "UnionType",
    "UntypedStorage",
    "Use",
    "Value",
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
    "align_tensors",
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
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "autocast",
    "autocast_decrement_nesting",
    "autocast_increment_nesting",
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
    "bit",
    "bits1x8",
    "bits2x4",
    "bits4x2",
    "bits8",
    "bits16",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "blackman_window",
    "block_diag",
    "bmm",
    "bool",
    "broadcast_tensors",
    "broadcast_to",
    "bucketize",
    "can_cast",
    "cartesian_prod",
    "cat",
    "ccol_indices_copy",
    "cdist",
    "cdouble",
    "ceil",
    "ceil_",
    "celu",
    "celu_",
    "cfloat",
    "chain_matmul",
    "chalf",
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
    "clear_autocast_cache",
    "clip",
    "clip_",
    "clone",
    "col_indices_copy",
    "column_stack",
    "combinations",
    "compile",
    "complex",
    "complex32",
    "complex64",
    "complex128",
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
    "cpp",
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
    "default_generator",
    "deg2rad",
    "deg2rad_",
    "dequantize",
    "det",
    "detach",
    "detach_",
    "detach_copy",
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
    "double",
    "dropout",
    "dropout_",
    "dsmm",
    "dsplit",
    "dstack",
    "dtype",
    "e",
    "einsum",
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
    "float4_e2m1fn_x2",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
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
    "fork",
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
    "get_autocast_cpu_dtype",
    "get_autocast_dtype",
    "get_autocast_gpu_dtype",
    "get_autocast_ipu_dtype",
    "get_autocast_xla_dtype",
    "get_default_device",
    "get_default_dtype",
    "get_deterministic_debug_mode",
    "get_device",
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
    "has_lapack",
    "has_mkl",
    "has_openmp",
    "has_spectral",
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
    "import_ir_module",
    "import_ir_module_from_buffer",
    "index_add",
    "index_copy",
    "index_fill",
    "index_put",
    "index_put_",
    "index_reduce",
    "index_select",
    "indices_copy",
    "inf",
    "inference_mode",
    "init_num_threads",
    "initial_seed",
    "inner",
    "instance_norm",
    "int",
    "int1",
    "int2",
    "int3",
    "int4",
    "int5",
    "int6",
    "int7",
    "int8",
    "int16",
    "int32",
    "int64",
    "int_repr",
    "inverse",
    "is_anomaly_check_nan_enabled",
    "is_anomaly_enabled",
    "is_autocast_cache_enabled",
    "is_autocast_cpu_enabled",
    "is_autocast_enabled",
    "is_autocast_ipu_enabled",
    "is_autocast_xla_enabled",
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
    "merge_type_from_type_comment",
    "meshgrid",
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
    "nan",
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
    "newaxis",
    "nextafter",
    "no_grad",
    "nonzero",
    "nonzero_static",
    "norm",
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
    "parse_ir",
    "parse_schema",
    "parse_type_comment",
    "pdist",
    "permute",
    "permute_copy",
    "pi",
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
    "qint8",
    "qint32",
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
    "quint2x4",
    "quint4x2",
    "quint8",
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
    "read_vitals",
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
    "select",
    "select_copy",
    "select_scatter",
    "selu",
    "selu_",
    "set_anomaly_enabled",
    "set_autocast_cache_enabled",
    "set_autocast_cpu_dtype",
    "set_autocast_cpu_enabled",
    "set_autocast_dtype",
    "set_autocast_enabled",
    "set_autocast_gpu_dtype",
    "set_autocast_ipu_dtype",
    "set_autocast_ipu_enabled",
    "set_autocast_xla_dtype",
    "set_autocast_xla_enabled",
    "set_default_device",
    "set_default_tensor_type",
    "set_deterministic_debug_mode",
    "set_float32_matmul_precision",
    "set_flush_denormal",
    "set_num_interop_threads",
    "set_num_threads",
    "set_printoptions",
    "set_rng_state",
    "set_vital",
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
    "stft",
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
    "sym_sqrt",
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
    "tensordot",
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
    "uint1",
    "uint2",
    "uint3",
    "uint4",
    "uint5",
    "uint6",
    "uint7",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "unbind",
    "unbind_copy",
    "unflatten",
    "unfold_copy",
    "unify_type_list",
    "unique_consecutive",
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
    "vitals_enabled",
    "vmap",
    "vsplit",
    "vstack",
    "wait",
    "where",
    "xlogy",
    "xlogy_",
    "zero_",
    "zeros",
    "zeros_like",
]

class SymInt:
    node: Incomplete
    def __init__(self, node) -> None: ...
    def __bool__(self) -> builtins.bool: ...
    def __int__(self) -> builtins.int: ...
    def __index__(self) -> builtins.int: ...
    def __round__(self, ndigits=None): ...
    def __truediv__(self, other): ...
    def __rtruediv__(self, other): ...
    def __floordiv__(self, other): ...
    def __rfloordiv__(self, other): ...
    def __pow__(self, other): ...
    def __rpow__(self, other): ...
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
    def __sym_max__(self, other) -> None: ...
    def __sym_min__(self, other) -> None: ...
    def __sym_float__(self) -> None: ...
    def __neg__(self) -> None: ...
    def __sub__(self, other: IntLikeType) -> SymInt: ...
    def __rsub__(self, other: IntLikeType) -> SymInt: ...
    def __and__(self, other) -> SymInt: ...
    def __or__(self, other) -> SymInt: ...
    def __hash__(self) -> builtins.int: ...
    def as_integer_ratio(self) -> tuple[SymInt, builtins.int]: ...
    def bit_length(self) -> builtins.int: ...
    def conjugate(self) -> SymInt: ...

class SymFloat:
    node: Incomplete
    def __init__(self, node) -> None: ...
    def __truediv__(self, other): ...
    def __rtruediv__(self, other): ...
    def __floordiv__(self, other): ...
    def __rfloordiv__(self, other): ...
    def __bool__(self) -> builtins.bool: ...
    def __float__(self) -> builtins.float: ...
    def __pow__(self, other): ...
    def __rpow__(self, other): ...
    def __eq__(self, other: object) -> builtins.bool: ...
    def __lt__(self, other) -> builtins.bool: ...
    def __gt__(self, other) -> builtins.bool: ...
    def __le__(self, other) -> builtins.bool: ...
    def __ge__(self, other) -> builtins.bool: ...
    def __float_pow__(self, other) -> SymFloat: ...
    def __rfloat_pow__(self, other) -> SymFloat: ...
    def __float_truediv__(self, other) -> SymFloat: ...
    def __rfloat_truediv__(self, other) -> SymFloat: ...
    def __trunc__(self) -> builtins.int: ...
    def __sym_max__(self, other) -> None: ...
    def __sym_min__(self, other) -> None: ...
    def __sym_int__(self) -> None: ...
    def is_integer(self) -> None: ...
    def as_integer_ratio(self) -> tuple[builtins.int, builtins.int]: ...
    def __hash__(self): ...
    def conjugate(self) -> SymFloat: ...
    def hex(self) -> str: ...

class SymBool:
    node: Incomplete
    def __init__(self, node) -> None: ...
    def __bool__(self) -> builtins.bool: ...
    def __int__(self) -> builtins.int: ...
    def __and__(self, other) -> SymBool: ...
    def __or__(self, other) -> SymBool: ...
    def __sym_not__(self) -> SymBool: ...
    def __sym_ite__(self, then_val, else_val) -> None: ...
    def __eq__(self, other) -> builtins.bool: ...
    def __hash__(self): ...

def sym_not(a): ...
def sym_float(a): ...
def sym_int(a): ...
def sym_max(a, b): ...
def sym_min(a, b): ...
def sym_sum(args): ...

sym_sqrt: Incomplete

def sym_ite(b, t, f): ...
def sym_fresh_size(expr): ...
def typename(obj: Any, /) -> str: ...
def is_tensor(obj: Any, /) -> TypeIs[torch.Tensor]: ...
def is_storage(obj: Any, /) -> TypeIs[TypedStorage | UntypedStorage]: ...
def get_default_device() -> torch.device: ...
def set_default_device(device: Device) -> None: ...
def set_default_tensor_type(t: type[torch.Tensor] | str, /) -> None: ...
def use_deterministic_algorithms(mode: builtins.bool, *, warn_only: builtins.bool = False) -> None: ...
def are_deterministic_algorithms_enabled() -> builtins.bool: ...
def is_deterministic_algorithms_warn_only_enabled() -> builtins.bool: ...
def set_deterministic_debug_mode(debug_mode: builtins.int | str) -> None: ...
def get_deterministic_debug_mode() -> builtins.int: ...
def get_float32_matmul_precision() -> str: ...
def set_float32_matmul_precision(precision: str) -> None: ...
def set_warn_always(b: builtins.bool, /) -> None: ...
def is_warn_always_enabled() -> builtins.bool: ...

newaxis: None

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

legacy_contiguous_format = contiguous_format

class _TorchCompileInductorWrapper:
    compiler_name: str
    config: dict[str, Any]
    dynamic: Incomplete
    def __init__(self, mode, options, dynamic) -> None: ...
    def __eq__(self, other): ...
    def apply_mode(self, mode: str | None): ...
    def apply_options(self, options: dict[str, Any] | None): ...
    def __call__(self, model_, inputs_): ...
    def get_compiler_config(self): ...
    def reset(self) -> None: ...

class _TorchCompileWrapper:
    compiler_name: Incomplete
    dynamic: Incomplete
    compiler_fn: Incomplete
    kwargs: Incomplete
    def __init__(self, backend, mode, options, dynamic) -> None: ...
    def __eq__(self, other): ...
    def __call__(self, model_, inputs_): ...
    def reset(self) -> None: ...

@overload
def compile[T: nn.Module](
    model: T,
    *,
    fullgraph: builtins.bool = False,
    dynamic: builtins.bool | None = None,
    backend: str | Callable = "inductor",
    mode: str | None = None,
    options: dict[str, str | builtins.int | builtins.bool | Callable] | None = None,
    disable: builtins.bool = False,
) -> T: ...
@overload
def compile[**InputT, RetT](
    model: Callable[InputT, RetT],
    *,
    fullgraph: builtins.bool = False,
    dynamic: builtins.bool | None = None,
    backend: str | Callable = "inductor",
    mode: str | None = None,
    options: dict[str, str | builtins.int | builtins.bool | Callable] | None = None,
    disable: builtins.bool = False,
) -> Callable[InputT, RetT]: ...
@overload
def compile[**InputT, RetT](
    model: None = None,
    *,
    fullgraph: builtins.bool = False,
    dynamic: builtins.bool | None = None,
    backend: str | Callable = "inductor",
    mode: str | None = None,
    options: dict[str, str | builtins.int | builtins.bool | Callable] | None = None,
    disable: builtins.bool = False,
) -> Callable[[Callable[InputT, RetT]], Callable[InputT, RetT]]: ...

class _TritonLibrary:
    lib: Incomplete
    ops_table: dict[tuple[str, str], Callable]
    @classmethod
    def registerOp(cls, op_key, full_schema, op_impl, dispatch_key): ...

@functools.cache
def get_device_module(device: torch.device | str | None = None): ...
