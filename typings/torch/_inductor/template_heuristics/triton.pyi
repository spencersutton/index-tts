import dataclasses
from collections.abc import Callable, Generator
from functools import partial
from threading import Lock
from typing import Any

import sympy
import torch
from torch._inductor.template_heuristics.triton_addmm import AddMMConfigMixin
from triton import Config as TritonConfig

from ..ir import Layout
from ..kernel.bmm import bmm_template
from ..kernel.mm import mm_template, persistent_tma_mm_template, scaled_mm_device_tma_template
from ..kernel.mm_plus_mm import mm_plus_mm_template
from ..kernel_inputs import KernelInputs
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics
from .registry import register_template_heuristic

@dataclasses.dataclass
class BaseConfig:
    """Base Gemm configuration used for most backends (CPU, CUDA)"""

    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_warps: int
    hint_override: int | None = ...

@dataclasses.dataclass
class GemmConfig(BaseConfig):
    """Gemm configuration used for most backends (CPU, CUDA)"""

    group_m: int = ...

ConvConfig = BaseConfig

@dataclasses.dataclass
class FlexConfig:
    """
    Base Config class for flex attention
    - FlexAttn forward, backward and flex decode will use this

    NOTE:
    For flex_attn bwd block_m and block_n are reused for block_m1, block_m2, block_n1, block_n2
    """

    block_m: int
    block_n: int
    num_stages: int
    num_warps: int

@dataclasses.dataclass
class FlexDecodeConfig:
    """Config class for flex decoding"""

    block_n: int
    num_stages: int
    num_warps: int

@dataclasses.dataclass
class ROCmGemmConfig(GemmConfig):
    """ROCm subclass for GEMMs, with AMD backend specific tuneable kernargs"""

    matrix_instr_nonkdim: int = ...
    waves_per_eu: int = ...
    kpack: int = ...

@dataclasses.dataclass
class ROCmConvConfig(ConvConfig):
    """ROCm subclass for Conv, with AMD backend specific tuneable kernargs"""

    matrix_instr_nonkdim: int = ...
    waves_per_eu: int = ...
    kpack: int = ...

@dataclasses.dataclass
class ROCmFlexConfig(FlexConfig):
    """ROCm subclass for FlexAttn, with AMD backend specific tuneable kernargs"""

    matrix_instr_nonkdim: int = ...
    waves_per_eu: int = ...
    kpack: int = ...

@dataclasses.dataclass
class ROCmFlexDecodeConfig(FlexDecodeConfig):
    """ROCm subclass for FlexDecode, with AMD backend specific tuneable kernargs"""

    matrix_instr_nonkdim: int = ...
    waves_per_eu: int = ...
    kpack: int = ...

class BaseHeuristicSingleton(type):
    """
    Thread-safe implementation of single to be used in the config heuristic subclasses
    to ensure heavy __init__ calls are not repeatedly run
    """

    _instances: dict[type[Any], Any] = ...
    _lock: Lock = ...
    def __call__(cls: BaseHeuristicSingleton, *args: Any, **kwargs: Any) -> BaseConfigHeuristic: ...

class BaseConfigHeuristic(metaclass=BaseHeuristicSingleton):
    """Base class for mm_configs, device specific triton kernels config inherit from here"""
    def __init__(self) -> None: ...
    def preprocess_mm_configs(
        self,
        m: int,
        n: int,
        k: int,
        configs: list[BaseConfig],
        has_int8_tensor: bool = ...,
        scale: float = ...,
        exclude: Callable[[sympy.Integer, sympy.Integer, sympy.Integer], bool] = ...,
        dtype_size: int = ...,
        op_name: str = ...,
    ) -> Generator[TritonConfig]: ...
    def triton_config(self, num_stages: int, num_warps: int, **kwargs: Any) -> TritonConfig: ...
    def get_mm_configs(self) -> partial[Generator[TritonConfig]]: ...
    def get_exhaustive_mm_configs(self) -> partial[Generator[TritonConfig]]: ...
    def get_conv_configs(self) -> partial[Generator[TritonConfig]]: ...
    def get_flex_attn_fwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]: ...
    def get_flex_attn_bwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]: ...
    def get_flex_decode_configs(self, head_dim: int, dtype: Any) -> list[FlexDecodeConfig]: ...

class CPUConfigHeuristic(BaseConfigHeuristic):
    """CPU-specific config heuristic with CPU-specific optimizations."""
    def preprocess_mm_configs(
        self,
        m: int,
        n: int,
        k: int,
        configs: list[BaseConfig],
        has_int8_tensor: bool = ...,
        scale: float = ...,
        exclude: Callable[[sympy.Integer, sympy.Integer, sympy.Integer], bool] = ...,
        dtype_size: int = ...,
        op_name: str = ...,
    ) -> Generator[TritonConfig]:
        """CPU-specific preprocessing that applies CPU-specific scaling (0.5) and exclusion logic."""

class CUDAConfigHeuristic(BaseConfigHeuristic):
    """Child class for CUDA device specific gemm/flex attention/conv/ configs."""
    def __init__(self) -> None: ...
    def get_flex_attn_fwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]: ...
    def get_flex_attn_bwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]: ...
    def get_flex_decode_configs(self, head_dim: int, dtype: Any) -> list[FlexDecodeConfig]: ...

class ROCmConfigHeuristic(BaseConfigHeuristic):
    """Child class for ROCm specific gemm/flex attention/conv/ configs."""
    def __init__(self) -> None: ...
    def get_flex_attn_fwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]: ...
    def get_flex_attn_bwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]: ...
    def get_flex_decode_configs(self, head_dim: int, dtype: Any) -> list[FlexDecodeConfig]: ...

class XPUConfigHeuristic(BaseConfigHeuristic):
    """Placeholder child class for Intel GPU specific overrides."""
    def __init__(self) -> None: ...
    def get_flex_attn_fwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]: ...
    def get_flex_attn_bwd_configs(self, head_dim: int, dtype: Any) -> list[FlexConfig]: ...
    def get_flex_decode_configs(self, head_dim: int, dtype: Any) -> list[FlexDecodeConfig]: ...

class MTIAConfigHeuristic(BaseConfigHeuristic):
    """Placeholder child class for MTIA specific overrides."""

class MMTemplateConfigMixin(GemmMaxAutotuneTemplateConfigHeuristics):
    """
    Mixin class that converts config lists to template kwargs.
    This handles the logic that was previously in choices.get_mm_configs.

    This mixin expects to be used with BaseConfigHeuristic or its subclasses.
    """

    get_mm_configs: Callable[[], partial[Generator[TritonConfig]]]
    get_exhaustive_mm_configs: Callable[[], partial[Generator[TritonConfig]]]
    _filter_configs: Callable[[list[BaseConfig]], list[BaseConfig]]

class INT8MMTemplateConfigMixin(MMTemplateConfigMixin):
    """Ensure that we feed in has_int8_tensor=True"""
    def __init__(self) -> None: ...

class MMPlusMMTemplateConfigMixin(MMTemplateConfigMixin):
    """Ensure that _should_scale_configs is False"""
    def __init__(self) -> None: ...

class TMAWorkspaceMixin(MMTemplateConfigMixin):
    """
    Small mixin to ensure that the workspace arg is correct for TMA
    and TMA specific filtering can happen.
    """
    def get_extra_kwargs(self, kernel_inputs: KernelInputs, layout: Layout, op_name: str) -> dict[str, Any]: ...

class TMATemplateConfigMixin(TMAWorkspaceMixin, MMTemplateConfigMixin):
    """
    TMA-specific mixin that uses persistent configs and adds TMA options.
    This inherits from MMTemplateConfigMixin and overrides config generation.
    """

class BaseScaledMMConfigMixin(MMTemplateConfigMixin):
    """
    This is a base that handles the common case for ScaledMM

    The TMA and non-TMA should build on top of this
    """
    def adjust_kernel_inputs(self, kernel_inputs: KernelInputs, op_name: str) -> KernelInputs:
        """for scaled_mm, we need to unsqueeze scale tensors, and bias"""

class ScaledMMConfigMixin(BaseScaledMMConfigMixin):
    """Mixing for scaled mm with the regular mm template"""
    def get_extra_kwargs(self, kernel_inputs: KernelInputs, layout: Layout, op_name: str) -> dict[str, Any]: ...

class ScaledTMAConfigMixin(TMAWorkspaceMixin, BaseScaledMMConfigMixin):
    """
    Scaled TMA-specific mixin that extends BaseScaledMMConfigMixin with TMA functionality.
    This is for scaled MM templates that use device TMA.
    This inherits from BaseScaledMMConfigMixin and adds TMA-specific options.
    """

@register_template_heuristic(mm_template.uid, "cuda", register=torch.version.hip is None)
@register_template_heuristic(bmm_template.uid, "cuda", register=torch.version.hip is None)
class CUDAMMTemplateConfigHeuristic(MMTemplateConfigMixin, CUDAConfigHeuristic):
    """Standard MM template heuristic for CUDA"""

@register_template_heuristic(mm_template.uid, "cuda", register=torch.version.hip is None, op_name="addmm")
@register_template_heuristic(bmm_template.uid, "cuda", register=torch.version.hip is None, op_name="baddbmm")
class CUDAAddMMTemplateConfigHeuristic(AddMMConfigMixin, CUDAMMTemplateConfigHeuristic):
    """Addmm specific mixin for CUDA"""

@register_template_heuristic(mm_template.uid, "cuda", register=torch.version.hip is None, op_name="mm-ah")
class CUDAMMAHTemplateConfigHeuristic(MMTemplateConfigMixin, CUDAConfigHeuristic):
    """Standard MM template heuristic for CUDA using the extra mm configs only (for autoheuristic)"""
    def __init__(self) -> None: ...

@register_template_heuristic(persistent_tma_mm_template.uid, "cuda", register=torch.version.hip is None)
class CUDAPersistentTMATemplateConfigHeuristic(TMATemplateConfigMixin, CUDAConfigHeuristic):
    """Persistent TMA template heuristic for CUDA"""
    def __init__(self) -> None: ...

@register_template_heuristic(
    persistent_tma_mm_template.uid, "cuda", register=torch.version.hip is None, op_name="addmm"
)
class CUDAAddmmPersistentTMATemplateConfigHeuristic(AddMMConfigMixin, CUDAPersistentTMATemplateConfigHeuristic):
    """Addmm specific mixin for CUDA"""

@register_template_heuristic(mm_template.uid, "cuda", register=torch.version.hip is None, op_name="scaled_mm")
class CUDAScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, CUDAConfigHeuristic):
    """Scaled MM template heuristic for CUDA"""
    def __init__(self) -> None: ...

@register_template_heuristic(scaled_mm_device_tma_template.uid, "cuda", register=torch.version.hip is None)
class CUDAScaledTMATemplateConfigHeuristic(ScaledTMAConfigMixin, CUDAConfigHeuristic):
    """Scaled TMA template heuristic for CUDA"""
    def __init__(self) -> None: ...

@register_template_heuristic(mm_plus_mm_template.uid, "cuda", register=torch.version.hip is None)
class CUDAMMPlusMMTemplateConfigHeuristic(MMPlusMMTemplateConfigMixin, CUDAConfigHeuristic):
    """MM Plus MM template heuristic for CUDA"""
    def __init__(self) -> None: ...

@register_template_heuristic(mm_template.uid, "cuda", register=torch.version.hip is None, op_name="int_mm")
class CUDAInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, CUDAConfigHeuristic):
    """Int8 MM template heuristic for CUDA"""
    def __init__(self) -> None: ...

@register_template_heuristic(mm_template.uid, "cuda", register=torch.version.hip is not None)
@register_template_heuristic(bmm_template.uid, "cuda", register=torch.version.hip is not None)
class ROCmMMTemplateConfigHeuristic(MMTemplateConfigMixin, ROCmConfigHeuristic):
    """Standard MM template heuristic for ROCm"""

@register_template_heuristic(mm_template.uid, "cuda", register=torch.version.hip is not None, op_name="addmm")
@register_template_heuristic(bmm_template.uid, "cuda", register=torch.version.hip is not None, op_name="baddbmm")
class ROCmAddMMTemplateConfigHeuristic(AddMMConfigMixin, ROCmMMTemplateConfigHeuristic):
    """Addmm specific mixin for ROCm"""

@register_template_heuristic("mm-ah", "cuda", register=torch.version.hip is not None)
class ROCmMMAHTemplateConfigHeuristic(MMTemplateConfigMixin, ROCmConfigHeuristic):
    """Standard MM template heuristic for ROCm using the extra mm configs only (for autoheuristic)"""
    def __init__(self) -> None: ...

@register_template_heuristic(mm_template.uid, "cuda", register=torch.version.hip is not None, op_name="scaled_mm")
class ROCmScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, ROCmConfigHeuristic):
    """Scaled MM template heuristic for ROCm (non-TMA)"""
    def __init__(self) -> None: ...

@register_template_heuristic(mm_template.uid, "cuda", register=torch.version.hip is not None, op_name="int_mm")
class ROCmInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, ROCmConfigHeuristic):
    """Int8 MM template heuristic for ROCm"""
    def __init__(self) -> None: ...

@register_template_heuristic(mm_plus_mm_template.uid, "cuda", register=torch.version.hip is not None)
class ROCmMMPlusMMTemplateConfigHeuristic(MMPlusMMTemplateConfigMixin, ROCmConfigHeuristic):
    """MM Plus MM template heuristic for ROCm"""
    def __init__(self) -> None: ...

@register_template_heuristic(mm_template.uid, "cpu")
@register_template_heuristic(bmm_template.uid, "cpu")
class CPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, CPUConfigHeuristic):
    """Standard MM template heuristic for CPU"""

@register_template_heuristic(mm_template.uid, "cpu", op_name="addmm")
@register_template_heuristic(bmm_template.uid, "cpu", op_name="baddbmm")
class CPUAddmmTemplateConfigHeuristic(AddMMConfigMixin, CPUMMTemplateConfigHeuristic):
    """Addmm specific mixin for CPU"""

@register_template_heuristic(mm_template.uid, "cpu", op_name="scaled_mm")
class CPUScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, CPUConfigHeuristic):
    """Scaled MM template heuristic for CPU (non-TMA)"""
    def __init__(self) -> None: ...

@register_template_heuristic(mm_template.uid, "cpu", op_name="int_mm")
class CPUInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, CPUConfigHeuristic):
    """Int8 MM template heuristic for CPU"""
    def __init__(self) -> None: ...

@register_template_heuristic(mm_plus_mm_template.uid, "cpu")
class CPUMMPlusMMTemplateConfigHeuristic(MMPlusMMTemplateConfigMixin, CPUConfigHeuristic):
    """MM Plus MM template heuristic for CPU"""
    def __init__(self) -> None: ...

@register_template_heuristic(mm_template.uid, "xpu")
@register_template_heuristic(bmm_template.uid, "xpu")
class XPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, XPUConfigHeuristic):
    """Standard MM template heuristic for XPU"""

@register_template_heuristic(mm_template.uid, "xpu", op_name="addmm")
@register_template_heuristic(bmm_template.uid, "xpu", op_name="baddbmm")
class XPUAddmmTemplateConfigHeuristic(AddMMConfigMixin, XPUMMTemplateConfigHeuristic):
    """Addmm specific mixin for XPU"""

@register_template_heuristic(persistent_tma_mm_template.uid, "xpu")
class XPUPersistentTMATemplateConfigHeuristic(TMATemplateConfigMixin, XPUConfigHeuristic):
    """Persistent TMA template heuristic for XPU"""
    def __init__(self) -> None: ...

@register_template_heuristic(persistent_tma_mm_template.uid, "xpu", op_name="addmm")
class XPUAddmmPersistentTMATemplateConfigHeuristic(AddMMConfigMixin, XPUPersistentTMATemplateConfigHeuristic):
    """Addmm specific mixin for XPU"""

@register_template_heuristic(mm_template.uid, "xpu", op_name="scaled_mm")
class XPUScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, XPUConfigHeuristic):
    """Scaled MM template heuristic for XPU (non-TMA)"""
    def __init__(self) -> None: ...

@register_template_heuristic(mm_template.uid, "xpu", op_name="int_mm")
class XPUInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, XPUConfigHeuristic):
    """Int8 MM template heuristic for XPU"""
    def __init__(self) -> None: ...

@register_template_heuristic(mm_plus_mm_template.uid, "xpu")
class XPUMMPlusMMTemplateConfigHeuristic(MMPlusMMTemplateConfigMixin, XPUConfigHeuristic):
    """MM Plus MM template heuristic for XPU"""
    def __init__(self) -> None: ...

@register_template_heuristic(mm_template.uid, "mtia")
@register_template_heuristic(bmm_template.uid, "mtia")
class MTIAMMTemplateConfigHeuristic(MMTemplateConfigMixin, MTIAConfigHeuristic):
    """Standard MM template heuristic for MTIA"""

@register_template_heuristic(mm_template.uid, "mtia", op_name="addmm")
@register_template_heuristic(bmm_template.uid, "mtia", op_name="baddbmm")
class MTIAAddMMTemplateConfigHeuristic(AddMMConfigMixin, MTIAMMTemplateConfigHeuristic):
    """Addmm specific mixin for MTIA"""

@register_template_heuristic(mm_template.uid, "mtia", op_name="scaled_mm")
class MTIAScaledMMTemplateConfigHeuristic(ScaledMMConfigMixin, MTIAConfigHeuristic):
    """Scaled MM template heuristic for MTIA (non-TMA)"""
    def __init__(self) -> None: ...

@register_template_heuristic(mm_template.uid, "mtia", op_name="int_mm")
class MTIAInt8MMTemplateConfigHeuristic(INT8MMTemplateConfigMixin, MTIAConfigHeuristic):
    """Int8 MM template heuristic for MTIA"""
    def __init__(self) -> None: ...

@register_template_heuristic(mm_plus_mm_template.uid, "mtia")
class MTIAMMPlusMMTemplateConfigHeuristic(MMPlusMMTemplateConfigMixin, MTIAConfigHeuristic):
    """MM Plus MM template heuristic for MTIA"""
    def __init__(self) -> None: ...
