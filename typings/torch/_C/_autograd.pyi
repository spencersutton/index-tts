"""autograd bindings"""

from enum import Enum
from typing import Any

from torch._C._profiler import _ProfilerEvent

class DeviceType(Enum):
    """
    Members:

    CPU

    CUDA

    MKLDNN

    OPENGL

    OPENCL

    IDEEP

    HIP

    FPGA

    MAIA

    XLA

    Vulkan

    Metal

    XPU

    MPS

    MTIA

    Meta

    HPU

    VE

    Lazy

    IPU

    PrivateUse1
    """

    CPU = ...
    CUDA = ...
    XPU = ...
    MKLDNN = ...
    OPENGL = ...
    OPENCL = ...
    IDEEP = ...
    HIP = ...
    FPGA = ...
    MAIA = ...
    XLA = ...
    MTIA = ...
    MPS = ...
    HPU = ...
    Meta = ...
    Vulkan = ...
    Metal = ...
    PrivateUse1 = ...

class ProfilerEvent:
    def cpu_elapsed_us(self, other: ProfilerEvent) -> float:
        """cpu_elapsed_us(self: torch._C._autograd.ProfilerEvent, arg0: torch._C._autograd.ProfilerEvent) -> float"""
    def cpu_memory_usage(self) -> int:
        """cpu_memory_usage(self: torch._C._autograd.ProfilerEvent) -> int"""
    def cuda_elapsed_us(self, other: ProfilerEvent) -> float:
        """cuda_elapsed_us(self: torch._C._autograd.ProfilerEvent, arg0: torch._C._autograd.ProfilerEvent) -> float"""
    def privateuse1_elapsed_us(self, other: ProfilerEvent) -> float: ...
    def cuda_memory_usage(self) -> int:
        """cuda_memory_usage(self: torch._C._autograd.ProfilerEvent) -> int"""
    def device(self) -> int:
        """device(self: torch._C._autograd.ProfilerEvent) -> int"""
    def handle(self) -> int:
        """handle(self: torch._C._autograd.ProfilerEvent) -> int"""
    def has_cuda(self) -> bool:
        """has_cuda(self: torch._C._autograd.ProfilerEvent) -> bool"""
    def is_remote(self) -> bool:
        """is_remote(self: torch._C._autograd.ProfilerEvent) -> bool"""
    def kind(self) -> int:
        """kind(self: torch._C._autograd.ProfilerEvent) -> str"""
    def name(self) -> str:
        """name(self: torch._C._autograd.ProfilerEvent) -> str"""
    def node_id(self) -> int:
        """node_id(self: torch._C._autograd.ProfilerEvent) -> int"""
    def sequence_nr(self) -> int:
        """sequence_nr(self: torch._C._autograd.ProfilerEvent) -> int"""
    def shapes(self) -> list[list[int]]:
        """shapes(self: torch._C._autograd.ProfilerEvent) -> list[list[int]]"""
    def thread_id(self) -> int:
        """thread_id(self: torch._C._autograd.ProfilerEvent) -> int"""
    def flops(self) -> float:
        """flops(self: torch._C._autograd.ProfilerEvent) -> int"""
    def is_async(self) -> bool:
        """is_async(self: torch._C._autograd.ProfilerEvent) -> bool"""

class _KinetoEvent:
    def name(self) -> str:
        """name(self: torch._C._autograd._KinetoEvent) -> str"""
    def overload_name(self) -> str:
        """overload_name(self: torch._C._autograd._KinetoEvent) -> str"""
    def device_index(self) -> int:
        """device_index(self: torch._C._autograd._KinetoEvent) -> int"""
    def device_resource_id(self) -> int:
        """device_resource_id(self: torch._C._autograd._KinetoEvent) -> int"""
    def start_ns(self) -> int:
        """start_ns(self: torch._C._autograd._KinetoEvent) -> int"""
    def end_ns(self) -> int:
        """end_ns(self: torch._C._autograd._KinetoEvent) -> int"""
    def duration_ns(self) -> int:
        """duration_ns(self: torch._C._autograd._KinetoEvent) -> int"""
    def is_async(self) -> bool:
        """is_async(self: torch._C._autograd._KinetoEvent) -> bool"""
    def linked_correlation_id(self) -> int:
        """linked_correlation_id(self: torch._C._autograd._KinetoEvent) -> int"""
    def shapes(self) -> list[list[int]]:
        """shapes(self: torch._C._autograd._KinetoEvent) -> list[list[int]]"""
    def dtypes(self) -> list[str]:
        """dtypes(self: torch._C._autograd._KinetoEvent) -> list[str]"""
    def concrete_inputs(self) -> list[Any]:
        """concrete_inputs(self: torch._C._autograd._KinetoEvent) -> list[object]"""
    def kwinputs(self) -> dict[str, Any]:
        """kwinputs(self: torch._C._autograd._KinetoEvent) -> dict[str, object]"""
    def device_type(self) -> DeviceType:
        """device_type(self: torch._C._autograd._KinetoEvent) -> torch._C._autograd.DeviceType"""
    def start_thread_id(self) -> int:
        """start_thread_id(self: torch._C._autograd._KinetoEvent) -> int"""
    def end_thread_id(self) -> int:
        """end_thread_id(self: torch._C._autograd._KinetoEvent) -> int"""
    def correlation_id(self) -> int:
        """correlation_id(self: torch._C._autograd._KinetoEvent) -> int"""
    def fwd_thread_id(self) -> int:
        """fwd_thread_id(self: torch._C._autograd._KinetoEvent) -> int"""
    def stack(self) -> list[str]:
        """stack(self: torch._C._autograd._KinetoEvent) -> list[str]"""
    def scope(self) -> int:
        """scope(self: torch._C._autograd._KinetoEvent) -> int"""
    def sequence_nr(self) -> int:
        """sequence_nr(self: torch._C._autograd._KinetoEvent) -> int"""
    def flops(self) -> int:
        """flops(self: torch._C._autograd._KinetoEvent) -> int"""
    def cuda_elapsed_us(self) -> int:
        """cuda_elapsed_us(self: torch._C._autograd._KinetoEvent) -> int"""
    def privateuse1_elapsed_us(self) -> int:
        """privateuse1_elapsed_us(self: torch._C._autograd._KinetoEvent) -> int"""
    def is_user_annotation(self) -> bool:
        """is_user_annotation(self: torch._C._autograd._KinetoEvent) -> bool"""
    def is_hidden_event(self) -> bool:
        """is_hidden_event(self: torch._C._autograd._KinetoEvent) -> bool"""

class _ProfilerResult:
    def events(self) -> list[_KinetoEvent]:
        """events(self: torch._C._autograd._ProfilerResult) -> list[torch._C._autograd._KinetoEvent]"""
    def legacy_events(self) -> list[list[ProfilerEvent]]: ...
    def save(self, path: str) -> None:
        """save(self: torch._C._autograd._ProfilerResult, arg0: str) -> None"""
    def experimental_event_tree(self) -> list[_ProfilerEvent]:
        """experimental_event_tree(self: torch._C._autograd._ProfilerResult) -> list[torch._C._profiler._ProfilerEvent]"""
    def trace_start_ns(self) -> int:
        """trace_start_ns(self: torch._C._autograd._ProfilerResult) -> int"""

class SavedTensor: ...

def kineto_available() -> bool:
    """kineto_available() -> bool"""

class CreationMeta(Enum):
    """
    Members:

    DEFAULT

    IN_CUSTOM_FUNCTION

    MULTI_OUTPUT_NODE

    NO_GRAD_MODE

    INFERENCE_MODE
    """

    DEFAULT = ...
    IN_CUSTOM_FUNCTION = ...
    MULTI_OUTPUT_NODE = ...
    NO_GRAD_MODE = ...
    INFERENCE_MODE = ...
