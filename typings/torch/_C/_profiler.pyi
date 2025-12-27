from enum import Enum
from typing import Literal

from torch._C import device, dtype, layout

class RecordScope(Enum):
    """
    Members:

    FUNCTION

    BACKWARD_FUNCTION

    TORCHSCRIPT_FUNCTION

    KERNEL_FUNCTION_DTYPE

    CUSTOM_CLASS

    BUILD_FEATURE

    LITE_INTERPRETER

    USER_SCOPE

    STATIC_RUNTIME_OP

    STATIC_RUNTIME_MODEL
    """

    FUNCTION = ...
    BACKWARD_FUNCTION = ...
    TORCHSCRIPT_FUNCTION = ...
    KERNEL_FUNCTION_DTYPE = ...
    CUSTOM_CLASS = ...
    BUILD_FEATURE = ...
    LITE_INTERPRETER = ...
    USER_SCOPE = ...
    STATIC_RUNTIME_OP = ...
    STATIC_RUNTIME_MODEL = ...

class ProfilerState(Enum):
    """
    Members:

    Disabled

    CPU

    CUDA

    NVTX

    ITT

    PRIVATEUSE1

    KINETO

    KINETO_GPU_FALLBACK

    KINETO_PRIVATEUSE1_FALLBACK
    """

    Disable = ...
    CPU = ...
    CUDA = ...
    NVTX = ...
    ITT = ...
    KINETO = ...
    KINETO_GPU_FALLBACK = ...
    KINETO_PRIVATEUSE1_FALLBACK = ...
    KINETO_PRIVATEUSE1 = ...

class ActiveProfilerType(Enum):
    """
    Members:

    NONE

    LEGACY

    KINETO

    NVTX

    ITT

    PRIVATEUSE1
    """

    NONE = ...
    LEGACY = ...
    KINETO = ...
    NVTX = ...
    ITT = ...

class ProfilerActivity(Enum):
    """
    Members:

    CPU

    XPU

    MTIA

    CUDA

    HPU

    PrivateUse1
    """

    CPU = ...
    CUDA = ...
    XPU = ...
    MTIA = ...
    HPU = ...
    PrivateUse1 = ...

class _EventType(Enum):
    """
    Members:

    TorchOp

    Backend

    Vulkan

    Allocation

    PyCall

    PyCCall

    Kineto
    """

    TorchOp = ...
    Backend = ...
    Allocation = ...
    OutOfMemory = ...
    PyCall = ...
    PyCCall = ...
    Kineto = ...

class _ExperimentalConfig:
    def __init__(
        self,
        profiler_metrics: list[str] = ...,
        profiler_measure_per_kernel: bool = ...,
        verbose: bool = ...,
        performance_events: list[str] = ...,
        enable_cuda_sync_events: bool = ...,
    ) -> None:
        """
        __init__(self: torch._C._profiler._ExperimentalConfig, profiler_metrics: collections.abc.Sequence[str] = [], profiler_measure_per_kernel: bool = False, verbose: bool = False, performance_events: collections.abc.Sequence[str] = [], enable_cuda_sync_events: bool = False, adjust_profiler_step: bool = False, disable_external_correlation: bool = False, profile_all_threads: bool = False, capture_overload_names: bool = False, record_python_gc_info: bool = False, custom_profiler_config: str = '') -> None

        An experimental config for Kineto features. Please note thatbackward compatibility is not guaranteed.
            profiler_metrics : a list of CUPTI profiler metrics used
               to measure GPU performance events.
               If this list contains values Kineto runs in CUPTI profiler mode
            profiler_measure_per_kernel (bool) : whether to profile metrics per kernel
               or for the entire measurement duration.
            verbose (bool) : whether the trace file has `Call stack` field or not.
            performance_events : a list of profiler events to be used for measurement.
            enable_cuda_sync_events : for CUDA profiling mode, enable adding CUDA synchronization events
               that expose CUDA device, stream and event synchronization activities. This feature is new
               and currently disabled by default.
            adjust_profiler_step (bool) : whether to adjust the profiler step to
               match the parent python event duration. This feature is new and currently disabled by default.
            disable_external_correlation (bool) : whether to disable external correlation
            profile_all_threads (bool) : whether to profile all threads
            capture_overload_names (bool) : whether to include ATen overload names in the profile
            record_python_gc_info (bool) : adds python gc events to profile
            custom_profiler_config (string) : Used to pass some configurations to the custom profiler backend.
        """

class ProfilerConfig:
    def __init__(
        self,
        state: ProfilerState,
        report_input_shapes: bool,
        profile_memory: bool,
        with_stack: bool,
        with_flops: bool,
        with_modules: bool,
        experimental_config: _ExperimentalConfig,
        trace_id: str | None = ...,
    ) -> None:
        """__init__(self: torch._C._profiler.ProfilerConfig, state: torch._C._profiler.ProfilerState, report_input_shapes: bool, profile_memory: bool, with_stack: bool, with_flops: bool, with_modules: bool, experimental_config: torch._C._profiler._ExperimentalConfig, trace_id: str = '') -> None"""

class _ProfilerEvent:
    start_tid: int
    start_time_ns: int
    children: list[_ProfilerEvent]
    extra_fields: (
        _ExtraFields_TorchOp
        | _ExtraFields_Backend
        | _ExtraFields_Allocation
        | _ExtraFields_OutOfMemory
        | _ExtraFields_PyCall
        | _ExtraFields_PyCCall
        | _ExtraFields_Kineto
    )
    @property
    def typed(
        self,
    ) -> (
        tuple[Literal[_EventType.TorchOp], _ExtraFields_TorchOp]
        | tuple[Literal[_EventType.Backend], _ExtraFields_Backend]
        | tuple[Literal[_EventType.Allocation], _ExtraFields_Allocation]
        | tuple[Literal[_EventType.OutOfMemory], _ExtraFields_OutOfMemory]
        | tuple[Literal[_EventType.PyCall], _ExtraFields_PyCall]
        | tuple[Literal[_EventType.PyCCall], _ExtraFields_PyCCall]
        | tuple[Literal[_EventType.Kineto], _ExtraFields_Kineto]
    ): ...
    @property
    def name(self) -> str: ...
    @property
    def tag(self) -> _EventType: ...
    @property
    def id(self) -> int: ...
    @property
    def parent(self) -> _ProfilerEvent | None: ...
    @property
    def correlation_id(self) -> int: ...
    @property
    def end_time_ns(self) -> int: ...
    @property
    def duration_time_ns(self) -> int: ...

class _TensorMetadata:
    impl_ptr: int | None
    storage_data_ptr: int | None
    id: int | None
    @property
    def allocation_id(self) -> int | None: ...
    @property
    def layout(self) -> layout: ...
    @property
    def device(self) -> device: ...
    @property
    def dtype(self) -> dtype: ...
    @property
    def sizes(self) -> list[int]: ...
    @property
    def strides(self) -> list[int]: ...

type Scalar = int | float | bool | complex
type Input = _TensorMetadata | list[_TensorMetadata] | Scalar | None

class _ExtraFields_TorchOp:
    name: str
    sequence_number: int
    allow_tf32_cublas: bool
    @property
    def inputs(self) -> list[Input]: ...
    @property
    def scope(self) -> RecordScope: ...

class _ExtraFields_Backend: ...

class _ExtraFields_Allocation:
    ptr: int
    id: int | None
    alloc_size: int
    total_allocated: int
    total_reserved: int
    @property
    def allocation_id(self) -> int | None: ...
    @property
    def device(self) -> device: ...

class _ExtraFields_OutOfMemory: ...

class _PyFrameState:
    line_number: int
    function_name: str
    @property
    def file_name(self) -> str: ...

class _NNModuleInfo:
    @property
    def self_ptr(self) -> int: ...
    @property
    def cls_ptr(self) -> int: ...
    @property
    def cls_name(self) -> str: ...
    @property
    def parameters(self) -> list[tuple[str, _TensorMetadata, _TensorMetadata | None]]: ...

class _OptimizerInfo:
    @property
    def parameters(self) -> list[tuple[_TensorMetadata, _TensorMetadata | None, list[tuple[str, _TensorMetadata]]]]: ...

class _ExtraFields_PyCCall:
    @property
    def caller(self) -> _PyFrameState: ...

class _ExtraFields_PyCall:
    @property
    def callsite(self) -> _PyFrameState: ...
    @property
    def caller(self) -> _PyFrameState: ...
    @property
    def module(self) -> _NNModuleInfo | None: ...
    @property
    def optimizer(self) -> _OptimizerInfo | None: ...

class _ExtraFields_Kineto: ...
class CapturedTraceback: ...

def gather_traceback(python: bool, script: bool, cpp: bool) -> CapturedTraceback:
    """gather_traceback(python: bool = True, script: bool = True, cpp: bool = True) -> torch._C._profiler.CapturedTraceback"""

def symbolize_tracebacks(to_symbolize: list[CapturedTraceback]) -> list[list[dict[str, str]]]:
    """symbolize_tracebacks(arg0: list) -> list[object]"""

class _RecordFunctionFast:
    def __init__(
        self, name: str, input_values: list | tuple | None = ..., keyword_values: dict | None = ...
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *exc_info: object) -> None: ...
