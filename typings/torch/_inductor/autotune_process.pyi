import dataclasses
import functools
from collections.abc import Callable, Iterable, Sequence
from typing import IO, Any

import torch
from torch._inductor import ir
from torch._inductor.select_algorithm import PartialRender, TritonTemplateCaller

CUDA_VISIBLE_DEVICES = ...
autotuning_log = ...

class NonzeroWorkspaceNotSupportedError(Exception): ...

class TuningProcess:
    """Class to launch and interact with a benchmarking subprocess."""
    @staticmethod
    def process_main(read_pipe: IO[bytes], write_pipe: IO[bytes]) -> None:
        """Entry point for the child process."""
    @staticmethod
    def send(obj: Any, write_pipe: IO[bytes]) -> None: ...
    @staticmethod
    def recv(read_pipe: IO[bytes]) -> Any: ...
    def __init__(self, device: int | None) -> None: ...
    def start(self):
        """Start the benchmarking subprocess."""
    def alive(self) -> bool:
        """True if the subprocess is still running."""
    def put(self, req: Any) -> None:
        """Push a work item to the child process."""
    def get(self, timeout: float = ...) -> Any:
        """
        Get a response from the child process. Raises TimeoutError on timeout;
        raises EOFError if the subprocess crashes.
        """
    def shutdown(self, wait: bool = ...) -> None:
        """Signal the child process to shut down gracefully."""
    def wait(self) -> None:
        """Wait for the child process to exit."""
    def close(self) -> None:
        """Close resources."""
    def kill(self) -> None:
        """Send a SIGKILL to the child process."""

class TuningProcessPool:
    """
    Maintains a pool of TuningProcesses to benchmark kernels in parallel
    across devices. By default, we create one TuningProcess per device and
    set the sub-process environment to make only that device visible.
    """
    def __init__(self) -> None:
        """Start the child processes."""
    @staticmethod
    def get_device_list() -> Sequence[int | None]:
        """Gather the list of devices to be used in the pool."""
    def shutdown(self) -> None:
        """Signal all child processes to exit."""
    def target(self, choice: TritonTemplateCaller) -> float:
        """
        Entry point for the thread-pool helper threads: Wait for an open TuningProcess,
        remove it from the queue, execute the benchmark in that subprocess, and return
        the TuningProcess to the queue.
        """
    def benchmark(self, choices: list[TritonTemplateCaller]) -> dict[TritonTemplateCaller, float]:
        """Benchmark each choice in a separate process."""

type LayoutOrBuffer = ir.Layout | ir.Buffer

@dataclasses.dataclass
class TensorMeta:
    """TensorMeta(device: 'torch.device', dtype: 'torch.dtype', sizes: 'torch._prims_common.ShapeType', strides: 'torch._prims_common.StrideType', offset: 'int', name: 'Optional[str]' = None)"""

    device: torch.device
    dtype: torch.dtype
    sizes: torch._prims_common.ShapeType
    strides: torch._prims_common.StrideType
    offset: int
    name: str | None = ...
    @classmethod
    def from_irnodes(cls, irnodes: LayoutOrBuffer | Sequence[LayoutOrBuffer]) -> TensorMeta | list[TensorMeta]: ...
    def to_tensor(self) -> torch.Tensor: ...

@dataclasses.dataclass
class BenchmarkRequest:
    """
    Only handle triton template benchmark for now. The extern kernel benchmark
    can be done inside the same process since they usually don't cause crash.

    Important: Instances of this class and subclasses have to be serializable
    across process boundaries. Do not put CUDA Tensors in here!
    """
    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
    ) -> None: ...
    def make_run_fn(self, *input_tensors: torch.Tensor, out: torch.Tensor) -> Callable[[], None]: ...
    def cleanup_run_fn(self) -> None: ...
    def do_bench(self, fn, *input_tensors: torch.Tensor, out: torch.Tensor | None = ...) -> float: ...
    def benchmark(self, *input_tensors: torch.Tensor, out: torch.Tensor | None = ...) -> float: ...

class _TestBenchmarkRequest(BenchmarkRequest):
    """
    Supports unit testing. Defined in this file instead of the test file so the
    TuningProcess sub-process can unpickle these objects.
    """
    def __init__(
        self,
        result: float = ...,
        device: int | None = ...,
        sleep: float | None = ...,
        exc: Exception | None = ...,
        crash: bool = ...,
    ) -> None: ...
    def benchmark(self, *input_tensors: torch.Tensor, out: torch.Tensor | None = ...) -> float: ...

class GPUDeviceBenchmarkMixin:
    def do_bench(self, fn, *input_tensors: torch.Tensor, out: torch.Tensor | None = ...) -> float: ...

class CPUDeviceBenchmarkMixin:
    def do_bench(self, fn, *input_tensors: torch.Tensor, out: torch.Tensor | None = ...) -> float: ...

class TritonBenchmarkRequest(BenchmarkRequest):
    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
        module_path: str,
        module_cache_key: str,
        num_stages: int,
        num_warps: int,
        num_consumer_groups: int = ...,
        num_buffers_warp_spec: int = ...,
        matrix_instr_nonkdim: int = ...,
        waves_per_eu: int = ...,
        kpack: int = ...,
    ) -> None: ...
    def make_run_fn(self, *input_tensors: torch.Tensor, out: torch.Tensor) -> Callable[[], None]: ...
    def precompile(self): ...

class TritonGPUBenchmarkRequest(GPUDeviceBenchmarkMixin, TritonBenchmarkRequest): ...
class TritonCPUBenchmarkRequest(CPUDeviceBenchmarkMixin, TritonBenchmarkRequest): ...

class CUDABenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    """
    A class to handle CUDA (CUTLASS) benchmark requests. This class is for
    managing the lifecycle of a CUDA kernel benchmark, including compiling
    the source code, managing workspace memory, and executing the kernel.

    Important: Instances of this class have to be serializable across
    process boundaries. Do not put CUDA Tensors in here!
    """
    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
        source_code: str,
    ) -> None: ...
    def precompile(self):
        """
        Precompile the CUDA source code to populate the CUDACodeCache.
        This may happen in a separate thread pool.
        """
    def make_run_fn(self, *input_tensors: torch.Tensor, out: torch.Tensor) -> Callable[[], None]:
        """Create a function to run the CUDA kernel with the given input and output tensors."""
    def update_workspace_size(self) -> None: ...
    def ensure_dll_loaded(self): ...
    def cleanup_run_fn(self) -> None: ...

class CppBenchmarkRequest(CPUDeviceBenchmarkMixin, BenchmarkRequest):
    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: Iterable[Any],
        source_code: str,
    ) -> None: ...
    def precompile(self): ...
    def make_run_fn(self, *input_tensors: torch.Tensor, out: torch.Tensor) -> Callable[[], None]: ...
    def cleanup_run_fn(self) -> None: ...

class CuteDSLBenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    """Benchmark request for CuteDSL (CUTLASS Python DSL) kernels."""
    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        extra_args: tuple[Any, ...],
        source_code: PartialRender,
    ) -> None: ...
    def make_run_fn(self, *input_tensors: torch.Tensor, out: torch.Tensor) -> Callable[[], None]:
        """
        Create a function to run the CuteDSL kernel with the given input and output tensors.
        Similar to TritonBenchmarkRequest.make_run_fn but for CuteDSL kernels.
        """
    def cleanup_run_fn(self) -> None:
        """Clean up any resources used by the kernel."""

@functools.cache
def get_tuning_process_pool() -> TuningProcessPool: ...
def benchmark_in_sub_process(choices: list[TritonTemplateCaller]) -> dict[TritonTemplateCaller, float]:
    """Do benchmarking in a subprocess and return the perf number (latency)."""
