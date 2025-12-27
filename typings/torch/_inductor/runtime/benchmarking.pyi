from collections.abc import Callable
from functools import cached_property
from typing import Any, Concatenate, ParamSpec, Self, TypeVar

import torch
from torch._inductor.config import use_experimental_benchmarker

logger = ...
use_experimental_benchmarker = ...
MILLISECONDS_PER_SECOND = ...
P = ParamSpec("P")
T = TypeVar("T")

def time_and_count[P, T](fn: Callable[Concatenate[Any, P], T]) -> Callable[Concatenate[Any, P], T]:
    """
    Wraps `fn` with `dynamo_timed` context, and increments the appropriate dynamo
    counters. It is expected that `fn` is a method of `Benchmarker` or one of its
    subclasses; typing limitations prevent us from declaring this directly.
    """

class Benchmarker:
    def __init__(self: Self) -> None: ...
    @time_and_count
    def benchmark(
        self: Self, fn: Callable[..., Any], fn_args: tuple[Any, ...], fn_kwargs: dict[str, Any], **kwargs: Any
    ) -> float:
        """
        Benchmark `fn(*fn_args, *fn_kwargs)` and return the runtime, in milliseconds (the
        actual runtime calculation is dictated by the benchmarking implementation, but may be
        one of [mean, median, minimum, etc.]). Functions as a convenience wrapper around
        device-specific implementations, like `benchmark_cpu` and `benchmark_gpu`. Raises
        `ValueError(...)` if we can't safely infer the device type of `fn`; for example,
        if multiple device types are found in `fn_args` and `fn_kwargs`, or if no device
        types are found.

        Arguments:
        - fn: The function to benchmark.
        - fn_args: The function's arguments.
        - fn_kwargs: The function's kwargs.

        Keyword Arguments:
        - **kwargs: The benchmarking implementation's kwargs.

        Returns:
        - The runtime of `fn(*fn_args, **fn_kwargs)`, in milliseconds.
        """
    @time_and_count
    def benchmark_cpu(self: Self, _callable: Callable[[], Any], warmup: int = ..., rep: int = ...) -> float:
        """
        Benchmark the CPU callable, `_callable`, and return the median runtime,
        in milliseconds.

        Arguments:
        - _callable: The CPU callable to benchmark.

        Keyword Arguments:
        - warmup: Optionally, the duration, in milliseconds, to run `_callable`
        before benchmarking starts.
        - rep: Optionally, the duration, in milliseconds, to run `_callable`
        during benchmarking.

        Returns:
        - The median runtime of `_callable`, in milliseconds.
        """
    @time_and_count
    def benchmark_gpu(self: Self, *args: Any, **kwargs: Any) -> float: ...

class TritonBenchmarker(Benchmarker):
    @cached_property
    def triton_do_bench(self: Self) -> Callable[..., Any]:
        """Lazily import Triton's `do_bench`."""
    @time_and_count
    def benchmark_gpu(self: Self, _callable: Callable[[], Any], **kwargs: Any) -> float:
        """
        Benchmark the GPU callable, `_callable`, and return the runtime, in milliseconds.

        Arguments:
        - _callable: The GPU callable to benchmark.

        Keyword Arguments:
        - quantiles: Optionally, a tuple of floats denoting the requested quantiles.
        - return_mode: Optionally, the requested return mode. Currently, Triton's
        `do_bench` supports min, max, mean, and median return modes.
        - **kwargs: Additional kwargs passed to Triton's `do_bench`.

        Returns:
        - The runtime of `callable`, in milliseconds. If `kwargs["quantiles"]` is specified,
        this is the first requested quantile. Else, if `kwargs["return_mode"]` is specified,
        this is the requested return mode. Otherwise, this is the median.
        """

class InductorBenchmarker(TritonBenchmarker):
    @cached_property
    def L2_cache_size(self: Self) -> int:
        """Get the L2 cache size, in bytes, of the current device."""
    def get_event_pairs(self: Self, iters: int) -> list[tuple[torch.cuda.Event, torch.cuda.Event]]:
        """Get `iters` pairs of CUDA events."""
    def get_event_pairs_min_timing(self: Self, event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]]) -> float:
        """Get the minimum timing, in milliseconds, for a group of CUDA event pairs."""
    @time_and_count
    def benchmark_gpu(
        self: Self,
        _callable: Callable[[], Any],
        estimation_iters: int = ...,
        memory_warmup_iters: int = ...,
        benchmark_iters: int = ...,
        max_benchmark_duration: int = ...,
        return_mode: str = ...,
        grad_to_none: list[torch.Tensor] | None = ...,
        **kwargs: Any,
    ) -> float | list[float]:
        """
        Benchmark a GPU callable using a custom benchmarking implementation.

        Arguments:
        - _callable: The callable to benchmark.

        Keyword Arguments:
        - estimation_iters: Optionally, the number of iterations to run `_callable`
        during runtime estimation.
        - memory_warmup_iters: Optionally, the number of iterations to flush the L2
        cache before starting benchmarking.
        - benchmark_iters: Optionally, the number of iterations to run `_callable`
        during the benchmarking.
        - max_benchmark_duration: Optionally, the maximum duration of the benchmarking,
        in milliseconds. An estimated duration is calculated based on the values
        of `memory_warmup_iters` and `benchmark_iters`, along with the estimated
        runtime of `_callable` and various other factors, and we then shrink
        `benchmark_iters` to fit in the allotted maximum duration.
        - return_mode: Return mode for benchmark results. Options are "min" (default),
        "all" (returns all measurements).
        - grad_to_none: Optionally, a list of tensors whose gradients should be cleared
        before each benchmark iteration.
        - **kwargs: Additional kwargs that may be passed to the fallback.

        Returns:
        - If return_mode="min": The minimum runtime of `_callable`, in milliseconds.
        - If return_mode="all": List of all runtime measurements, in milliseconds.
        """

benchmarker = ...
