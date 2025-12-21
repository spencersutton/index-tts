"""Profiling utilities for IndexTTS inference."""

from __future__ import annotations

import contextlib
import functools
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class TimingStats:
    """Accumulated timing statistics for a named operation."""

    name: str
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float("inf")
    max_time: float = 0.0
    samples: list[float] = field(default_factory=list)

    def record(self, duration: float) -> None:
        """Record a timing sample."""
        self.total_time += duration
        self.call_count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.samples.append(duration)

    @property
    def avg_time(self) -> float:
        """Average time per call."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    def __str__(self) -> str:
        if self.call_count == 0:
            return f"{self.name}: no calls"
        if self.call_count == 1:
            return f"{self.name}: {self.total_time * 1000:.2f}ms"
        return (
            f"{self.name}: {self.total_time * 1000:.2f}ms total, "
            f"{self.avg_time * 1000:.2f}ms avg, "
            f"{self.min_time * 1000:.2f}ms min, "
            f"{self.max_time * 1000:.2f}ms max "
            f"({self.call_count} calls)"
        )


class InferenceProfiler:
    """Comprehensive profiler for IndexTTS inference pipeline.

    Usage:
        profiler = InferenceProfiler()
        with profiler.measure("gpt_generation"):
            # ... code to measure ...

        # Or as a decorator:
        @profiler.profile("my_function")
        def my_function():
            ...

        # Print results:
        profiler.report()
    """

    def __init__(self, enabled: bool = True, cuda_sync: bool = True) -> None:
        """Initialize profiler.

        Args:
            enabled: Whether profiling is active
            cuda_sync: Whether to synchronize CUDA before/after measurements
        """
        self.enabled = enabled
        self.cuda_sync = cuda_sync and torch.cuda.is_available()
        self._stats: dict[str, TimingStats] = defaultdict(lambda: TimingStats(name=""))
        self._stack: list[str] = []
        self._start_time: float | None = None
        self._end_time: float | None = None

    def reset(self) -> None:
        """Reset all collected statistics."""
        self._stats.clear()
        self._stack.clear()
        self._start_time = None
        self._end_time = None

    def start_session(self) -> None:
        """Mark the start of an inference session."""
        self._start_time = time.perf_counter()

    def end_session(self) -> None:
        """Mark the end of an inference session."""
        self._end_time = time.perf_counter()

    @property
    def session_time(self) -> float:
        """Total session time in seconds."""
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time is not None else time.perf_counter()
        return end - self._start_time

    @contextlib.contextmanager
    def measure(self, name: str) -> Generator[None]:
        """Context manager to measure execution time of a code block.

        Args:
            name: Name for this timing measurement
        """
        if not self.enabled:
            yield
            return

        # Ensure stats entry exists with correct name
        if name not in self._stats:
            self._stats[name] = TimingStats(name=name)

        # Sync CUDA if needed
        if self.cuda_sync:
            torch.cuda.synchronize()

        self._stack.append(name)
        start = time.perf_counter()

        try:
            yield
        finally:
            if self.cuda_sync:
                torch.cuda.synchronize()

            duration = time.perf_counter() - start
            self._stats[name].record(duration)
            self._stack.pop()

    def profile(self, name: str | None = None) -> Callable[[F], F]:
        """Decorator to profile a function.

        Args:
            name: Optional name override (defaults to function name)
        """

        def decorator(func: F) -> F:
            profile_name = name if name is not None else func.__name__

            @functools.wraps(func)
            def wrapper(*args: object, **kwargs: object) -> object:
                with self.measure(profile_name):
                    return func(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    def get_stats(self, name: str) -> TimingStats | None:
        """Get timing statistics for a named operation."""
        return self._stats.get(name)

    def report(self, min_time_ms: float = 0.1) -> str:
        """Generate a human-readable profiling report.

        Args:
            min_time_ms: Minimum total time in ms to include in report

        Returns:
            Formatted report string
        """
        lines = ["=" * 60, "IndexTTS Inference Profiling Report", "=" * 60]

        if self._start_time is not None:
            lines.append(f"Total session time: {self.session_time * 1000:.2f}ms")
            lines.append("-" * 60)

        # Sort by total time descending
        sorted_stats = sorted(self._stats.values(), key=lambda s: s.total_time, reverse=True)

        accounted_time = 0.0
        for stats in sorted_stats:
            if stats.total_time * 1000 >= min_time_ms:
                lines.append(str(stats))
                accounted_time += stats.total_time

        if self._start_time is not None and self.session_time > 0:
            lines.append("-" * 60)
            pct = (accounted_time / self.session_time) * 100
            lines.append(f"Profiled time: {accounted_time * 1000:.2f}ms ({pct:.1f}% of session)")

        lines.append("=" * 60)
        return "\n".join(lines)

    def report_dict(self) -> Mapping[str, dict[str, float]]:
        """Return profiling data as a dictionary for programmatic access."""
        return {
            name: {
                "total_ms": stats.total_time * 1000,
                "avg_ms": stats.avg_time * 1000,
                "min_ms": stats.min_time * 1000,
                "max_ms": stats.max_time * 1000,
                "calls": stats.call_count,
            }
            for name, stats in self._stats.items()
        }

    def log_report(self, log_level: int = logging.INFO, min_time_ms: float = 0.1) -> None:
        """Log the profiling report."""
        for line in self.report(min_time_ms).split("\n"):
            logger.log(log_level, line)


# Global profiler instance for easy access
_global_profiler: InferenceProfiler | None = None


def get_profiler() -> InferenceProfiler:
    """Get or create the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = InferenceProfiler()
    return _global_profiler


def set_profiler(profiler: InferenceProfiler | None) -> None:
    """Set the global profiler instance."""
    global _global_profiler
    _global_profiler = profiler


@contextlib.contextmanager
def profile_section(name: str) -> Generator[None]:
    """Profile a section using the global profiler.

    Args:
        name: Name for this timing measurement
    """
    profiler = get_profiler()
    with profiler.measure(name):
        yield


def profile_function(name: str | None = None) -> Callable[[F], F]:
    """Decorator to profile a function using the global profiler."""
    profiler = get_profiler()
    return profiler.profile(name)
