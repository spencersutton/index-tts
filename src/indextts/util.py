import time
from collections.abc import Callable
from typing import Self, override

from torch import nn


def patch_call[**P, R](_src_func: Callable[P, R]) -> Callable[..., Callable[P, R]]:
    """Decorator that replaces a module's ``__call__`` stub with the real ``nn.Module.__call__``.

    PyTorch's ``nn.Module.__call__`` is what triggers hooks and calls ``forward``, but type
    checkers cannot infer the correct signature from it.  The pattern used throughout this
    codebase is to declare a typed ``__call__`` stub and annotate it with ``@patch_call(forward)``
    so that Pyright/mypy see the correct signature while the runtime still uses the proper
    ``nn.Module.__call__`` machinery.

    Args:
        _src_func: The ``forward`` method whose signature should be used for type-checking.

    Returns:
        A decorator that, when applied to a ``__call__`` stub, replaces it with
        ``nn.Module.__call__`` at runtime.
    """

    def _returns_nn_module_call(*_args: object) -> Callable[P, R]:
        return nn.Module.__call__

    return _returns_nn_module_call


def unwrap[T](value: T | None) -> T:
    """Assert that *value* is not ``None`` and return it.

    Raises:
        ValueError: If *value* is ``None``.
    """
    if value is None:
        raise ValueError("Expected value to be not None")
    return value


class Timer:
    """A simple cumulative wall-clock timer that supports the context-manager protocol.

    Elapsed time accumulates across multiple ``start``/``stop`` pairs so that the same
    ``Timer`` instance can be reused inside a loop to measure total time spent in a
    section across iterations.

    Example::

        timer = Timer()
        for item in items:
            with timer:
                process(item)
        print(f"total: {timer:.2f}s")
    """

    _start: float = 0
    _end: float = 0
    elapsed: float = 0

    def start(self) -> None:
        """Record the current time as the start of a new interval."""
        self._start = time.perf_counter()

    def stop(self) -> None:
        """Stop the current interval and add its duration to :attr:`elapsed`."""
        self._end = time.perf_counter()
        self.elapsed += self._end - self._start

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.stop()

    @override
    def __repr__(self) -> str:
        return f"Timer(elapsed={self.elapsed:.4f}s)"

    @override
    def __format__(self, __format_spec: str, /) -> str:
        return f"{self.elapsed:{__format_spec}}"
