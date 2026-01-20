import functools
import time
from collections.abc import Callable

from torch import nn


def patch_call[**P, R](_src_func: Callable[P, R]) -> Callable[..., Callable[P, R]]:
    @functools.wraps(_src_func)
    def _returns_nn_module_call(*_args: object) -> Callable[P, R]:
        return nn.Module.__call__

    return _returns_nn_module_call


def unwrap[T](value: T | None) -> T:
    if value is None:
        raise ValueError("Expected value to be not None")
    return value


class Timer:
    _start: float = 0
    _end: float = 0
    elapsed: float = 0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> None:
        self._end = time.perf_counter()
        self.elapsed += self._end - self._start

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.stop()

    def __format__(self, __format_spec: str, /) -> str:
        return f"{self.elapsed:{__format_spec}}"
