import contextlib
from collections.abc import Generator
from typing import Any, TypeAlias, Union

__all__ = ["exposed_in", "argnums_t", "enable_single_level_autograd_function", "unwrap_dead_wrappers"]

@contextlib.contextmanager
def enable_single_level_autograd_function() -> Generator[None]: ...
def unwrap_dead_wrappers(args: tuple[Any, ...]) -> tuple[Any, ...]: ...

type argnums_t = int | tuple[int, ...]
