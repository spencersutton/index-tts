import contextlib
from collections.abc import Generator
from typing import Any, Union, TypeAlias

__all__ = ["exposed_in", "argnums_t", "enable_single_level_autograd_function", "unwrap_dead_wrappers"]

@contextlib.contextmanager
def enable_single_level_autograd_function() -> Generator[None, None, None]: ...
def unwrap_dead_wrappers(args: tuple[Any, ...]) -> tuple[Any, ...]: ...

argnums_t: TypeAlias = Union[int, tuple[int, ...]]
