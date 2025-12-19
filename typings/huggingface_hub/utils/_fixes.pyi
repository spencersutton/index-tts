import contextlib
from collections.abc import Callable, Generator
from pathlib import Path

from filelock import BaseFileLock

logger = ...
yaml_dump: Callable[..., str] = ...

@contextlib.contextmanager
def SoftTemporaryDirectory(
    suffix: str | None = ..., prefix: str | None = ..., dir: Path | str | None = ..., **kwargs
) -> Generator[Path]: ...
@contextlib.contextmanager
def WeakFileLock(lock_file: str | Path, *, timeout: float | None = ...) -> Generator[BaseFileLock]: ...
