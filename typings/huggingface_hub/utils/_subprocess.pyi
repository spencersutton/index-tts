import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import IO

"""Contains utilities to easily handle subprocesses in `huggingface_hub`."""
logger = ...

@contextmanager
def capture_output() -> Generator[StringIO]: ...
def run_subprocess(
    command: str | list[str], folder: str | Path | None = ..., check=..., **kwargs
) -> subprocess.CompletedProcess: ...
@contextmanager
def run_interactive_subprocess(
    command: str | list[str], folder: str | Path | None = ..., **kwargs
) -> Generator[tuple[IO[str], IO[str]]]: ...
