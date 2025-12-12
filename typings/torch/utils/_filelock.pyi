from types import TracebackType
from typing import Optional
from typing_extensions import Self
from filelock import FileLock as base_FileLock

class FileLock(base_FileLock):
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None: ...
