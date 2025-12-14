import contextlib
from collections.abc import Generator
from typing import Any

from torch._C import _SDPBackend as SDPBackend

__all__: list[str] = ["SDPBackend", "sdpa_kernel", "WARN_FOR_UNFUSED_KERNELS"]
WARN_FOR_UNFUSED_KERNELS = ...
_backend_names = ...

@contextlib.contextmanager
def sdpa_kernel(
    backends: list[SDPBackend] | SDPBackend, set_priority: bool = ...
) -> Generator[dict[Any, Any], Any]: ...
