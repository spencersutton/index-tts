from typing import Any
from warnings import deprecated

import torch

__all__ = ["autocast"]

class autocast(torch.amp.autocast_mode.autocast):
    @deprecated(
        "`torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.",
        category=FutureWarning,
    )
    def __init__(self, enabled: bool = ..., dtype: torch.dtype = ..., cache_enabled: bool = ...) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False] | None: ...
    def __call__(self, func) -> _Wrapped[..., Any, ..., Any]: ...
