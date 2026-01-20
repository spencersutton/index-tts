from typing import Any

from _typeshed import Incomplete
from torch.types import _dtype

__all__ = ["autocast", "autocast_decorator", "custom_bwd", "custom_fwd", "is_autocast_available"]

def is_autocast_available(device_type: str) -> bool: ...
def autocast_decorator(autocast_instance, func): ...

class autocast:
    device: Incomplete
    fast_dtype: Incomplete
    custom_backend_name: Incomplete
    custom_device_mod: Incomplete
    def __init__(
        self, device_type: str, dtype: _dtype | None = None, enabled: bool = True, cache_enabled: bool | None = None
    ) -> None: ...
    prev_cache_enabled: Incomplete
    prev: Incomplete
    prev_fastdtype: Incomplete
    def __enter__(self): ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool | None: ...
    def __call__(self, func): ...

def custom_fwd(fwd=None, *, device_type: str, cast_inputs: _dtype | None = None): ...
def custom_bwd(bwd=None, *, device_type: str): ...
