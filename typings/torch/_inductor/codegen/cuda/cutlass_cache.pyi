import functools
from typing import Any

from torch._inductor.utils import clear_on_fresh_cache

log = ...
CONFIG_PREFIX: str = ...

def get_config_request_key(arch: str, cuda_version: str, instantiation_level: str) -> str:
    """Return a key for the full ops, based on cutlass key, arch, cuda version, instantiation level, and serialization.py file hash."""

@clear_on_fresh_cache
@functools.cache
def maybe_fetch_ops() -> list[Any] | None:
    """Fetch ops from databases."""
