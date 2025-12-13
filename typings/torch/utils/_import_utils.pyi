import functools
from types import ModuleType
from typing import Optional

@functools.lru_cache
def dill_available() -> bool: ...
@functools.lru_cache
def import_dill() -> ModuleType | None: ...
