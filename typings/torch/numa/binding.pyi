from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

__all__ = ["AffinityMode", "NumaOptions", ...]
logger = ...

class AffinityMode(StrEnum):
    """
    See behavior description for each affinity mode
    in torch.distributed.run.
    """

    NODE = ...
    SOCKET = ...
    EXCLUSIVE = ...
    CORE_COMPLEX = ...

@dataclass(frozen=True)
class NumaOptions:
    """NumaOptions(affinity_mode: torch.numa.binding.AffinityMode, should_fall_back_if_binding_fails: bool = False)"""

    affinity_mode: AffinityMode
    should_fall_back_if_binding_fails: bool = ...

@contextmanager
def maybe_temporarily_apply_numa_binding_to_current_thread(
    *, gpu_index: int, numa_options: NumaOptions | None
) -> Iterator[None]:
    """
    1. Applies NUMA binding to the current thread, suitable for the thread
    which will be interacting with GPU gpu_index.
    2. Resets to the original CPU affinity before exiting the context manager.
    """

K = TypeVar("K")
V = TypeVar("V")
