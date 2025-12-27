"""This package enables an interface for accessing MTIA backend in python"""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, device as _device

from .memory import *

type _device_t = _device | str | int
Event = torch.Event
Stream = torch.Stream
_initialized = ...
_queued_calls: list[tuple[Callable[[], None], list[str]]] = ...
_tls = ...
_initialization_lock = ...
_lazy_seed_tracker = ...
if hasattr(torch._C, "_mtia_exchangeDevice"):
    _exchange_device = ...
if hasattr(torch._C, "_mtia_maybeExchangeDevice"):
    _maybe_exchange_device = ...

def init() -> None: ...
def is_initialized() -> bool:
    """Return whether PyTorch's MTIA state has been initialized."""

class DeferredMtiaCallError(Exception): ...

def is_available() -> bool:
    """Return true if MTIA device is available"""

def synchronize(device: _device_t | None = ...) -> None:
    """Waits for all jobs in all streams on a MTIA device to complete."""

def device_count() -> int:
    """Return the number of MTIA devices available."""

def current_device() -> int:
    """Return the index of a currently selected device."""

def current_stream(device: _device_t | None = ...) -> Stream:
    """
    Return the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.mtia.current_device`, if :attr:`device` is ``None``
            (default).
    """

def default_stream(device: _device_t | None = ...) -> Stream:
    """
    Return the default :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch.mtia.current_device`, if :attr:`device` is ``None``
            (default).
    """

def record_memory_history(enabled: str | None = ..., stacks: str = ..., max_entries: int = ...) -> None:
    """
    Enable/Disable the memory profiler on MTIA allocator

    Args:
        enabled (all or state, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).

        stacks ("python" or "cpp", optional). Select the stack trace to record.

        max_entries (int, optional). Maximum number of entries to record.
    """

def snapshot() -> dict[str, Any]:
    """Return a dictionary of MTIA memory allocator history"""

def attach_out_of_memory_observer(observer: Callable[[int, int, int, int], None]) -> None:
    """Attach an out-of-memory observer to MTIA memory allocator"""

def is_bf16_supported(including_emulation: bool = ...) -> Literal[True]: ...
def get_device_capability(device: _device_t | None = ...) -> tuple[int, int]:
    """
    Return capability of a given device as a tuple of (major version, minor version).

    Args:
        device (torch.device or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """

def empty_cache() -> None:
    """Empty the MTIA device cache."""

def set_stream(stream: Stream) -> None:
    """
    Set the current stream.This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """

def set_device(device: _device_t) -> None:
    """
    Set the current device.

    Args:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """

def get_device_properties(device: _device_t | None = ...) -> dict[str, Any]:
    """
    Return a dictionary of MTIA device properties

    Args:
        device (torch.device or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """

class device:
    """
    Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """
    def __init__(self, device: Any) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type: Any, value: Any, traceback: Any) -> Literal[False]: ...

class StreamContext:
    """
    Context-manager that selects a given stream.

    All MTIA kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """

    cur_stream: torch.mtia.Stream | None
    def __init__(self, stream: torch.mtia.Stream | None) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, type: Any, value: Any, traceback: Any) -> None: ...

def stream(stream: torch.mtia.Stream | None) -> StreamContext:
    """
    Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: In eager mode stream is of type Stream class while in JIT it doesn't support torch.mtia.stream
    """

def get_rng_state(device: int | str | torch.device = ...) -> Tensor:
    """
    Returns the random number generator state as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'mtia'`` (i.e., ``torch.device('mtia')``, the current mtia device).
    """

def set_rng_state(new_state: Tensor, device: int | str | torch.device = ...) -> None:
    """
    Sets the random number generator state.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'mtia'`` (i.e., ``torch.device('mtia')``, the current mtia device).
    """

__all__ = [
    "attach_out_of_memory_observer",
    "current_device",
    "current_stream",
    "default_stream",
    "device",
    "device_count",
    "empty_cache",
    "get_device_capability",
    "get_device_properties",
    "get_rng_state",
    "init",
    "is_available",
    "is_bf16_supported",
    "is_initialized",
    "max_memory_allocated",
    "memory_allocated",
    "memory_stats",
    "record_memory_history",
    "reset_peak_memory_stats",
    "set_device",
    "set_rng_state",
    "set_stream",
    "snapshot",
    "stream",
    "synchronize",
]
