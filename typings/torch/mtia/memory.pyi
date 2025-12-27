"""This package adds support for device memory management implemented in MTIA."""

from typing import Any

from . import _device_t

def memory_stats(device: _device_t | None = ...) -> dict[str, Any]:
    """
    Return a dictionary of MTIA memory allocator statistics for a given device.

    Args:
        device (torch.device, str, or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """

def max_memory_allocated(device: _device_t | None = ...) -> int:
    """
    Return the maximum memory allocated in bytes for a given device.

    Args:
        device (torch.device, str, or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """

def memory_allocated(device: _device_t | None = ...) -> int:
    """
    Return the current MTIA memory occupied by tensors in bytes for a given device.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mtia.current_device`,
            if :attr:`device` is ``None`` (default).
    """

def reset_peak_memory_stats(device: _device_t | None = ...) -> None:
    """
    Reset the peak memory stats for a given device.


    Args:
        device (torch.device, str, or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """

__all__ = ["max_memory_allocated", "memory_allocated", "memory_stats", "reset_peak_memory_stats"]
