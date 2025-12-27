"""
This package enables an interface for accessing MPS (Metal Performance Shaders) backend in Python.
Metal is Apple's API for programming metal GPU (graphics processor unit). Using MPS means that increased
performance can be achieved, by running work on the metal GPU(s).
See https://developer.apple.com/documentation/metalperformanceshaders for more details.
"""

import torch
from torch import Tensor

from . import profiler
from .event import Event

_is_in_bad_fork = ...
_default_mps_generator: torch._C.Generator = ...

def device_count() -> int:
    """Returns the number of available MPS devices."""

def synchronize() -> None:
    """Waits for all kernels in all streams on a MPS device to complete."""

def get_rng_state(device: int | str | torch.device = ...) -> Tensor:
    """
    Returns the random number generator state as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'mps'`` (i.e., ``torch.device('mps')``, the current MPS device).
    """

def set_rng_state(new_state: Tensor, device: int | str | torch.device = ...) -> None:
    """
    Sets the random number generator state.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'mps'`` (i.e., ``torch.device('mps')``, the current MPS device).
    """

def manual_seed(seed: int) -> None:
    """
    Sets the seed for generating random numbers.

    Args:
        seed (int): The desired seed.
    """

def seed() -> None:
    """Sets the seed for generating random numbers to a random number."""

def empty_cache() -> None:
    """
    Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU applications.
    """

def set_per_process_memory_fraction(fraction) -> None:
    """
    Set memory fraction for limiting process's memory allocation on MPS device.
    The allowed value equals the fraction multiplied by recommended maximum device memory
    (obtained from Metal API device.recommendedMaxWorkingSetSize).
    If trying to allocate more than the allowed value in a process, it will raise an out of
    memory error in allocator.

    Args:
        fraction(float): Range: 0~2. Allowed memory equals total_memory * fraction.

    .. note::
       Passing 0 to fraction means unlimited allocations
       (may cause system failure if out of memory).
       Passing fraction greater than 1.0 allows limits beyond the value
       returned from device.recommendedMaxWorkingSetSize.
    """

def current_allocated_memory() -> int:
    """
    Returns the current GPU memory occupied by tensors in bytes.

    .. note::
       The returned size does not include cached allocations in
       memory pools of MPSAllocator.
    """

def driver_allocated_memory() -> int:
    """
    Returns total GPU memory allocated by Metal driver for the process in bytes.

    .. note::
       The returned size includes cached allocations in MPSAllocator pools
       as well as allocations from MPS/MPSGraph frameworks.
    """

def recommended_max_memory() -> int:
    """
    Returns recommended max Working set size for GPU memory in bytes.

    .. note::
       Recommended max working set size for Metal.
       returned from device.recommendedMaxWorkingSetSize.
    """

def compile_shader(source: str):
    """
    Compiles compute shader from source and allows one to invoke kernels
    defined there from the comfort of Python runtime
    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_MPS)
        >>> lib = torch.mps.compile_shader(
        ... "kernel void full(device float* out, constant float& val, uint idx [[thread_position_in_grid]]) { out[idx] = val; }"
        ...  )
        >>> x = torch.zeros(16, device="mps")
        >>> lib.full(x, 3.14)
    """

def is_available() -> bool: ...

__all__ = [
    "Event",
    "compile_shader",
    "current_allocated_memory",
    "device_count",
    "driver_allocated_memory",
    "empty_cache",
    "get_rng_state",
    "is_available",
    "manual_seed",
    "profiler",
    "recommended_max_memory",
    "seed",
    "set_per_process_memory_fraction",
    "set_rng_state",
    "synchronize",
]
