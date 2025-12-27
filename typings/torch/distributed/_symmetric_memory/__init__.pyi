from collections.abc import Sequence
from datetime import timedelta
from enum import Enum
from typing import Any, Literal, overload

import torch
import torch.distributed.distributed_c10d as c10d
from torch._C._distributed_c10d import ProcessGroup, Work as _Work, _SymmetricMemory
from torch.types import _device, _dtype, _int

_group_name_to_store: dict[str, c10d.Store] = ...

def enable_symm_mem_for_group(group_name: str) -> None:
    """
    Enables symmetric memory for a process group.

    Args:
        group_name (str): the name of the process group.
    """

_is_test_mode: bool = ...
_mocked_group_names: set[str] | None = ...

def is_symm_mem_enabled_for_group(group_name: str) -> bool:
    """
    Check if symmetric memory is enabled for a process group.

    Args:
        group_name (str): the name of the process group.
    """

_group_name_to_workspace_tensor: dict[str, torch.Tensor | None] = ...

def get_symm_mem_workspace(group_name: str, min_size: int) -> _SymmetricMemory:
    """
    Get the symmetric memory workspace associated with the process group. If
    ``min_size`` is greater than the workspace associated with ``group_name``,
    the workspace will be re-allocated and re-rendezvous'd.

    Args:
        group_name (str): the name of the process group.
        min_size (int): the size requirement for the workspace in bytes.

    Returns:
        _SymmetricMemory: the symmetric memory workspace associated with the
        group.
    """

_backend_streams: dict[int, torch.cuda.Stream] = ...
lib = ...

class _ScaleMode(Enum):
    UNSCALED = ...
    TENSOR_WISE = ...
    ROW_WISE_SHARDED = ...
    ROW_WISE_REPLICATED = ...

def make_contiguous_for_perm(t: torch.Tensor, perm: list[int]) -> torch.Tensor:
    """Restride `t` such that `t.permute(perm)` is contiguous."""

def restride_A_shard_for_fused_all_gather_matmul(t: torch.Tensor, gather_dim: int) -> torch.Tensor:
    """
    Restride the `A_shard` arg of `fused_all_gather_matmul` for optimal perf.
    See the doc for `fused_all_gather_matmul` for detail.
    """

def restride_A_for_fused_matmul_reduce_scatter(t: torch.Tensor, scatter_dim: int) -> torch.Tensor:
    """
    Restride the `A_shard` arg of `fused_matmul_reduce_scatter` for optimal
    perf. See the doc for `fused_matmul_reduce_scatter` for detail.
    """

class Work(_Work):
    def __init__(self) -> None: ...
    def wait(self, timeout: timedelta = ...) -> bool: ...

@overload
def empty(*size: _int, dtype: _dtype | None = ..., device: _device | None = ...) -> torch.Tensor:
    """
    empty(*size, *, dtype=None, device=None) -> Tensor

    Similar to :func:`torch.empty()`. The returned tensor can be used by
    :func:`torch._distributed._symmetric_memory.rendezvous()` to establish a
    symmetric memory tensor among participating processes.

    Args:
        size (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        device (:class:`torch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_device`). :attr:`device` will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.
    """

@overload
def empty(size: Sequence[_int], *, dtype: _dtype | None = ..., device: _device | None = ...) -> torch.Tensor:
    """
    empty(*size, *, dtype=None, device=None) -> Tensor

    Similar to :func:`torch.empty()`. The returned tensor can be used by
    :func:`torch._distributed._symmetric_memory.rendezvous()` to establish a
    symmetric memory tensor among participating processes.

    Args:
        size (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        device (:class:`torch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_device`). :attr:`device` will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.
    """

def empty(*size: Any, dtype: _dtype | None = ..., device: _device | None = ...) -> torch.Tensor:
    """
    empty(*size, *, dtype=None, device=None) -> Tensor

    Similar to :func:`torch.empty()`. The returned tensor can be used by
    :func:`torch._distributed._symmetric_memory.rendezvous()` to establish a
    symmetric memory tensor among participating processes.

    Args:
        size (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        device (:class:`torch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_device`). :attr:`device` will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.
    """

def rendezvous(tensor: torch.Tensor, group: str | ProcessGroup) -> _SymmetricMemory:
    """
    rendezvous(tensor, group) -> _SymmetricMemory

    Establish a symmetric memory tensor among participating processes. This is
    a collective operation.

    Args:
        tensor (:class:`torch.Tensor`): the local tensor used to establish the symmetric memory tensor.
            It must be allocated via :func:`torch._distributed._symmetric_memory.empty()`. The shape,
            dtype, and device type must be identical across all participating processes.
        group (Union[str, :class:`torch.distributed.ProcessGroup`]): The group identifying the
            participating processes. This can be either a group name or a process group object.
    """

def is_nvshmem_available() -> bool:
    """
    is_nvshmem_available() -> bool

    Check if NVSHMEM is available in current build and on current system.
    """

def set_backend(name: Literal["NVSHMEM", "CUDA", "NCCL"]) -> None:
    """
    Set the backend for symmetric memory allocation. This is a global setting
    and affects all subsequent calls to
    :func:`torch._distributed._symmetric_memory.empty()`.  Note that the backend
    cannot be changed once a symmetric memory tensor has been allocated.

    Args:
        backend (str): the backend for symmetric memory allocation. Currently,
        only "NVSHMEM", "CUDA", "NCCL" are supported.
    """

def get_backend(device: _device) -> str | None:
    """
    Get the backend for symmetric memory allocation for a given device. If not
    found, return None.

    Args:
        device (class:`torch.device` or str): the device for which to get the
        backend.
    """

def get_mempool_allocator(device: _device):
    """
    Get the MemPool allocator for symmetric memory for a given device.
    Args:
        device (class:`torch.device` or str): the device for which to get the
        MemPool allocator.
    """

__all__ = ["empty", "get_backend", "is_nvshmem_available", "rendezvous", "set_backend"]
