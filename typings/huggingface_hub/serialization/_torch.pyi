import os
from collections import namedtuple
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import torch

from ._base import StateDictSplit

"""Contains pytorch-specific helpers."""
logger = ...

def save_torch_model(
    model: torch.nn.Module,
    save_directory: str | Path,
    *,
    filename_pattern: str | None = ...,
    force_contiguous: bool = ...,
    max_shard_size: int | str = ...,
    metadata: dict[str, str] | None = ...,
    safe_serialization: bool = ...,
    is_main_process: bool = ...,
    shared_tensors_to_discard: list[str] | None = ...,
):  # -> None:
    ...
def save_torch_state_dict(
    state_dict: dict[str, torch.Tensor],
    save_directory: str | Path,
    *,
    filename_pattern: str | None = ...,
    force_contiguous: bool = ...,
    max_shard_size: int | str = ...,
    metadata: dict[str, str] | None = ...,
    safe_serialization: bool = ...,
    is_main_process: bool = ...,
    shared_tensors_to_discard: list[str] | None = ...,
) -> None: ...
def split_torch_state_dict_into_shards(
    state_dict: dict[str, torch.Tensor], *, filename_pattern: str = ..., max_shard_size: int | str = ...
) -> StateDictSplit: ...
def load_torch_model(
    model: torch.nn.Module,
    checkpoint_path: str | os.PathLike,
    *,
    strict: bool = ...,
    safe: bool = ...,
    weights_only: bool = ...,
    map_location: str | torch.device | None = ...,
    mmap: bool = ...,
    filename_pattern: str | None = ...,
) -> NamedTuple: ...
def load_state_dict_from_file(
    checkpoint_file: str | os.PathLike,
    map_location: str | torch.device | None = ...,
    weights_only: bool = ...,
    mmap: bool = ...,
) -> dict[str, torch.Tensor] | Any: ...
def get_torch_storage_id(tensor: torch.Tensor) -> tuple[torch.device, int | tuple[Any, ...], int] | None: ...
def get_torch_storage_size(tensor: torch.Tensor) -> int: ...
@lru_cache
def is_torch_tpu_available(check_device=...):  # -> bool:
    ...
def storage_ptr(tensor: torch.Tensor) -> int | tuple[Any, ...]: ...

class _IncompatibleKeys(namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])):
    __str__ = ...
