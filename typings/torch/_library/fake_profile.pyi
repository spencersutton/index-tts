import contextlib
import torch
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Optional, Union
from torch.types import FileLike

log = ...

class MissingOpProfile(RuntimeError): ...

@dataclass(frozen=True)
class TensorMetadata:
    rank: int
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    @staticmethod
    def maybe_from_tensor(t: Any) -> Optional[TensorMetadata]: ...

@dataclass(frozen=True)
class OpProfile:
    args_profile: tuple[Optional[TensorMetadata]]
    out_profile: Union[TensorMetadata, tuple[TensorMetadata]]

@contextlib.contextmanager
def unsafe_generate_fake_kernels(op_profiles: dict[str, set[OpProfile]]) -> Generator: ...
def get_torch_version() -> str: ...
def generate_yaml_from_profiles(op_profiles: dict[str, set[OpProfile]]) -> str: ...
def save_op_profiles(op_profiles: dict[str, set[OpProfile]], f: FileLike) -> None: ...
def read_profiles_from_yaml(yaml_str: str) -> dict[str, set[OpProfile]]: ...
def load_op_profiles(f: FileLike) -> dict[str, set[OpProfile]]: ...
