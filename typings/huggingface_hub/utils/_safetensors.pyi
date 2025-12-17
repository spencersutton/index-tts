from dataclasses import dataclass
from typing import Literal

FILENAME_T = str
TENSOR_NAME_T = str
type DTYPE_T = Literal["F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8", "BOOL"]

@dataclass
class TensorInfo:
    dtype: DTYPE_T
    shape: list[int]
    data_offsets: tuple[int, int]
    parameter_count: int = ...
    def __post_init__(self) -> None: ...

@dataclass
class SafetensorsFileMetadata:
    metadata: dict[str, str]
    tensors: dict[TENSOR_NAME_T, TensorInfo]
    parameter_count: dict[DTYPE_T, int] = ...
    def __post_init__(self) -> None: ...

@dataclass
class SafetensorsRepoMetadata:
    metadata: dict | None
    sharded: bool
    weight_map: dict[TENSOR_NAME_T, FILENAME_T]
    files_metadata: dict[FILENAME_T, SafetensorsFileMetadata]
    parameter_count: dict[DTYPE_T, int] = ...
    def __post_init__(self) -> None: ...
