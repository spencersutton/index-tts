import os
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch._C import _onnx as _C_onnx
from torch._C._onnx import OperatorExportTypes as OperatorExportTypes
from torch._C._onnx import TensorProtoDataType as TensorProtoDataType
from torch._C._onnx import TrainingMode as TrainingMode

from . import errors, ops
from ._internal.exporter._onnx_program import ONNXProgram
from .errors import OnnxExporterError

__all__ = [
    "ONNXProgram",
    "OnnxExporterError",
    "errors",
    "export",
    "is_in_onnx_export",
    "ops",
]
if TYPE_CHECKING: ...
producer_name = ...
producer_version = ...

def export(
    model: torch.nn.Module | torch.export.ExportedProgram | torch.jit.ScriptModule | torch.jit.ScriptFunction,
    args: tuple[Any, ...] = ...,
    f: str | os.PathLike | None = ...,
    *,
    kwargs: dict[str, Any] | None = ...,
    verbose: bool | None = ...,
    input_names: Sequence[str] | None = ...,
    output_names: Sequence[str] | None = ...,
    opset_version: int | None = ...,
    dynamo: bool = ...,
    external_data: bool = ...,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = ...,
    custom_translation_table: dict[Callable, Callable | Sequence[Callable]] | None = ...,
    report: bool = ...,
    optimize: bool = ...,
    verify: bool = ...,
    profile: bool = ...,
    dump_exported_program: bool = ...,
    artifacts_dir: str | os.PathLike = ...,
    fallback: bool = ...,
    export_params: bool = ...,
    keep_initializers_as_inputs: bool = ...,
    dynamic_axes: Mapping[str, Mapping[int, str]] | Mapping[str, Sequence[int]] | None = ...,
    training: _C_onnx.TrainingMode = ...,
    operator_export_type: _C_onnx.OperatorExportTypes = ...,
    do_constant_folding: bool = ...,
    custom_opsets: Mapping[str, int] | None = ...,
    export_modules_as_functions: bool | Collection[type[torch.nn.Module]] = ...,
    autograd_inlining: bool = ...,
) -> ONNXProgram | None: ...
def is_in_onnx_export() -> bool: ...
