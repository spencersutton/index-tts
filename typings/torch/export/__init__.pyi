import logging
from collections.abc import Callable, Mapping
from typing import Any
from warnings import deprecated

import torch
from torch.fx.passes.infra.pass_base import PassResult
from torch.types import FileLike

from .decomp_utils import CustomDecompTable
from .dynamic_shapes import AdditionalInputs, Constraint, Dim, ShapesCollection, dims
from .exported_program import ExportedProgram, ModuleCallEntry, ModuleCallSignature, default_decompositions
from .graph_signature import ExportBackwardSignature, ExportGraphSignature
from .unflatten import FlatArgsAdapter, UnflattenedModule, unflatten

__all__ = [
    "AdditionalInputs",
    "Constraint",
    "CustomDecompTable",
    "Dim",
    "ExportBackwardSignature",
    "ExportGraphSignature",
    "ExportedProgram",
    "FlatArgsAdapter",
    "ModuleCallEntry",
    "ModuleCallSignature",
    "ShapesCollection",
    "UnflattenedModule",
    "default_decompositions",
    "dims",
    "draft_export",
    "export",
    "export_for_training",
    "load",
    "register_dataclass",
    "save",
    "unflatten",
]
type PassType = Callable[[torch.fx.GraphModule], PassResult | None]
log: logging.Logger = ...

@deprecated(
    "`torch.export.export_for_training` is deprecated and will be removed in PyTorch 2.10. Please use `torch.export.export` instead, which is functionally equivalent.",
    category=FutureWarning,
)
def export_for_training(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any] | None = ...,
    *,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = ...,
    strict: bool = ...,
    preserve_module_call_signature: tuple[str, ...] = ...,
    prefer_deferred_runtime_asserts_over_guards: bool = ...,
) -> ExportedProgram: ...
def export(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any] | None = ...,
    *,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = ...,
    strict: bool = ...,
    preserve_module_call_signature: tuple[str, ...] = ...,
    prefer_deferred_runtime_asserts_over_guards: bool = ...,
) -> ExportedProgram: ...

DEFAULT_PICKLE_PROTOCOL = ...

def save(
    ep: ExportedProgram,
    f: FileLike,
    *,
    extra_files: dict[str, Any] | None = ...,
    opset_version: dict[str, int] | None = ...,
    pickle_protocol: int = ...,
) -> None: ...
def load(
    f: FileLike, *, extra_files: dict[str, Any] | None = ..., expected_opset_version: dict[str, int] | None = ...
) -> ExportedProgram: ...
def draft_export(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any] | None = ...,
    *,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = ...,
    preserve_module_call_signature: tuple[str, ...] = ...,
    strict: bool = ...,
    prefer_deferred_runtime_asserts_over_guards: bool = ...,
) -> ExportedProgram: ...
def register_dataclass(cls: type[Any], *, serialized_type_name: str | None = ...) -> None: ...
