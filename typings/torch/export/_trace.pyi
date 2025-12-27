import dataclasses
from collections import UserDict
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx._symbolic_trace import _ConstantAttributeType
from torch.utils._pytree import TreeSpec

from .exported_program import ExportedProgram
from .graph_signature import ExportGraphSignature

log = ...
NONSTRICT_EXPORT_SANITIZE_TRACE = ...
type _DynamicShapesSpec = dict[str, Any] | tuple[Any, ...] | list[Any]

@dataclasses.dataclass
class ExportDynamoConfig:
    """Manage Export-specific configurations of Dynamo."""

    allow_rnn: bool = ...
    reorderable_logging_functions: set[Callable] = ...
    do_not_emit_runtime_asserts: bool = ...
    specialize_int: bool = ...
    specialize_float: bool = ...
    assume_static_by_default: bool = ...
    automatic_dynamic_shapes: bool = ...
    capture_dynamic_output_shape_ops: bool = ...
    capture_scalar_outputs: bool = ...
    prefer_deferred_runtime_asserts_over_guards: bool = ...

@dataclasses.dataclass
class ATenExportArtifact:
    """ATenExportArtifact(gm: torch.fx.graph_module.GraphModule, sig: torch.export.graph_signature.ExportGraphSignature, constants: dict[str, typing.Union[torch.Tensor, torch.ScriptObject, torch._library.fake_class_registry.FakeScriptObject, torch.utils._pytree.TreeSpec]])"""

    gm: torch.fx.GraphModule
    sig: ExportGraphSignature
    constants: dict[str, _ConstantAttributeType]

@dataclasses.dataclass(frozen=True)
class ExportArtifact:
    """ExportArtifact(aten: torch.export._trace.ATenExportArtifact, in_spec: torch.utils._pytree.TreeSpec, out_spec: torch.utils._pytree.TreeSpec, fake_mode: torch._subclasses.fake_tensor.FakeTensorMode, module_call_specs: dict[str, dict[str, torch.utils._pytree.TreeSpec]])"""

    aten: ATenExportArtifact
    in_spec: TreeSpec
    out_spec: TreeSpec
    fake_mode: FakeTensorMode
    module_call_specs: dict[str, dict[str, pytree.TreeSpec]]

DEFAULT_EXPORT_DYNAMO_CONFIG = ...

def custom_triton_ops_decomposition_disabled(): ...
def detect_shape_env(inputs: Any = ...): ...

class _ExportModuleSpecTrackerDict(UserDict): ...

def get_ep_stats(ep: ExportedProgram) -> dict[str, Any]: ...

_EXPORT_FLAGS: set[str] | None = ...
_EXPORT_MODULE_HIERARCHY: dict[str, str] | None = ...

@contextmanager
def patch_forward(obj: torch.nn.Module, new_method):
    """
    Helper method to make it easier to cleanly torch.export() a method on a
    module that is not `forward`.
    """

def set_missing_meta_vals(gm, flat_args, num_params_buffers): ...
