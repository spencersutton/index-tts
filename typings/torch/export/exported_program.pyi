import dataclasses
from collections.abc import Callable, Iterator
from typing import Any

import sympy
import torch
import torch.utils._pytree as pytree
from torch._export.verifier import Verifier
from torch.export.decomp_utils import CustomDecompTable
from torch.fx._compatibility import compatibility
from torch.fx._symbolic_trace import _ConstantAttributeType
from torch.fx.passes.infra.pass_base import PassResult
from torch.utils._sympy.value_ranges import ValueRanges

from .graph_signature import ArgumentSpec, ExportGraphSignature

__all__ = ["ExportedProgram", "ModuleCallEntry", "ModuleCallSignature", "default_decompositions"]
type PassType = Callable[[torch.fx.GraphModule], PassResult | None]

@dataclasses.dataclass
class ModuleCallSignature:
    inputs: list[ArgumentSpec]
    outputs: list[ArgumentSpec]
    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec
    forward_arg_names: list[str] | None = ...
    def replace_all_uses_with(self, original_node, new_node): ...

@dataclasses.dataclass
class ModuleCallEntry:
    fqn: str
    signature: ModuleCallSignature | None = ...

_AUTOGRAD_ALIAS_BACKEND_KEYS_TO_OVERRIDE = ...
_BACKEND_KEYS_TO_OVERRIDE = ...

def default_decompositions() -> CustomDecompTable: ...

class ExportedProgram:
    _graph_module: torch.fx.GraphModule
    _graph_signature: ExportGraphSignature
    _state_dict: dict[str, Any]
    _range_constraints: dict[sympy.Symbol, ValueRanges]
    _module_call_graph: list[ModuleCallEntry]
    _example_inputs: tuple[tuple[Any, ...], dict[str, Any]] | None
    _constants: dict[str, _ConstantAttributeType]
    _verifiers: list[type[Verifier]]
    _guards_code: list[str]
    def __init__(
        self,
        root: torch.nn.Module | dict[str, Any],
        graph: torch.fx.Graph,
        graph_signature: ExportGraphSignature,
        state_dict: dict[str, torch.Tensor | torch.nn.Parameter],
        range_constraints: dict[sympy.Symbol, Any],
        module_call_graph: list[ModuleCallEntry],
        example_inputs: tuple[tuple[Any, ...], dict[str, Any]] | None = ...,
        constants: dict[str, _ConstantAttributeType] | None = ...,
        *,
        verifiers: list[type[Verifier]] | None = ...,
    ) -> None: ...
    @property
    @compatibility(is_backward_compatible=False)
    def graph_module(self): ...
    @graph_module.setter
    @compatibility(is_backward_compatible=False)
    def graph_module(self, value): ...
    @property
    @compatibility(is_backward_compatible=False)
    def graph(self): ...
    @graph.setter
    @compatibility(is_backward_compatible=False)
    def graph(self, value): ...
    @property
    @compatibility(is_backward_compatible=False)
    def graph_signature(self): ...
    @graph_signature.setter
    @compatibility(is_backward_compatible=False)
    def graph_signature(self, value): ...
    @property
    @compatibility(is_backward_compatible=False)
    def state_dict(self): ...
    @state_dict.setter
    @compatibility(is_backward_compatible=False)
    def state_dict(self, value): ...
    @compatibility(is_backward_compatible=False)
    def parameters(self) -> Iterator[torch.nn.Parameter]: ...
    @compatibility(is_backward_compatible=False)
    def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]: ...
    @compatibility(is_backward_compatible=False)
    def buffers(self) -> Iterator[torch.Tensor]: ...
    @compatibility(is_backward_compatible=False)
    def named_buffers(self) -> Iterator[tuple[str, torch.Tensor]]: ...
    @property
    @compatibility(is_backward_compatible=False)
    def range_constraints(self): ...
    @range_constraints.setter
    @compatibility(is_backward_compatible=False)
    def range_constraints(self, value): ...
    @property
    @compatibility(is_backward_compatible=False)
    def module_call_graph(self): ...
    @module_call_graph.setter
    @compatibility(is_backward_compatible=False)
    def module_call_graph(self, value): ...
    @property
    @compatibility(is_backward_compatible=False)
    def example_inputs(self): ...
    @example_inputs.setter
    @compatibility(is_backward_compatible=False)
    def example_inputs(self, value): ...
    @property
    @compatibility(is_backward_compatible=False)
    def call_spec(self) -> None: ...
    @call_spec.setter
    @compatibility(is_backward_compatible=False)
    def call_spec(self, value): ...
    @property
    @compatibility(is_backward_compatible=False)
    def verifier(self) -> Any: ...
    @verifier.setter
    @compatibility(is_backward_compatible=False)
    def verifier(self, value): ...
    @property
    @compatibility(is_backward_compatible=False)
    def dialect(self) -> str: ...
    @dialect.setter
    @compatibility(is_backward_compatible=False)
    def dialect(self, value): ...
    @property
    @compatibility(is_backward_compatible=False)
    def verifiers(self): ...
    @verifiers.setter
    @compatibility(is_backward_compatible=False)
    def verifiers(self, value): ...
    @property
    @compatibility(is_backward_compatible=False)
    def tensor_constants(self): ...
    @tensor_constants.setter
    @compatibility(is_backward_compatible=False)
    def tensor_constants(self, value): ...
    @property
    @compatibility(is_backward_compatible=False)
    def constants(self): ...
    @constants.setter
    @compatibility(is_backward_compatible=False)
    def constants(self, value): ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def module(self, check_guards=...) -> torch.fx.GraphModule: ...
    @_disable_prexisiting_fake_mode
    def run_decompositions(
        self,
        decomp_table: dict[torch._ops.OperatorBase, Callable] | None = ...,
        decompose_custom_triton_ops: bool = ...,
    ) -> ExportedProgram: ...
    @compatibility(is_backward_compatible=False)
    def validate(self): ...
