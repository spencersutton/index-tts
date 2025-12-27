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
    """ModuleCallSignature(inputs: list[typing.Union[torch.export.graph_signature.TensorArgument, torch.export.graph_signature.SymIntArgument, torch.export.graph_signature.SymFloatArgument, torch.export.graph_signature.SymBoolArgument, torch.export.graph_signature.ConstantArgument, torch.export.graph_signature.CustomObjArgument, torch.export.graph_signature.TokenArgument]], outputs: list[typing.Union[torch.export.graph_signature.TensorArgument, torch.export.graph_signature.SymIntArgument, torch.export.graph_signature.SymFloatArgument, torch.export.graph_signature.SymBoolArgument, torch.export.graph_signature.ConstantArgument, torch.export.graph_signature.CustomObjArgument, torch.export.graph_signature.TokenArgument]], in_spec: torch.utils._pytree.TreeSpec, out_spec: torch.utils._pytree.TreeSpec, forward_arg_names: Optional[list[str]] = None)"""

    inputs: list[ArgumentSpec]
    outputs: list[ArgumentSpec]
    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec
    forward_arg_names: list[str] | None = ...
    def replace_all_uses_with(self, original_node, new_node): ...

@dataclasses.dataclass
class ModuleCallEntry:
    """ModuleCallEntry(fqn: str, signature: Optional[torch.export.exported_program.ModuleCallSignature] = None)"""

    fqn: str
    signature: ModuleCallSignature | None = ...

_AUTOGRAD_ALIAS_BACKEND_KEYS_TO_OVERRIDE = ...
_BACKEND_KEYS_TO_OVERRIDE = ...

def default_decompositions() -> CustomDecompTable:
    """
    This is the default decomposition table which contains decomposition of
    all ATEN operators to core aten opset. Use this API together with
    :func:`run_decompositions()`
    """

class ExportedProgram:
    """
    Package of a program from :func:`export`. It contains
    an :class:`torch.fx.Graph` that represents Tensor computation, a state_dict containing
    tensor values of all lifted parameters and buffers, and various metadata.

    You can call an ExportedProgram like the original callable traced by
    :func:`export` with the same calling convention.

    To perform transformations on the graph, use ``.module`` property to access
    an :class:`torch.fx.GraphModule`. You can then use
    `FX transformation <https://pytorch.org/docs/stable/fx.html#writing-transformations>`_
    to rewrite the graph. Afterwards, you can simply use :func:`export`
    again to construct a correct ExportedProgram.
    """

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
    def graph_module(self):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @graph_module.setter
    @compatibility(is_backward_compatible=False)
    def graph_module(self, value):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @property
    @compatibility(is_backward_compatible=False)
    def graph(self):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @graph.setter
    @compatibility(is_backward_compatible=False)
    def graph(self, value):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @property
    @compatibility(is_backward_compatible=False)
    def graph_signature(self):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @graph_signature.setter
    @compatibility(is_backward_compatible=False)
    def graph_signature(self, value):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @property
    @compatibility(is_backward_compatible=False)
    def state_dict(self):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @state_dict.setter
    @compatibility(is_backward_compatible=False)
    def state_dict(self, value):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @compatibility(is_backward_compatible=False)
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """
        Returns an iterator over original module's parameters.

        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @compatibility(is_backward_compatible=False)
    def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]:
        """
        Returns an iterator over original module parameters, yielding
        both the name of the parameter as well as the parameter itself.

        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @compatibility(is_backward_compatible=False)
    def buffers(self) -> Iterator[torch.Tensor]:
        """
        Returns an iterator over original module buffers.

        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @compatibility(is_backward_compatible=False)
    def named_buffers(self) -> Iterator[tuple[str, torch.Tensor]]:
        """
        Returns an iterator over original module buffers, yielding
        both the name of the buffer as well as the buffer itself.

        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @property
    @compatibility(is_backward_compatible=False)
    def range_constraints(self):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @range_constraints.setter
    @compatibility(is_backward_compatible=False)
    def range_constraints(self, value):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @property
    @compatibility(is_backward_compatible=False)
    def module_call_graph(self):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @module_call_graph.setter
    @compatibility(is_backward_compatible=False)
    def module_call_graph(self, value):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @property
    @compatibility(is_backward_compatible=False)
    def example_inputs(self):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @example_inputs.setter
    @compatibility(is_backward_compatible=False)
    def example_inputs(self, value):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @property
    @compatibility(is_backward_compatible=False)
    def call_spec(self) -> None:
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @call_spec.setter
    @compatibility(is_backward_compatible=False)
    def call_spec(self, value):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @property
    @compatibility(is_backward_compatible=False)
    def verifier(self) -> Any:
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @verifier.setter
    @compatibility(is_backward_compatible=False)
    def verifier(self, value):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @property
    @compatibility(is_backward_compatible=False)
    def dialect(self) -> str:
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @dialect.setter
    @compatibility(is_backward_compatible=False)
    def dialect(self, value):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @property
    @compatibility(is_backward_compatible=False)
    def verifiers(self):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @verifiers.setter
    @compatibility(is_backward_compatible=False)
    def verifiers(self, value):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @property
    @compatibility(is_backward_compatible=False)
    def tensor_constants(self):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @tensor_constants.setter
    @compatibility(is_backward_compatible=False)
    def tensor_constants(self, value):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @property
    @compatibility(is_backward_compatible=False)
    def constants(self):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    @constants.setter
    @compatibility(is_backward_compatible=False)
    def constants(self, value):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def module(self, check_guards=...) -> torch.fx.GraphModule:
        """
        Returns a self contained GraphModule with all the parameters/buffers inlined.

        - When `check_guards=True` (default), a `_guards_fn` submodule is generated
          and a call to a `_guards_fn` submodule is inserted right after placeholders
          in the graph. This module checks guards on inputs.
        - When `check_guards=False`, a subset of these checks are performed by a
          forward pre-hook on the graph module. No `_guards_fn` submodule is generated.
        """
    @_disable_prexisiting_fake_mode
    def run_decompositions(
        self,
        decomp_table: dict[torch._ops.OperatorBase, Callable] | None = ...,
        decompose_custom_triton_ops: bool = ...,
    ) -> ExportedProgram:
        """
        Run a set of decompositions on the exported program and returns a new
        exported program. By default we will run the Core ATen decompositions to
        get operators in the
        `Core ATen Operator Set <https://pytorch.org/docs/stable/torch.compiler_ir.html>`_.

        For now, we do not decompose joint graphs.

        Args:
            decomp_table:
             An optional argument that specifies decomp behaviour for Aten ops
             (1) If None, we decompose to core aten decompositions
             (2) If empty, we don't decompose any operator


        Some examples:

        If you don't want to decompose anything

        .. code-block:: python

            ep = torch.export.export(model, ...)
            ep = ep.run_decompositions(decomp_table={})

        If you want to get a core aten operator set except for certain operator, you can do following:

        .. code-block:: python

            ep = torch.export.export(model, ...)
            decomp_table = torch.export.default_decompositions()
            decomp_table[your_op] = your_custom_decomp
            ep = ep.run_decompositions(decomp_table=decomp_table)
        """
    @compatibility(is_backward_compatible=False)
    def validate(self):
        """
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
        """
