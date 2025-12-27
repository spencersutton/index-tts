"""
This module implements variable tracking for PyTorch nn.Module instances during Dynamo tracing.

It provides specialized handling for different types of nn.Module instances through several key classes:

- NNModuleVariable: Handles instance-specific module tracing, specializing on module id() and placing
  parameters directly on the torch.fx.GraphModule. This creates one graph per module instance.

- UnspecializedNNModuleVariable: Provides class-level module tracing, treating nn.Modules like other
  user-defined objects and passing parameters as inputs to the FX graph. This creates one graph per
  module class.

- UnspecializedBuiltinNNModuleVariable: Specifically handles built-in PyTorch modules (e.g. nn.Linear)
  with appropriate optimizations.

- FSDPManagedNNModuleVariable: Special handling for FSDP-wrapped modules with modified guarding behavior
  and parameter handling.

The module integrates with Dynamo's broader tracing functionality to handle module method calls,
parameter access, hooks, and other nn.Module behaviors while maintaining proper scoping and guarding
of module state.
"""

from contextlib import contextmanager

import torch.nn
from torch._dynamo.symbolic_convert import InstructionTranslator

from .base import VariableTracker
from .user_defined import UserDefinedObjectVariable

def initialize_lazy_module(tx: InstructionTranslator, mod, args, kwargs):
    """
    Fairly coupled helper used by NNModuleVariable and UnspecializedNNModuleVariable.

    Used to cause lazy module to be initialized (and delete its init hook) before tracing. Especially
    useful now that 'allowed' modules graph-break on hooks, calling this first ensures there is no hook
    by the time we trace __call__ and thus no graph-break for lazy allowed modules.
    """

@contextmanager
def record_nn_module_stack(module_key: str, source, tx, mod: torch.nn.Module): ...
def guard_to_detect_forward_monkeypatching(source, mod): ...

class NNModuleVariable(VariableTracker):
    _nonvar_fields = ...
    def __init__(self, module_type: type, module_key: str, value: torch.nn.Module, **kwargs) -> None: ...
    def get_nn_module_stack_source(self): ...
    def set_nn_module_stack_source(self, source): ...
    def python_type(self): ...
    def unpack_var_sequence(self, tx): ...
    def call_obj_hasattr(self, tx: InstructionTranslator, name: str) -> VariableTracker: ...
    def is_training(self, tx): ...
    def convert_to_unspecialized(self, tx):
        """Restart analysis treating this module as an UnspecializedNNModuleVariable"""
    def has_key_in_generic_dict(self, tx: InstructionTranslator, key): ...
    def var_getattr(self, tx: InstructionTranslator, name): ...
    def call_function(self, tx, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...
    def call_method(
        self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker], constant=...
    ) -> VariableTracker: ...

class UnspecializedNNModuleVariable(UserDefinedObjectVariable):
    _nonvar_fields = ...
    def __init__(self, value, **kwargs) -> None: ...
    def get_nn_module_stack_source(self): ...
    def set_nn_module_stack_source(self, source): ...
    def unpack_var_sequence(self, tx): ...
    def call_function(
        self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> VariableTracker: ...
    def call_method(
        self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> VariableTracker: ...
    def getattr_helper(self, tx: InstructionTranslator, field, name_vt): ...
    def var_getattr(self, tx: InstructionTranslator, name): ...
    def manually_trace_nn_module_getattr(self, tx: InstructionTranslator, name):
        """
        Dynamo tracing of nn.Module __getattr__ can be expensive if the model
        has deep submodule hierarchy. Since the __getattr__ is stable, we can
        directly look into the underlying datastructures. This saves a lot of
        compilation time.
        """

class UnspecializedBuiltinNNModuleVariable(UnspecializedNNModuleVariable):
    """Differentiates between builtin nn modules (e.g. torch.nn.Linear) and user defined nn modules."""

class FSDPManagedNNModuleVariable(UnspecializedNNModuleVariable):
    """
    Tracing behavior: trace into submodules and treat them as Unspecialized, do not
    register parameters to the top-level, treat them as function inputs.

    Guards behavior: if 'skip_fsdp_guards', many guards that would be installed
    by a vanilla UnspecializedNNModuleVariable are simply dropped, on the basis
    that a user wrapping their model in FSDP(model) is already opting into a
    requirement to not modify internal model state, which would already break FSDP without
    compilation.
    """
    def __init__(self, value, **kwargs) -> None: ...
