"""
This module implements variable tracking for PyTorch optimizers during Dynamo tracing.

The OptimizerVariable class provides specialized handling for optimizer instances by:
- Optimizing the tracing of expensive optimizer initialization
- Managing optimizer state and parameter group tracking
- Handling tensor sources and guards for optimizer state tensors
- Supporting CUDA graph execution through static tensor address management
- Providing special handling for parameter gradients and optimizer state tensors

Key features include:
- Efficient initialization tracing via _init_group optimization
- Automatic marking of optimizer state tensors as static for CUDA graphs
- Proper source tracking for parameter groups, gradients, and state tensors
- Guard installation for optimizer state structure
- Support for both CPU and GPU tensor handling
- Cleanup of static tensor references via finalizers

The module integrates with Dynamo's broader tracing system while providing
optimizer-specific optimizations and safety guarantees.
"""

from torch._dynamo.symbolic_convert import InstructionTranslator

from .base import VariableTracker
from .user_defined import UserDefinedObjectVariable

class ArgMappingException(Exception): ...
class GuardInstallException(Exception): ...

perf_hint_log = ...

class OptimizerVariable(UserDefinedObjectVariable):
    _nonvar_fields = ...
    def __init__(self, value, grad_to_source=..., static_tensor_names=..., tensor_to_source=..., **kwargs) -> None: ...
    def call_method(self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker:
        """This is an optimization to avoid tracing the very slow initialization of the optimizer"""
    def var_getattr(self, tx: InstructionTranslator, name): ...
    def graph_break_if_pending_mutation(self, tx): ...
    def get_python_args(self, *args, **kwargs):
        """Get python values equivalent to the variable tracker args"""
    def move_step_if_cpu(self): ...
    def map_sources_and_install_guards(self, tx): ...
    def wrap_tensor(self, tx: InstructionTranslator, tensor_value):
        """Wrap state tensor in a TensorVariable"""
    def update_list_args(self, tx: InstructionTranslator, args, kwargs, py_args, py_kwargs):
        """Update the args and kwargs to the traced optimizer call"""
    def create_finalizer(self, tx): ...
