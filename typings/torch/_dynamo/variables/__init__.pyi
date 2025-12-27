"""
This package implements variable tracking and symbolic execution capabilities for Dynamo,
which are essential for converting Python code into FX graphs. It provides a comprehensive
set of variable types that handle different Python constructs during tracing.

Each variable type (like BuiltinVariable, TensorVariable, NNModuleVariable, etc.) is responsible
for tracking and symbolically executing operations on specific Python objects. This enables
Dynamo to:
- Track the flow of values through Python code
- Maintain correct semantics during graph conversion
- Handle complex Python features like context managers, iterators, and custom objects
- Support both eager and symbolic execution modes

The VariableTracker base class provides the foundation for all variable types, with each
subclass implementing specific behavior for different Python constructs. This modular design
allows Dynamo to accurately trace and optimize Python code while preserving its semantics.
"""

from .base import VariableTracker
from .builtin import BuiltinVariable
from .constant import ConstantVariable, EnumVariable
from .ctx_manager import (
    CatchWarningsCtxManagerVariable,
    ContextWrappingVariable,
    CUDADeviceVariable,
    DeterministicAlgorithmsVariable,
    DynamoConfigPatchVariable,
    ErrorOnGraphBreakVariable,
    GradModeVariable,
    TemporarilyPopInterpreterStackCtxManagerVariable,
    WithExitFunctionVariable,
)
from .dicts import ConstDictVariable, DefaultDictVariable, DictKeySetVariable, MappingProxyVariable
from .distributed import BackwardHookVariable, PlacementVariable
from .functions import (
    CreateTMADescriptorExperimentalVariable,
    CreateTMADescriptorStableVariable,
    NestedUserFunctionVariable,
    PolyfilledFunctionVariable,
    SkipFunctionVariable,
    TMADescriptorExperimentalVariable,
    TMADescriptorStableVariable,
    UserFunctionVariable,
    UserMethodVariable,
)
from .iter import CountIteratorVariable, IteratorVariable, ItertoolsVariable, RepeatIteratorVariable
from .lazy import LazyVariableTracker
from .lists import (
    BaseListVariable,
    ListIteratorVariable,
    ListVariable,
    NamedTupleVariable,
    RangeVariable,
    SliceVariable,
    TupleVariable,
)
from .misc import (
    AutogradFunctionContextVariable,
    AutogradFunctionVariable,
    CellVariable,
    DeletedVariable,
    GetAttrVariable,
    LambdaVariable,
    NewGlobalVariable,
    NumpyVariable,
    PythonModuleVariable,
    RegexPatternVariable,
    StringFormatVariable,
    SuperVariable,
    TorchVersionVariable,
    UnknownVariable,
)
from .nn_module import NNModuleVariable, UnspecializedNNModuleVariable
from .optimizer import OptimizerVariable
from .sdpa import SDPAParamsVariable
from .tensor import (
    DataPtrVariable,
    FakeItemVariable,
    NumpyNdarrayVariable,
    TensorVariable,
    UnspecializedPythonVariable,
    UntypedStorageVariable,
)
from .torch import TorchCtxManagerClassVariable, TorchInGraphFunctionVariable
from .user_defined import (
    RemovableHandleVariable,
    UserDefinedClassVariable,
    UserDefinedObjectVariable,
    UserDefinedTupleVariable,
)

__all__ = [
    "AutogradFunctionContextVariable",
    "AutogradFunctionVariable",
    "BackwardHookVariable",
    "BaseListVariable",
    "BuiltinVariable",
    "CUDADeviceVariable",
    "CatchWarningsCtxManagerVariable",
    "CellVariable",
    "ConstDictVariable",
    "ConstantVariable",
    "ContextWrappingVariable",
    "CountIteratorVariable",
    "CreateTMADescriptorExperimentalVariable",
    "CreateTMADescriptorStableVariable",
    "DataPtrVariable",
    "DefaultDictVariable",
    "DeletedVariable",
    "DeterministicAlgorithmsVariable",
    "DictKeySetVariable",
    "DynamoConfigPatchVariable",
    "EnumVariable",
    "ErrorOnGraphBreakVariable",
    "ErrorOnGraphBreakVariable",
    "FakeItemVariable",
    "GetAttrVariable",
    "GradModeVariable",
    "IteratorVariable",
    "ItertoolsVariable",
    "LambdaVariable",
    "LazyVariableTracker",
    "ListIteratorVariable",
    "ListVariable",
    "MappingProxyVariable",
    "NNModuleVariable",
    "NamedTupleVariable",
    "NestedUserFunctionVariable",
    "NewGlobalVariable",
    "NumpyNdarrayVariable",
    "NumpyVariable",
    "OptimizerVariable",
    "PlacementVariable",
    "PolyfilledFunctionVariable",
    "PythonModuleVariable",
    "RangeVariable",
    "RegexPatternVariable",
    "RemovableHandleVariable",
    "RepeatIteratorVariable",
    "SDPAParamsVariable",
    "SkipFunctionVariable",
    "SliceVariable",
    "StringFormatVariable",
    "SuperVariable",
    "TMADescriptorExperimentalVariable",
    "TMADescriptorStableVariable",
    "TemporarilyPopInterpreterStackCtxManagerVariable",
    "TensorVariable",
    "TorchCtxManagerClassVariable",
    "TorchInGraphFunctionVariable",
    "TorchVersionVariable",
    "TupleVariable",
    "UnknownVariable",
    "UnspecializedNNModuleVariable",
    "UnspecializedPythonVariable",
    "UntypedStorageVariable",
    "UserDefinedClassVariable",
    "UserDefinedObjectVariable",
    "UserDefinedTupleVariable",
    "UserFunctionVariable",
    "UserMethodVariable",
    "VariableTracker",
    "WithExitFunctionVariable",
]
