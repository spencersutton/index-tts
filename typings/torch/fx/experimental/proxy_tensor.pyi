import threading
import types
from collections.abc import Callable, Generator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ParamSpec, Protocol, Self, TypeVar, TypeVarTuple, Unpack, overload

import sympy
import torch
from torch import Tensor, fx
from torch._library.fake_class_registry import FakeScriptObject
from torch._ops import OpOverload
from torch.fx import GraphModule, Proxy, Tracer
from torch.fx.node import Argument, Target
from torch.nn import Module
from torch.overrides import TorchFunctionMode
from torch.types import PySymType
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._stats import count
from torch.utils._thunk import Thunk

from ._backward_state import BackwardState

__all__ = [
    "DecompositionInterpreter",
    "PythonKeyTracer",
    "dispatch_trace",
    "get_innermost_proxy_mode",
    "get_proxy_mode",
    "handle_sym_dispatch",
    "make_fx",
    "maybe_disable_thunkify",
    "maybe_enable_thunkify",
    "py_sym_types",
]
type _ProxyTracer = PythonKeyTracer | _GraphAppendingTracerEx
_AnyScriptObject = ...
type _AnyScriptObjectType = torch.ScriptObject | FakeScriptObject
aten = ...
prim = ...
log = ...
not_implemented_log = ...
CURRENT_DECOMPOSITION_TABLE: Mapping[OpOverload, Callable] = ...
CONSTANT_NUMEL_LIMIT = ...
T = TypeVar("T")
U = TypeVar("U")
_P = ParamSpec("_P")
R = TypeVar("R")
_Ts = TypeVarTuple("_Ts")
null_ctx_type = ...
_pytree_subclasses_that_lose_info = ...

def fake_signature[P, R](fn: Callable[_P, R], nargs: int) -> Callable[_P, R]:
    """FX gets confused by varargs, de-confuse it"""

@contextmanager
def decompose(
    decomposition_table: Mapping[OpOverload, Callable] | None,
) -> Generator[Mapping[OpOverload, Callable]]: ...

proxy_slot = ...

class _NoDefault: ...

no_default = ...

class _HasMeta(Protocol):
    meta: dict[str, PySymType]

def is_sym_node(node: _HasMeta) -> bool: ...
@overload
def set_proxy_slot(obj: Tensor, tracer: _ProxyTracer, proxy: _ProxyTensor) -> None: ...
@overload
def set_proxy_slot(obj: _AnyScriptObjectType, tracer: _ProxyTracer, proxy: Proxy) -> None: ...
@overload
def set_proxy_slot(obj: PySymType, tracer: _ProxyTracer, proxy: _PySymProxyType) -> None: ...

class _DisableUpdateTensorTracker(threading.local):
    value: bool = ...

_disable_update_tensor_tracker_tls = ...
_FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT: dict[int, torch.fx.Node] = ...

def set_proxy_slot(obj: PySymType | _AnyScriptObjectType | Tensor, tracer: _ProxyTracer, proxy: object) -> None: ...
def has_proxy_slot(obj: Tensor, tracer: _ProxyTracer) -> bool: ...

type _PySymProxyType = Thunk[Proxy]

@overload
def get_proxy_slot(obj: Tensor, tracer: _ProxyTracer) -> _ProxyTensor: ...
@overload
def get_proxy_slot[U](obj: Tensor, tracer: _ProxyTracer, default: U) -> _ProxyTensor | U: ...
@overload
def get_proxy_slot[U, R](
    obj: Tensor, tracer: _ProxyTracer, default: U, transform: Callable[[_ProxyTensor], R]
) -> R | U: ...
@overload
def get_proxy_slot(obj: _AnyScriptObjectType, tracer: _ProxyTracer) -> Proxy: ...
@overload
def get_proxy_slot[U](obj: _AnyScriptObjectType, tracer: _ProxyTracer, default: U) -> Proxy | U: ...
@overload
def get_proxy_slot[U, R](
    obj: _AnyScriptObjectType, tracer: _ProxyTracer, default: U, transform: Callable[[Proxy], R]
) -> R | U: ...
@overload
def get_proxy_slot(obj: PySymType, tracer: _ProxyTracer) -> _PySymProxyType: ...
@overload
def get_proxy_slot[T](obj: PySymType, tracer: _ProxyTracer, default: T) -> T | _PySymProxyType: ...
@overload
def get_proxy_slot[U, R](
    obj: PySymType, tracer: _ProxyTracer, default: U, transform: Callable[[_PySymProxyType], R]
) -> R | U: ...
def get_proxy_slot(
    obj: Tensor | _AnyScriptObjectType | PySymType,
    tracer: _ProxyTracer,
    default: object = ...,
    transform: Callable = ...,
) -> object: ...
def snapshot_fake(val: Tensor, include_real: bool = ...) -> Tensor | None: ...

type _ExtractValType = (
    PySymType
    | _AnyScriptObjectType
    | BackwardState
    | list[_ExtractValType]
    | tuple[_ExtractValType, ...]
    | dict[str, _ExtractValType]
    | Tensor
    | int
    | float
    | bool
    | None
)

def extract_val(val: _ExtractValType, include_real: bool = ...) -> _ExtractValType: ...
@contextmanager
def maybe_disable_thunkify() -> Generator[None]:
    """
    Within a context, disable thunkification.  See :func:`maybe_enable_thunkify`
    for more details.  This is helpful if you have a wrapper function which
    you want to enable thunkification on, but in some segment on the inside (say,
    the original user function), you want to disable thunkification as you know
    it is not needed there.
    """

@contextmanager
def maybe_enable_thunkify() -> Generator[None]:
    """
    Within this context manager, if you are doing make_fx tracing, we will thunkify
    all SymNode compute and avoid tracing it into the graph unless it is actually needed.
    You should prefer to avoid using this as much as possible, as lazy evaluation of
    SymNode tracing can lead to long chains of thunks which will stack overflow
    if you evaluate them.  However, this is currently sometimes necessary as there
    are buggy parts of PT2 which will fail with "s0 is not tracked with proxy" error
    due to insufficient tracing of SymNode computation.
    """

def set_meta(proxy: Proxy, val: _ExtractValType) -> Proxy: ...
def thunkify(tracer: _ProxyTracer, f: Callable[_P, R], *args: _P.args, **kwargs: _P.kwargs) -> Thunk[R]:
    """
    Delays computation of f until it's called again
    Also caches the result
    """

def track_tensor(tensor: Tensor, proxy: Proxy, *, constant: Tensor | None, tracer: _ProxyTracer) -> None: ...

type _NestedProxys = Proxy | Sequence[_NestedProxys] | Mapping[object, _NestedProxys]
type _NestedTensors = Tensor | Sequence[_NestedTensors] | Mapping[object, _NestedTensors]

def track_tensor_tree[T](
    inner_res: T, proxy_res: _NestedProxys, *, constant: _NestedTensors | None, tracer: _ProxyTracer
) -> T: ...

@dataclass
class _ProxyTensor:
    """_ProxyTensor(proxy: 'Proxy', constant: 'Optional[Tensor]')"""

    proxy: Proxy
    constant: Tensor | None

def fetch_sym_proxy(tracer: _ProxyTracer) -> Callable[[PySymType], bool | int | float | Proxy]: ...
@overload
def fetch_object_proxy(tracer: _ProxyTracer, t: Tensor) -> _ProxyTensor | Tensor: ...
@overload
def fetch_object_proxy(tracer: _ProxyTracer, t: _AnyScriptObjectType) -> Proxy | _AnyScriptObjectType: ...
@overload
def fetch_object_proxy(tracer: _ProxyTracer, t: PySymType) -> _PySymProxyType | PySymType: ...
def fetch_object_proxy(tracer: _ProxyTracer, t: Tensor | _AnyScriptObjectType | PySymType) -> object: ...

HANDLED_TYPES = ...

def proxy_call(
    proxy_mode: ProxyTorchDispatchMode,
    func: OpOverload,
    pre_dispatch: bool,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object: ...

class _SymNodeDict:
    """Wrapper around a dictionary that will hash SymInts with their nodes"""
    def __init__(self) -> None: ...
    def __setitem__(self, key: PySymType, value: _PySymProxyType) -> None: ...
    def __getitem__(self, key: PySymType) -> _PySymProxyType: ...
    def __contains__(self, key: PySymType) -> bool: ...
    def get(self, key: PySymType, default: _PySymProxyType | None = ...) -> _PySymProxyType: ...
    def __iter__(self) -> Any: ...
    def __len__(self) -> int: ...

class PythonKeyTracer(Tracer):
    script_object_tracker: MutableMapping[_AnyScriptObjectType, Proxy]
    symnode_tracker: _SymNodeDict
    sympy_expr_tracker: dict[sympy.Symbol, object]
    tensor_tracker: MutableMapping[Tensor, _ProxyTensor]
    torch_fn_counts: dict[OpOverload, int]
    enable_thunkify: bool = ...
    def __init__(self) -> None: ...
    def call_module(
        self, m: Module, forward: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any: ...
    def getattr(self, attr: str, attr_val: object, parameter_proxy_cache: dict[str, Proxy]) -> object: ...
    def create_arg(self, a: object) -> fx.node.Node: ...
    @overload
    def unwrap_proxy(self, e: Tensor) -> Proxy | Tensor: ...
    @overload
    def unwrap_proxy(self, e: PySymType) -> Proxy | PySymType: ...
    @overload
    def unwrap_proxy(self, e: _AnyScriptObjectType) -> Proxy | _AnyScriptObjectType: ...
    def unwrap_proxy(self, e: T) -> object: ...
    def create_node(
        self,
        kind: str,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        name: str | None = ...,
        type_expr: Any | None = ...,
    ) -> torch.fx.Node: ...

@torch._disable_dynamo
def dispatch_trace(
    root: Module | Callable, tracer: Tracer, concrete_args: tuple[Any, ...] | None = ...
) -> GraphModule: ...
def wrap_key(
    f: Callable[[Unpack[_Ts]], R], tensors: tuple[*_Ts], tracer: _ProxyTracer, pre_dispatch: bool
) -> Callable[_P, R]: ...

ORIGINAL_ATEN: object | None = ...

@contextmanager
def set_original_aten_op(func: OpOverload) -> Generator[None]: ...

class TorchFunctionMetadataMode(TorchFunctionMode):
    def __init__(self, tracer: _ProxyTracer) -> None: ...
    def __torch_function__(
        self,
        func: OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = ...,
        kwargs: dict[str, object] | None = ...,
    ) -> object: ...

_temp_remove_metadata_torch_function_mode = ...

class PreDispatchTorchFunctionMode(TorchFunctionMode):
    def __init__(self, tracer: _ProxyTracer) -> None: ...
    def __torch_function__(
        self,
        func: OpOverload | Callable,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = ...,
        kwargs: dict[str, object] | None = ...,
    ) -> object: ...

_temp_remove_pre_dispatch_torch_function_mode = ...

class ProxyTorchDispatchMode(TorchDispatchMode):
    @property
    def enable_tracing(self) -> bool: ...
    def __init__(
        self,
        tracer: _ProxyTracer,
        tracing_mode: str,
        pre_dispatch: bool = ...,
        _allow_fake_constant: bool = ...,
        _error_on_data_dependent_ops: bool = ...,
    ) -> None: ...
    @count
    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = ...,
        kwargs: dict[str, object] | None = ...,
    ) -> object: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool | None: ...
    @classmethod
    def is_infra_mode(cls) -> bool: ...
    def __sym_dispatch__(
        self,
        func: OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object: ...

class _GraphAppendingTracerEx(fx.proxy.GraphAppendingTracer):
    script_object_tracker: MutableMapping[_AnyScriptObjectType, Proxy]
    symnode_tracker: MutableMapping[PySymType, _PySymProxyType]
    tensor_tracker: MutableMapping[Tensor, _ProxyTensor]
    sympy_expr_tracker: dict[sympy.Symbol, object]
    torch_fn_metadata: OpOverload | None
    torch_fn_counts: dict[OpOverload, int]
    enable_thunkify: bool = ...
    def __init__(self, graph: fx.graph.Graph) -> None: ...

class DecompositionInterpreter(fx.Interpreter):
    def __init__(
        self,
        module: fx.GraphModule,
        new_graph: fx.Graph,
        decomposition_table: Mapping[OpOverload, Callable] | None = ...,
        **kwargs: object,
    ) -> None: ...
    def placeholder(self, target: str, args: tuple[object, ...], kwargs: dict[str, object]) -> object: ...
    def get_attr(self, target: str, args: tuple[object, ...], kwargs: dict[str, object]) -> object: ...
    def output(self, target: str, args: tuple[object, ...], kwargs: dict[str, object]) -> object: ...
    def run(self, *args: object, **kwargs: object) -> object: ...

def wrapper_and_args_for_make_fx[R](
    func: Callable[..., R], args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[Callable[[list[object]], R], list[object]]: ...
@contextmanager
def disable_autocast_cache() -> Generator[None]: ...

class _ModuleNotInstalledAsSubmoduleError(NameError): ...

class _AttrProxy:
    def reset_proxy_mapping(self, base: Module, path: str) -> None: ...

class _ModuleStackTracer(PythonKeyTracer):
    r"""
    Customized version of PythonKeyTracer that retains module stack
    information in node.meta["nn_module_stack"].

    FX symbolic trace actually does this already, but it relies on `self.root`
    being the actual module being traced. Since make_fx traces a lambda of our
    creation, things don't work properly.

    So for this version we hold onto a reference to the original module
    (scope_root) and use that to match the path. Also when we see,
            A
           / \
          B   C
           \ /
            D
    we want to record the path as A.B.D by recording only one path.
    See Note [Preserving the nn module stack metadata during export non-strict mode]  # noqa: W605
    """
    def __init__(self, scope_root: GraphModule) -> None: ...
    def path_of_module(self, mod: Module) -> str:
        """
        Use tracked access path during tracing instead of the default BFS behavior.
        Still use all the possible module paths to verify the result.
        """
    def getattr(self, attr: str, attr_val: object, parameter_proxy_cache: dict[str, Proxy]) -> object: ...
    def trace(self, root: Module | Callable, concrete_args: dict[str, object] | None) -> fx.Graph: ...
    def call_module(self, m: Module, forward: Callable, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        """
        PythonKeyTracer overrides call_module to avoid the scope handling,
        but we actually want it.
        """
    def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool: ...
    def create_node(self, *args: object, **kwargs: object) -> fx.node.Node:
        """
        Create node and add on metadata.
        Add nn_module_stack here instead of TracerBase,
        since calls to make_fx() might not want to record module stack metadata.
        Add torch_fn by looking at torch_fn_metadata and torch_fn_counts.
        Add stack_trace by filtering out forward() stack frames.
        """

class _MakefxTracer:
    def __init__(
        self,
        decomposition_table: Mapping[OpOverload, Callable] | None,
        tracing_mode: str,
        _allow_non_fake_inputs: bool,
        pre_dispatch: bool,
        record_module_stack: bool,
        _allow_fake_constant: bool,
        _error_on_data_dependent_ops: bool,
        record_stack_traces: bool = ...,
        parent_tracer: _MakefxTracer | None = ...,
    ) -> None: ...
    def trace(self, f: Callable, *args: object) -> fx.GraphModule: ...
    def is_hop_subgraph_tracer(self) -> bool: ...
    def trace_subgraph(self, f: Callable, *args: object) -> GraphModule: ...

_CURRENT_MAKE_FX_TRACER: _MakefxTracer | None = ...

def make_fx(
    f: Callable,
    decomposition_table: Mapping[OpOverload, Callable] | None = ...,
    tracing_mode: str = ...,
    _allow_non_fake_inputs: bool = ...,
    *,
    pre_dispatch: bool = ...,
    record_module_stack: bool = ...,
    _allow_fake_constant: bool = ...,
    _error_on_data_dependent_ops: bool = ...,
    record_stack_traces: bool = ...,
) -> Callable[..., GraphModule]:
    """
    Given a function f, return a new function which when executed with valid
    arguments to f, returns an FX GraphModule representing the set of operations that
    were executed during the course of execution.

    If record_stack_traces is True, the stack trace will be preserved on node.meta["stack_trace"]
    """

def get_torch_dispatch_modes() -> list[TorchDispatchMode]: ...
def get_innermost_proxy_mode() -> ProxyTorchDispatchMode | None: ...
def get_proxy_mode() -> ProxyTorchDispatchMode | None:
    """
    Current the currently active proxy tracing mode, or None if
    we are not currently tracing.  This includes pre-dispatch proxy
    tracing.
    """

def handle_sym_dispatch(func: Callable[_P, R], args: _P.args, kwargs: _P.kwargs) -> R:
    """
    Call into the currently active proxy tracing mode to do a
    SymInt/SymFloat/SymBool dispatch trace on a function that operates on
    these arguments.
    """

@contextmanager
def disable_proxy_modes_tracing() -> Generator[ProxyTorchDispatchMode]: ...
def maybe_handle_decomp(
    proxy_mode: ProxyTorchDispatchMode, op: OpOverload, args: tuple[object, ...], kwargs: dict[str, object]
) -> object: ...
def get_isolated_graphmodule(
    func: Callable,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    tracing_mode: str = ...,
    decomposition_table: Mapping[OpOverload, Callable] | None = ...,
) -> GraphModule:
    """
    A helper function used to get the GraphModule for the given func.

    It's expected to be used in the ProxyTensor tracing context.
    It detaches the args and kwargs from the current tracer so that the trace of
    the current graph module can be created without any side-effects.
    """
