"""This module provides decorators and utilities for controlling TorchDynamo's behavior during compilation."""

from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, ParamSpec, TypeVar, overload

from torch.utils._contextlib import _DecoratorContextManager

_P = ParamSpec("_P")
_R = TypeVar("_R")
type FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

def run[P, R](fn: Callable[_P, _R] | None = ...) -> Any:
    """Don't do any dynamic compiles, just use prior optimizations"""

def disable(fn=..., recursive=..., *, reason=..., wrapping=...):
    """
    Decorator to disable TorchDynamo

    If recursive=True, Dynamo is completely skipped on the decorated function
    frame as well as the recursively invoked functions.

    If recursive=False, Dynamo skips frames associated with the function code,
    but still process recursively invoked frames.

    If reason is provided, it will be printed when Dynamo attempts to trace the disabled function.
    """

_nonrecursive_disable_wrapper_code = ...

def skip[P, R](fn: Callable[_P, _R] | None = ...) -> Callable[..., Any]:
    """
    Skip frames associated with the function code, but still process recursively
    invoked frames
    """

class set_stance(_DecoratorContextManager):
    """
    Decorator, context manager, function to set the current stance of the compiler.

    Stances documented in corresponding function in torch/compiler/__init__.py
    """

    _dynamo_forbidden = ...
    def __init__(
        self,
        stance: str = ...,
        *,
        skip_guard_eval_unsafe: bool = ...,
        force_backend: str | Callable[..., Any] | None = ...,
    ) -> None: ...
    def __call__(self, fn: F) -> F: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None: ...
    def clone(self) -> set_stance: ...

def assume_constant_result(fn): ...
def allow_in_graph(fn):
    """
    Tells the compiler frontend (Dynamo) to skip symbolic introspection of the function
    and instead directly write it to the graph when encountered.

    See :func:`torch.compiler.allow_in_graph`'s docstring for the full documentation

    WARNING: this API can be a footgun, please read the documentation carefully.
    """

def nonstrict_trace[P, R](traceable_fn: Callable[_P, _R]) -> Callable[_P, _R]: ...
def disallow_in_graph(fn: Callable[..., Any]) -> Any:
    """
    Customize which functions TorchDynamo will exclude in the generated
    graph and force a graph break on.
    ::

        torch._dynamo.disallow_in_graph(torch.sub)


        @torch._dynamo.optimize(...)
        def fn(a):
            x = torch.add(x, 1)
            x = torch.sub(x, 1)
            x = torch.add(x, 1)
            return x


        fn(...)

    Will break the graph on `torch.sub`, and give two graphs each with a
    single `torch.add()` op.
    """

@_disallow_in_graph_helper(throw_if_not_allowed=False)
def graph_break(msg: str = ...) -> None:
    """Force a graph break"""

@_disallow_in_graph_helper(throw_if_not_allowed=False)
def skip_frame(msg: str = ...) -> None:
    """Force a skipped frame"""

def forbid_in_graph(fn: Any) -> Any:
    """
    Customize which functions TorchDynamo will assert are not present while tracing.

    If you want a graph break on this function instead, use disallow_in_graph.
    TODO(voz): We now have allow_in_graph, disallow_in_graph, forbid_in_graph - some more robust
    documentation would not be amiss.
    """

def substitute_in_graph[P, R](
    original_fn: Callable[_P, _R],
    *,
    can_constant_fold_through: bool = ...,
    skip_signature_check: bool = ...,
    is_embedded_type: bool = ...,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """
    Register a polyfill handler for a function, usually a C function from the C extension, to be
    used in place of the original function when inlining the original function in the graph.

    .. note::

        The polyfill handler is only used when inlining the original function. It is not used when
        the original function is called directly. In the eager mode, the decorated function calls
        the performant C function rather than the polyfill handler.

    The polyfill handler is a function that will be called in place of the original function when
    inlining the original function. The polyfill handler should have the same signature and the same
    behavior as the original function.

    Args:
        original_fn (callable): The original function, usually a C function, to register a polyfill
            handler for.
        can_constant_fold_through (bool, optional): Whether the polyfill handler can be constant
            folded through. That is, if the polyfill handler is a pure function and its arguments
            are constant, the result of the polyfill handler can be constant folded during the
            compilation. Defaults to ``False``.
        skip_signature_check (bool, optional): Whether to skip the signature check between the
            original function and the polyfill handler. Defaults to ``False``.

    Returns:
        A decorator that registers the polyfill handler for the original function.

    Example::

        >>> # xdoctest: +SKIP("conflict with the tests: duplicate polyfill handlers")
        >>> import operator
        >>> operator.indexOf([1, 2, 3, 4, 5], 3)
        2
        >>> torch.compile(operator.indexOf, fullgraph=True)([1, 2, 3, 4, 5], 3)
        Traceback (most recent call last):
        ...
        torch._dynamo.exc.Unsupported: ...

        >>> @torch.compiler.substitute_in_graph(operator.indexOf)
        ... def indexOf(a, b, /):
        ...     for i, item in enumerate(a):
        ...         if item is b or item == b:
        ...             return i
        ...     raise ValueError("sequence.index(x): x not in sequence")
        >>>
        >>> torch.compile(operator.indexOf, fullgraph=True)([1, 2, 3, 4, 5], 3)
        2
    """

@dataclass(frozen=True)
class _DimRange:
    """
    This represents an dimension of a tensor and the corresponding
    min and max values it can take.  Don't create this
    class directly; instead, use :func:`mark_dynamic`.
    """

    dim: int
    min: int
    max: int

@forbid_in_graph
def mark_unbacked(
    t: Any, index: int | list[Any] | tuple[Any], strict: bool = ..., specialize_on: list[Any] | None = ...
) -> None:
    """
    Mark a tensor as having an unbacked dim.  This changes the semantics of operations,
    we will always report the size does not equal zero/one, we will turn asserts
    on this index into runtime asserts, and if you try to get the real value we will
    raise an exception.  In other words, we will treat this dimension as if it was
    data dependent (we do not know anything about its value.)

    For historical reasons, by default if an unbacked dim is specialized, we will
    happily specialize it and continue. If you want to error in these cases, pass
    strict=True.
    """

@forbid_in_graph
def mark_dynamic(
    t: Any,
    index: int | list[Any] | tuple[Any],
    *,
    hint_override: int | None = ...,
    min: int | None = ...,
    max: int | None = ...,
    specialize_on: list[Any] | None = ...,
) -> None:
    """
    Mark a tensor as having a dynamic dim and set corresponding min and max range for the dim.

    [Note - on the state of mark_dynamic]

    The behavior of having a dynamic dimension on a tensor is governed by a few factors:

    1) torch._dynamo.config dynamic_shapes True or False.
        a) dynamic_shapes=True - dynamic_shapes must be True for mark_dynamic to work.
        a) dynamic_shapes=False - This config will raise an exception when used in conjunction with
        mark_dynamic. We will eventually support this.

    2) If the dimension is fully constrained - as in, it does not allow more than a single value
    in both eager (torch.compile, torch._dynamo.optimize) mode and export mode (torch._dynamo.export),
    we will raise an error

    3) If the dimension is partially constrained - allowing at least 2 values but not the full unbounded
    range of shapes, in eager we will pass it through, but export will raise an error.

    4) Attempts to trace this function will explicitly raise. As such, all calls to mark_dynamic must be made
    before torch.compile.

    5) If specialize_on is passed in, we will perform a single generic Dynamo trace followed by
    multiple specialized compilations in addition to a single generic compilation. NB: For now we only support
    per dimension specialization, or in other words we do not generate a cross product of specializations.
    At runtime, we will dispatch to a specialized compiled region if the input matches the specialization criteria.

    For example:
        mark_dynamic(..., specialize_on=[
            lambda x: x == 8,
            lambda x: x == 16
        ])

    This approach results in one Dynamo trace and two backend compilations. When the input dimension equals 8 or 16
    at runtime, execution will be directed to the specialized compiled region. Performance measurements indicate
    2-8x speedups depending on the specific specialization and model architecture.
    """

@forbid_in_graph
def maybe_mark_dynamic(t: Any, index: int | list[Any] | tuple[Any]) -> None:
    """
    Mark a tensor as having a dynamic dim, but don't enforce it (i.e., if this
    dimension ends up getting specialized, don't error).
    """

def mark_static(t: Any, index: int | list[Any] | tuple[Any] | None = ...) -> None:
    """
    Mark a tensor as having a static dim or mark a nn module class as static.

    For tensors
    ===========
    This will prevent us from attempting to compile it dynamically
    when dynamic=True; this can improve trace-time performance.

    This has lower precedence than mark_dynamic.

    Unlike mark_dynamic, this can be done inside a graph, in which case it
    induces specialization on the tensor.

    For nn.Module classes
    =====================
    For static nn.Module classes, TorchDynamo assumes that the module instance
    attributes will not be modified after compilation. This will ensure that
    TorchDynamo keeps integer attributes CONSTANT and not symints.

    From TorchDynamo implementation side, the instances of static-marked
    nn.Module class will be converted to UnspecializedBuiltinNNModuleVariable,
    which have the same properties.

    Note that we still have to guard on the attributes, because different
    instances of the nn.Module can have different values of the attributes. The
    key point here is that the attributes are static.
    """

@forbid_in_graph
def mark_static_address(t: Any, guard: bool = ...) -> None:
    """
    Marks an input tensor whose address should be treated as constant across calls to the
    same dynamo-compiled function. This indicates to cudagraphs that an extra allocation
    is not needed for this input. The data_ptr will be guarded if guard=True, and cause a full
    recompile if the data_ptr changes. Note: If this address changes, cudagraphs will re-record
    if guard=False.
    """

class DynamoConfigPatchProxy:
    def __init__(self, config_patch: Any) -> None: ...
    @property
    def changes(self) -> dict[str, Any]: ...

    __call__ = ...
    def __enter__(self) -> None: ...
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None: ...

_allowed_config_patches = ...

def patch_dynamo_config(
    arg1: str | dict[str, Any] | tuple[tuple[str, Any], ...] | None = ..., arg2: Any = ..., **kwargs: Any
) -> DynamoConfigPatchProxy:
    """
    A wrapper around torch._dynamo.config.patch that can be traced by Dynamo to
    temporarily change config values DURING tracing.

    See _allowed_config_patches for the list of allowed config patches.

    Arguments are the same as with torch._dynamo.config.patch.

    Can be used as a decorator or a context manager.

    User code SHOULD NOT MODIFY the return value of this function.

    WARNING: changing Dynamo config during tracing can lead to unpredictable tracing behavior!
        Proceed only as advised!
    """

@overload
def dont_skip_tracing(fn: None = ...) -> DynamoConfigPatchProxy:
    """
    Context manager/decorator to trace into functions intentionally marked by developers to be skipped
    when tracing.

    This decorator will also apply to recursively invoked functions.
    """

@overload
def dont_skip_tracing[P, R](fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Context manager/decorator to trace into functions intentionally marked by developers to be skipped
    when tracing.

    This decorator will also apply to recursively invoked functions.
    """

def dont_skip_tracing(fn: Any | None = ...) -> Any:
    """
    Context manager/decorator to trace into functions intentionally marked by developers to be skipped
    when tracing.

    This decorator will also apply to recursively invoked functions.
    """

class ErrorOnGraphBreakDecoratorContextManager:
    def __init__(self, error_on_graph_break: bool) -> None: ...

    __call__ = ...
    def __enter__(self) -> None: ...
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None: ...

def error_on_graph_break(error_on_graph_break: bool) -> ErrorOnGraphBreakDecoratorContextManager:
    """
    Context manager/decorator to toggle torch.compile's `error_on_graph_break` setting at compile time.

    If `fullgraph` is set, then `error_on_graph_break` does nothing
    (i.e. `fullgraph = True` takes higher precedence). If `fullgraph` is False, then
    `error_on_graph_break` determines whether `torch.compile` throws an error upon
    encountering a graph break, or attempts to continue tracing.

    `error_on_graph_break` can be toggled during compile time with this decorator to allow graph breaks in some
    compiled regions but not others. One key difference from `fullgraph` is that `error_on_graph_break = True`
    does NOT guarantee that a single graph is captured from the compiled function.

    The default value of torch.compile's `error_on_graph_break` setting is False.
    """
