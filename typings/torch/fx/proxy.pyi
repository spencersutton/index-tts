from collections import OrderedDict
from collections.abc import Callable, Iterator
from typing import Any

from ._compatibility import compatibility
from .graph import Graph
from .node import Argument, Node, Target

__all__ = [
    "Attribute",
    "GraphAppendingTracer",
    "MetaProxy",
    "ParameterProxy",
    "Proxy",
    "Scope",
    "ScopeContextManager",
    "TraceError",
    "TracerBase",
]
log = ...

@compatibility(is_backward_compatible=False)
class Scope:
    """
    Scope object that records the module path and the module type
    of a module. Scope is used to track the information of the module
    that contains a Node in a Graph of GraphModule. For example::

        class Sub(torch.nn.Module):
            def forward(self, x):
                # This will be a call_method Node in GraphModule,
                # scope for this would be (module_path="sub", module_type=Sub)
                return x.transpose(1, 2)


        class M(torch.nn.Module):
            def __init__(self) -> None:
                self.sub = Sub()

            def forward(self, x):
                # This will be a call_method Node as well,
                # scope for this would be (module_path="", None)
                x = x.transpose(1, 2)
                x = self.sub(x)
                return x


    .. warning::
        This API is experimental and is *NOT* backward-compatible.
    """
    def __init__(self, module_path: str, module_type: Any) -> None: ...

@compatibility(is_backward_compatible=False)
class ScopeContextManager:
    """
    A context manager to track the Scope of Node during symbolic tracing.
    When entering a forward function of a Module, we'll update the scope information of
    the current module, and when we exit, we'll restore the previous scope information.

    .. warning::
        This API is experimental and is *NOT* backward-compatible.
    """
    def __init__(self, scope: Scope, current_scope: Scope) -> None: ...
    def __enter__(self): ...
    def __exit__(self, *args): ...

_COPY_META_FIELDS = ...

@compatibility(is_backward_compatible=True)
class TracerBase:
    """
    .. note::
        Backwards-compatibility for this API is guaranteed.
    """

    graph: Graph
    record_stack_traces: bool = ...
    _record_forward_stack_traces_only: bool = ...
    check_mutable_operations: bool = ...
    trace_asserts: bool = ...
    proxy_buffer_attributes: bool = ...
    traced_func_name: str = ...
    scope: Scope
    module_stack: OrderedDict[str, tuple[str, Any]]
    node_name_to_scope: dict[str, tuple[str, type]]
    @compatibility(is_backward_compatible=True)
    def create_node(
        self,
        kind: str,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        name: str | None = ...,
        type_expr: Any | None = ...,
    ) -> Node:
        """
        Inserts a graph node given target, args, kwargs, and name.

        This method can be overridden to do extra checking, validation, or
        modification of values used in node creation. For example, one might
        want to disallow in-place operations from being recorded.

        .. note::
            Backwards-compatibility for this API is guaranteed.
        """
    @compatibility(is_backward_compatible=True)
    def proxy(self, node: Node) -> Proxy:
        """
        .. note::
            Backwards-compatibility for this API is guaranteed.
        """
    @compatibility(is_backward_compatible=True)
    def create_proxy(
        self,
        kind: str,
        target: Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        name: str | None = ...,
        type_expr: Any | None = ...,
        proxy_factory_fn: Callable[[Node], Proxy] = ...,
    ):
        """
        Create a Node from the given arguments, then return the Node
        wrapped in a Proxy object.

        If kind = 'placeholder', then we're creating a Node that
        represents the parameter of a function. If we need to encode
        a default parameter, we use the ``args`` tuple. ``args`` is
        otherwise empty for ``placeholder`` Nodes.

        .. note::
            Backwards-compatibility for this API is guaranteed.
        """
    @compatibility(is_backward_compatible=True)
    def create_arg(self, a: Any) -> Argument:
        """
        A method that lowers the objects seen as arguments during symbolic evaluation
        into Argument types that can be stored in IR.

        Can be override to support more trace-specific types.

        .. note::
            Backwards-compatibility for this API is guaranteed.
        """
    @compatibility(is_backward_compatible=True)
    def to_bool(self, obj: Proxy) -> bool:
        """
        Called when a proxy object is being converted to a boolean, such as
        when used in control flow.  Normally we don't know what to do because
        we don't know the value of the proxy, but a custom tracer can attach more
        information to the graph node using create_node and can choose to return a value.

        .. note::
            Backwards-compatibility for this API is guaranteed.
        """
    @compatibility(is_backward_compatible=True)
    def iter(self, obj: Proxy) -> Iterator:
        """
        Called when a proxy object is being iterated over, such as
        when used in control flow.  Normally we don't know what to do because
        we don't know the value of the proxy, but a custom tracer can attach more
        information to the graph node using create_node and can choose to return an iterator.

        .. note::
            Backwards-compatibility for this API is guaranteed.
        """
    @compatibility(is_backward_compatible=True)
    def keys(self, obj: Proxy) -> Any:
        """
        Called when a proxy object is has the keys() method called.
        This is what happens when ** is called on a proxy. This should return an
        iterator it ** is suppose to work in your custom tracer.

        .. note::
            Backwards-compatibility for this API is guaranteed.
        """

@compatibility(is_backward_compatible=True)
class GraphAppendingTracer(TracerBase):
    """
    .. note::
        Backwards-compatibility for this API is guaranteed.
    """
    def __init__(self, graph: Graph) -> None: ...

@compatibility(is_backward_compatible=False)
def assert_fn(x):
    """
    .. warning::
        This API is experimental and is *NOT* backward-compatible.
    """

@compatibility(is_backward_compatible=True)
class TraceError(ValueError):
    """
    .. note::
        Backwards-compatibility for this API is guaranteed.
    """

@compatibility(is_backward_compatible=True)
class Proxy:
    """
    ``Proxy`` objects are ``Node`` wrappers that flow through the
    program during symbolic tracing and record all the operations
    (``torch`` function calls, method calls, operators) that they touch
    into the growing FX Graph.

    If you're doing graph transforms, you can wrap your own ``Proxy``
    method around a raw ``Node`` so that you can use the overloaded
    operators to add additional things to a ``Graph``.

    ``Proxy`` objects cannot be iterated. In other words, the symbolic
    tracer will throw an error if a ``Proxy`` is used in a loop or as
    an ``*args``/``**kwargs`` function argument.

    There are two main ways around this:
    1. Factor out the untraceable logic into a top-level function and
    use ``fx.wrap`` on it.
    2. If the control flow is static (i.e. the loop trip count is
    based on some hyperparameter), the code can be kept in its original
    position and refactored into something like::

        for i in range(self.some_hyperparameter):
            indexed_item = proxied_value[i]

    For a more detailed description into the Proxy internals, check out
    the "Proxy" section in `torch/fx/README.md`

    .. note::
        Backwards-compatibility for this API is guaranteed.
    """
    @compatibility(is_backward_compatible=True)
    def __init__(self, node: Node, tracer: TracerBase | None = ...) -> None:
        """
        .. note::
            Backwards-compatibility for this API is guaranteed.
        """
    def __getattr__(self, k) -> Attribute: ...
    def __getstate__(self) -> dict: ...
    def __deepcopy__(self, memo) -> dict: ...
    def __setstate__(self, d): ...
    def __call__(self, *args, **kwargs) -> Proxy: ...
    def __iter__(self) -> Iterator[Proxy]: ...
    def __abs__(self): ...
    def __bool__(self) -> bool: ...
    @compatibility(is_backward_compatible=True)
    def keys(self):
        """
        .. note::
            Backwards-compatibility for this API is guaranteed.
        """
    def __len__(self) -> int: ...
    @classmethod
    def __torch_function__(cls, orig_method, types, args=..., kwargs=...): ...

@compatibility(is_backward_compatible=False)
class MetaProxy(Proxy):
    """
    A Proxy subclass that propagates metadata (meta['val']) during graph tracing.

    .. warning::
        This API is experimental and is *NOT* backward-compatible.
    """
    def __init__(self, node: Node, tracer: TracerBase | None = ..., fake_mode=...) -> None: ...
    @classmethod
    def __torch_function__(cls, orig_method, types, args=..., kwargs=...): ...

@compatibility(is_backward_compatible=True)
class Attribute(Proxy):
    """
    .. note::
        Backwards-compatibility for this API is guaranteed.
    """
    @compatibility(is_backward_compatible=True)
    def __init__(self, root: Proxy, attr: str) -> None:
        """
        .. note::
            Backwards-compatibility for this API is guaranteed.
        """
    @property
    def node(self): ...
    def __call__(self, *args, **kwargs): ...

@compatibility(is_backward_compatible=False)
class ParameterProxy(Proxy):
    """
    A special proxy which lets "shape", "size", "dim", and a few other
    attribute accesses pass through to the underlying  module parameter object,
    so that conditional tests on these attributes will not throw exception during tracing

    .. warning::
        This API is experimental and is *NOT* backward-compatible.
    """
    def __init__(self, tracer: TracerBase, node: Node, name, param) -> None: ...
    @property
    def shape(self): ...
    def size(self): ...
    def dim(self): ...
    @property
    def ndim(self): ...
    def numel(self): ...
    def nelement(self): ...

_create_arg_bypass = ...
