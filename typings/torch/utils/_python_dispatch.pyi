import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeIs, overload

import torch

_is_in_torch_dispatch_mode = ...
_is_in_non_infra_torch_dispatch_mode = ...
_is_in_any_mode_without_ignore_compile_internals = ...

def is_in_torch_dispatch_mode(include_infra_modes=...) -> bool: ...
def is_in_any_mode_without_ignore_compile_internals() -> bool: ...

class TorchDispatchMode:
    """
    A ``TorchDispatchMode`` allows you to override the meaning of all
    ``__torch_dispatch__`` overrideable functions within a dynamic scope,
    without having to actually create a tensor subclass or manually
    monkey-patch functions in the PyTorch API.  Some common situations
    where you should use a mode:

        * You want to override the meaning of factory functions, or other
          functions that do not otherwise take a tensor as an argument
          (these cannot be overridden with tensor subclasses).

        * You want to override the behavior of all functions without needing
          to wrap your inputs in tensor subclasses; e.g., if you are just
          interested in logging intermediate computations.

        * You want to control the order of execution of various tensor
          subclasses explicitly, rather than implicitly via the return of
          ``NotImplemented``.

    Independent subclasses of :class:`TorchDispatchMode` are compositional:
    modes can be pushed onto a stack using ``with MyMode():``.
    When you call functions in the PyTorch API inside your
    ``__torch_dispatch__`` implementation, by default, they will forward on to
    the next mode on the mode stack.  If you want recursively call back into
    your current ``__torch_dispatch__`` implementation, either explicitly
    invoke ``self.__torch_dispatch__(...)``, or use the context manager
    ``__torch_dispatch__(self)`` to make PyTorch
    API self-referential (beware of infinite loops, in this case!)
    """

    supports_higher_order_operators = ...
    def __init__(self, _dispatch_key=...) -> None: ...
    def __torch_dispatch__(self, func, types, args=..., kwargs=...): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
    @classmethod
    def push(cls, *args, **kwargs): ...
    @classmethod
    def is_infra_mode(cls): ...
    @classmethod
    def ignore_compile_internals(cls):
        """
        Ignore operators that are compiled via torch.compile.

        If ``True``, then this TorchDispatchMode ignores operators that
        are optimized by :func:`torch.compile`. Mechanically, this involves
        turning off the TorchDispatchMode throughout the whole compilation process,
        and turning it back on for the runtime of the compiled artifact(s).
        For example,

        @torch.compile
        def f(x):
            return x.sin().cos()

        with LoggingMode():
            f(x)

        The above example will not log anything if
        ``LoggingMode.ignore_compile_internals()`` is True.
        torch.compile will fuse sin() and cos() into a single operation
        and this TorchDispatchMode will not be passed sin and cos.

        If ``False`` (default), :func:`torch.compile` will respect
        the eager semantics of passing this TorchDispatchMode all
        operators that would have run during eager execution.
        The way this will usually happen is that :func:`torch.compile`
        will just fallback to eager-mode PyTorch.
        """

class BaseTorchDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=..., kwargs=...): ...

class TensorWithFlatten(Protocol):
    def __tensor_flatten__(self) -> tuple[Sequence[str], object]: ...
    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: int, flatten_spec: int, outer_size: int, outer_stride: int
    ) -> torch.Tensor: ...

    shape: torch._C.Size
    @overload
    def stride(self, dim: None = ...) -> tuple[int, ...]:
        """Helper for @overload to raise when called."""
        ...
    @overload
    def stride(self, dim: int) -> int:
        """Helper for @overload to raise when called."""
        ...
    @overload
    def size(self, dim: None = ...) -> tuple[int, ...]:
        """Helper for @overload to raise when called."""
        ...
    @overload
    def size(self, dim: int) -> int:
        """Helper for @overload to raise when called."""
        ...
    def storage_offset(self) -> int: ...
    def dim(self) -> int: ...
    @overload
    def to(
        self,
        dtype: torch.types._dtype,
        non_blocking: bool = ...,
        copy: bool = ...,
        *,
        memory_format: torch.memory_format | None = ...,
    ) -> torch.Tensor:
        """Helper for @overload to raise when called."""
        ...
    @overload
    def to(
        self,
        device: torch._prims_common.DeviceLikeType | None = ...,
        dtype: torch.types._dtype | None = ...,
        non_blocking: bool = ...,
        copy: bool = ...,
        *,
        memory_format: torch.memory_format | None = ...,
    ) -> torch.Tensor:
        """Helper for @overload to raise when called."""
        ...
    @overload
    def to(
        self,
        other: torch.Tensor,
        non_blocking: bool = ...,
        copy: bool = ...,
        *,
        memory_format: torch.memory_format | None = ...,
    ) -> torch.Tensor:
        """Helper for @overload to raise when called."""
        ...

def is_traceable_wrapper_subclass(t: object) -> TypeIs[TensorWithFlatten]:
    """
    Returns whether or not a tensor subclass that implements __torch_dispatch__
    is 'traceable' with torch.compile.
    In order for a tensor subclass to support TorchDispatchMode-style tracing in PT2,
    It must implement two magic methods: __tensor_flatten__ and __tensor_unflatten__.
    It is also expected to obey some restrictions around traceability and aliasing:
        * The subclass's __torch_dispatch__() implementation should desugar into pytorch
            dispatcher operations that can be traced into a graph.
        * The subclass should use return_and_correct_aliasing(). This is needed today to make
            sure that torch.compile does the right thing in a few cases around input mutation
            and output aliasing.

    Expected magic method signatures:
        attrs, ctx = t.__tensor_flatten__()
            attrs: list of attribute name strings for inner tensors
            ctx: dict containing any other subclass-specific metadata needed for unflattening

        t = MySubClass.__tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride)
            inner_tensors: dict mapping attribute name -> tensor for each inner tensor
            ctx: dict with subclass metadata in the form that __tensor_flatten__() produces
            outer_size: expected (possibly symbolic) size that the returned subclass
                instance should have. Note that this arg is useful for certain subclasses
                that require the shape info to be constructed. In most cases, this arg can be
                safely ignored.
            outer_stride: expected (possibly symbolic) stride that the returned subclass
                instance should have. Note that this arg is useful for certain subclasses
                that require the stride info to be constructed. In most cases, this arg can be
                safely ignored.
    """

def is_traceable_wrapper_subclass_type(t: type) -> TypeIs[type[TensorWithFlatten]]:
    """Same as above, but takes a type argument instead of an instance."""

def transform_subclass(t, callback, outer_size=..., outer_stride=...):
    """
    Given a traceable, wrapper tensor subclass ``t`` that implements
    ``__torch_dispatch__`` and holds some inner tensors,
    and a callback of type ``Callable[[str, torch.Tensor], torch.Tensor]``,
    `transform_subclass` will construct a fresh instance of the wrapper tensor subclass.
    It will do so by grabbing each inner tensor attribute from the wrapper,
    passing them into ``callback`` to get a transformed tensor,
    and putting each transformed tensor into the fresh tensor subclass instance.

    Note: this function will not handle ensuring that the fresh subclass
    gets the same (autograd, and aliasing) metadata as the original tensor.
    This is generally handled in other subsystems like AOTAutograd.
    """

@dataclass
class AliasInfo:
    """AliasInfo(alias_set: set[str], is_write: bool, name: Optional[str])"""

    alias_set: set[str]
    is_write: bool
    name: str | None

@dataclass
class SchemaInfo:
    """SchemaInfo(args: list[torch.utils._python_dispatch.AliasInfo], outs: list[torch.utils._python_dispatch.AliasInfo], int_tags: list[int])"""

    args: list[AliasInfo]
    outs: list[AliasInfo]
    int_tags: list[int]

@functools.cache
def get_alias_info(func) -> SchemaInfo: ...

_TORCH_TAG_INPLACE_VIEW_INT = ...

def return_and_correct_aliasing(func, args, kwargs, out):
    """
    This function should be used by wrapper tensor ``__torch_dispatch__`` subclasses
    that would like to work with torch.compile. It ensures that the subclass
    properly implements the aliasing behavior of every op,
    which is needed for correctness in AOTAutograd.
    This function will handle:

        * When we see a view op, we will alias the storages of any
          input and output tensor subclasses

        * When we see an inplace or out= op, we will directly
          return the corresponding input tensor, instead of returning
          a (potentially) fresh output tensor.
    """
