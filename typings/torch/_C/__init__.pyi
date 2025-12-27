"""An Enum that contains tags that can be assigned to an operator registered in C++."""

from builtins import bool as _bool, complex as _complex, float as _float, int as _int, str as _str
from collections.abc import Callable, Iterable, Iterator, Sequence
from enum import Enum
from pathlib import Path
from types import EllipsisType
from typing import IO, Any, Literal, ParamSpec, Protocol, Self, SupportsIndex, TypeVar, overload, runtime_checkable

import numpy as np
import torch
from torch import SymInt, Tensor
from torch._prims_common import DeviceLikeType
from torch.autograd.graph import Node as _Node
from torch.cuda import _POOL_HANDLE
from torch.fx.node import Node as FxNode
from torch.storage import TypedStorage, UntypedStorage
from torch.types import (
    IntLikeType,
    Number,
    Storage,
    _device,
    _dispatchkey,
    _dtype,
    _layout,
    _qscheme,
    _size,
    _symsize,
)

K = TypeVar("K")
T = TypeVar("T")
S = TypeVar("S", bound=torch.Tensor)
P = ParamSpec("P")
R = TypeVar("R", covariant=True)
T_co = TypeVar("T_co", covariant=True)

@runtime_checkable
class _NestedSequence(Protocol[T_co]):
    def __len__(self, /) -> _int: ...
    def __getitem__(self, index: _int, /) -> T_co | _NestedSequence[T_co]: ...
    def __contains__(self, x: object, /) -> _bool: ...
    def __iter__(self, /) -> Iterator[T_co | _NestedSequence[T_co]]: ...
    def __reversed__(self, /) -> Iterator[T_co | _NestedSequence[T_co]]: ...
    def count(self, value: Any, /) -> _int: ...
    def index(self, value: Any, /) -> _int: ...

class device:
    type: str
    index: _int
    def __get__(self, instance, owner=...) -> device: ...
    @overload
    def __init__(self, device: DeviceLikeType) -> None: ...
    @overload
    def __init__(self, type: str, index: _int) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
    def __reduce__(self) -> tuple[Any, ...]: ...

class Stream:
    """
    Stream(device, *, priority) -> Stream

    An in-order queue of executing the respective tasks asynchronously in first in first out (FIFO) order.
    It can control or synchronize the execution of other Stream or block the current host thread to ensure
    the correct task sequencing. It supports with statement as a context manager to ensure the operators
    within the with block are running on the corresponding stream.

    See in-depth description of the CUDA behavior at :ref:`cuda-semantics` for details
    on the exact semantic that applies to all devices.

    Arguments:
        device (:class:`torch.device`, optional): the desired device for the Stream.
            If not given, the current :ref:`accelerator<accelerators>` type will be used.
        priority (int, optional): priority of the stream, should be 0 or negative, where negative
            numbers indicate higher priority. By default, streams have priority 0.

    Returns:
        Stream: An torch.Stream object.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> with torch.Stream(device='cuda') as s_cuda:
        >>>     a = torch.randn(10, 5, device='cuda')
        >>>     b = torch.randn(5, 10, device='cuda')
        >>>     c = torch.mm(a, b)
    """

    stream_id: _int
    device_index: _int
    device_type: _int
    device: _device
    @overload
    def __new__(cls, device: DeviceLikeType | None = ..., *, priority: _int = ...) -> Self: ...
    @overload
    def __new__(cls, stream_id: _int, device_index: _int, device_type: _int, *, priority: _int = ...) -> Self: ...
    def query(self) -> _bool:
        """
        Stream.query() -> bool

        Check if all the work submitted has been completed.

        Returns:
            bool: A boolean indicating if all kernels in this stream are completed.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> s_cuda = torch.Stream(device='cuda')
            >>> s_cuda.query()
            True
        """
    def synchronize(self) -> None:
        """
        Stream.synchronize() -> None

        Wait for all the kernels in this stream to complete.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> s_cuda = torch.Stream(device='cuda')
            >>> s_cuda.synchronize()
        """
    def wait_event(self, event: Event) -> None:
        """
        Stream.wait_event(event) -> None

        Make all future work submitted to the stream wait for an event.

        Arguments:
            event (:class:`torch.Event`): an event to wait for.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> s1_cuda = torch.Stream(device='cuda')
            >>> s2_cuda = torch.Stream(device='cuda')
            >>> e_cuda = s1_cuda.record_event()
            >>> s2_cuda.wait_event(e_cuda)
        """
    def wait_stream(self, other: Stream) -> None:
        """
        Stream.wait_stream(stream) -> None

        Synchronize with another stream. All future work submitted to this stream will wait until all kernels
        already submitted to the given stream are completed.

        Arguments:
            stream (:class:`torch.Stream`): a stream to synchronize.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> s1_cuda = torch.Stream(device='cuda')
            >>> s2_cuda = torch.Stream(device='cuda')
            >>> s2_cuda.wait_stream(s1_cuda)
        """
    def record_event(self, event: Event | None = ...) -> Event:
        """
        Stream.record_event(event) -> Event

        Record an event. En-queuing it into the Stream to allow further synchronization from the current point in the FIFO queue.

        Arguments:
            event (:class:`torch.Event`, optional): event to record. If not given, a new one will be allocated.

        Returns:
            Event: Recorded event.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> s_cuda = torch.Stream(device='cuda')
            >>> e_cuda = s_cuda.record_event()
        """
    def __hash__(self) -> _int:
        """Return hash(self)."""
    def __eq__(self, other: object) -> _bool:
        """Return self==value."""
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

class Event:
    """
    Event(device=None, *, enable_timing=False, blocking=False, interprocess=False)

    Query and record Stream status to identify or control dependencies across Stream and measure timing.

    Arguments:
        device (:class:`torch.device`, optional): the desired device for the Event.
            If not given, the current :ref:`accelerator<accelerators>` type will be used.
        enable_timing (bool, optional): indicates if the event should measure time (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes (default: ``False``)

    .. warning::

        Both blocking and interprocess are not supported right now and are noops.

    Returns:
        Event: An torch.Event object.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> event = torch.Event()
        >>> e_cuda = torch.Event(device='cuda')
    """

    device: _device
    event_id: _int
    def __new__(
        cls,
        device: DeviceLikeType | None = ...,
        *,
        enable_timing: _bool = ...,
        blocking: _bool = ...,
        interprocess: _bool = ...,
    ) -> Self: ...
    @classmethod
    def from_ipc_handle(cls, device: _device, ipc_handle: bytes) -> Event: ...
    def record(self, stream: Stream | None = ...) -> None:
        """
        Event.record(stream=None) -> None

        Record the event in a given stream. The stream's device must match the event's device.
        This function is equivalent to ``stream.record_event(self)``.

        Arguments:
            stream (:class:`torch.Stream`, optional): A stream to be recorded.
                If not given, the current stream will be used.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> e_cuda = torch.Event(device='cuda')
            >>> e_cuda.record()
        """
    def wait(self, stream: Stream | None = ...) -> None:
        """
        Event.wait(stream=None) -> None

        Make all future work submitted to the given stream wait for this event.

        Arguments:
            stream (:class:`torch.Stream`, optional): A stream to synchronize.
                If not given, the current stream will be used.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> s1_cuda = torch.Stream(device='cuda')
            >>> s2_cuda = torch.Stream(device='cuda')
            >>> e_cuda = s1_cuda.record()
            >>> e_cuda.wait(s2)
        """
    def query(self) -> _bool:
        """
        Event.query() -> bool

        Check if the stream where this event was recorded already moved past the point where the event was recorded.
        Always returns ``True`` if the Event was not recorded.

        Returns:
            bool: A boolean indicating if all work currently captured by event has completed.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> s_cuda = torch.Stream(device='cuda')
            >>> e_cuda = s_cuda.record_event()
            >>> e_cuda.query()
            True
        """
    def elapsed_time(self, other: Event) -> _float:
        """
        Event.elapsed_time(end_event) -> float

        Returns the elapsed time in milliseconds between when this event and the :attr:`end_event` are
        each recorded via :func:`torch.Stream.record_event`.

        Arguments:
            end_event (:class:`torch.Event`): The ending event has been recorded.

        Returns:
            float: Time between starting and ending event in milliseconds.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> s_cuda = torch.Stream(device='cuda')
            >>> e1_cuda = s_cuda.record_event()
            >>> e2_cuda = s_cuda.record_event()
            >>> ms = e1_cuda.elapsed_time(e2_cuda)
        """
    def synchronize(self) -> None:
        """
        Event.synchronize() -> None

        Wait for the event to complete. This prevents the CPU thread from proceeding until the event completes.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> s_cuda = torch.Stream(device='cuda')
            >>> e_cuda = s_cuda.record_event()
            >>> e_cuda.synchronize()
        """
    def ipc_handle(self) -> bytes: ...

class Size(tuple[_int, ...]):
    @overload
    def __getitem__(self: Size, key: SupportsIndex, /) -> _int:
        """Return self[key]."""
    @overload
    def __getitem__(self: Size, key: slice, /) -> Size:
        """Return self[key]."""
    def __add__(self, other: tuple[_int, ...], /) -> Size:
        """Return self+value."""
    def __radd__(self: Size, other: tuple[_int, ...], /) -> Size:
        """Return value+self."""
    def __mul__(self, other: SupportsIndex, /) -> Size:
        """Return self*value."""
    def __rmul__(self, other: SupportsIndex, /) -> Size:
        """Return value*self."""
    def numel(self: Size, /) -> _int:
        """
        numel() -> int

        Returns the number of elements a :class:`torch.Tensor` with the given size would contain.

        More formally, for a tensor ``x = tensor.ones(10, 10)`` with size ``s = torch.Size([10, 10])``,
        ``x.numel() == x.size().numel() == s.numel() == 100`` holds true.

        Example::

            >>> x=torch.ones(10, 10)
            >>> s=x.size()
            >>> s
            torch.Size([10, 10])
            >>> s.numel()
            100
            >>> x.numel() == s.numel()
            True


        .. warning::

            This function does not return the number of dimensions described by :class:`torch.Size`, but instead the number
            of elements a :class:`torch.Tensor` with that size would contain.
        """

class dtype:
    is_floating_point: _bool
    is_complex: _bool
    is_signed: _bool
    itemsize: _int
    def to_real(self) -> dtype: ...
    def to_complex(self) -> dtype: ...

class iinfo:
    bits: _int
    min: _int
    max: _int
    dtype: str
    def __init__(self, dtype: _dtype) -> None: ...

class finfo:
    bits: _int
    min: _float
    max: _float
    eps: _float
    tiny: _float
    smallest_normal: _float
    resolution: _float
    dtype: str
    @overload
    def __init__(self, dtype: _dtype) -> None: ...
    @overload
    def __init__(self) -> None: ...

float32: dtype = ...
float: dtype = ...
float64: dtype = ...
double: dtype = ...
float16: dtype = ...
bfloat16: dtype = ...
float8_e4m3fn: dtype = ...
float8_e4m3fnuz: dtype = ...
float8_e5m2: dtype = ...
float8_e5m2fnuz: dtype = ...
float8_e8m0fnu: dtype = ...
float4_e2m1fn_x2: dtype = ...
half: dtype = ...
uint8: dtype = ...
uint16: dtype = ...
uint32: dtype = ...
uint64: dtype = ...
int8: dtype = ...
int16: dtype = ...
short: dtype = ...
int32: dtype = ...
int: dtype = ...
int64: dtype = ...
long: dtype = ...
complex32: dtype = ...
complex64: dtype = ...
chalf: dtype = ...
cfloat: dtype = ...
complex128: dtype = ...
cdouble: dtype = ...
quint8: dtype = ...
qint8: dtype = ...
qint32: dtype = ...
bool: dtype = ...
quint4x2: dtype = ...
quint2x4: dtype = ...
bits1x8: dtype = ...
bits2x4: dtype = ...
bits4x2: dtype = ...
bits8: dtype = ...
bits16: dtype = ...

class layout: ...

def DisableTorchFunction(): ...
def DisableTorchFunctionSubclass(): ...

strided: layout = ...
sparse_coo: layout = ...
sparse_csr: layout = ...
sparse_csc: layout = ...
sparse_bsr: layout = ...
sparse_bsc: layout = ...
_mkldnn: layout = ...
jagged: layout = ...

class memory_format: ...

contiguous_format: memory_format = ...
channels_last: memory_format = ...
channels_last_3d: memory_format = ...
preserve_format: memory_format = ...

class qscheme: ...

per_tensor_affine: qscheme = ...
per_channel_affine: qscheme = ...
per_tensor_symmetric: qscheme = ...
per_channel_symmetric: qscheme = ...
per_channel_affine_float_qparams: qscheme = ...

class _FunctionBase:
    saved_tensors: tuple[Tensor]
    _raw_saved_tensors: tuple[Any]
    next_functions: tuple[tuple[Any, _int], ...]
    needs_input_grad: tuple[_bool]
    metadata: dict
    _materialize_non_diff_grads: _bool

class _LegacyVariableBase(Tensor):
    def __init__(
        self,
        data: Tensor | None = ...,
        requires_grad: _bool | None = ...,
        volatile: _bool | None = ...,
        _grad_fn: _FunctionBase | None = ...,
    ) -> None: ...

class IODescriptor: ...
class JITException(Exception): ...

class Future[T]:
    def __init__(self, devices: list[device]) -> None:
        """__init__(self: torch._C.Future, arg0: collections.abc.Sequence[torch.device]) -> None"""
    def done(self) -> _bool:
        """done(self: torch._C.Future) -> bool"""
    def value(self) -> T:
        """value(self: torch._C.Future) -> object"""
    def wait(self) -> T:
        """wait(self: torch._C.Future) -> object"""
    def add_done_callback(self, callback: Callable) -> None:
        """add_done_callback(self: torch._C.Future, arg0: collections.abc.Callable) -> None"""
    def then(self, callback: Callable) -> Future[T]:
        """then(self: torch._C.Future, arg0: collections.abc.Callable) -> torch._C.Future"""
    def set_result(self, result: T) -> None:
        """set_result(self: torch._C.Future, arg0: object) -> None"""

class _Await:
    def __init__(self) -> None: ...
    def fn(self) -> Callable:
        """fn(self: torch._C._Await) -> collections.abc.Callable"""
    def args(self) -> tuple[Any, ...]:
        """args(self: torch._C._Await) -> tuple"""
    def is_nowait(self) -> _bool:
        """is_nowait(self: torch._C._Await) -> bool"""

class _MobileOptimizerType:
    """
    Members:

    CONV_BN_FUSION

    INSERT_FOLD_PREPACK_OPS

    REMOVE_DROPOUT

    FUSE_ADD_RELU

    HOIST_CONV_PACKED_PARAMS

    VULKAN_AUTOMATIC_GPU_TRANSFER
    """

CONV_BN_FUSION: _MobileOptimizerType
INSERT_FOLD_PREPACK_OPS: _MobileOptimizerType
REMOVE_DROPOUT: _MobileOptimizerType
FUSE_ADD_RELU: _MobileOptimizerType
HOIST_CONV_PACKED_PARAMS: _MobileOptimizerType
VULKAN_AUTOMATIC_GPU_TRANSFER: _MobileOptimizerType

def fork(*args: Any, **kwargs: Any) -> Future:
    """fork(*args, **kwargs) -> torch._C.Future"""

def wait(fut: Future) -> Any:
    """wait(arg0: torch._C.Future) -> object"""

def unify_type_list(types: list[JitType]) -> JitType:
    """unify_type_list(arg0: collections.abc.Sequence[c10::Type]) -> c10::Type"""

type ResolutionCallback = Callable[[str], Callable[..., Any]]

def parse_type_comment(comment: str) -> Decl:
    """parse_type_comment(arg0: str) -> torch._C._jit_tree_views.Decl"""

def merge_type_from_type_comment(decl: Decl, type_annotation_decl: Decl, is_method: _bool) -> Decl:
    """merge_type_from_type_comment(arg0: torch._C._jit_tree_views.Decl, arg1: torch._C._jit_tree_views.Decl, arg2: bool) -> torch._C._jit_tree_views.Decl"""

def parse_ir(input: str, parse_tensor_constants: _bool = ...) -> Graph:
    """parse_ir(input: str, parse_tensor_constants: bool = False) -> torch::jit::Graph"""

def parse_schema(schema: str) -> FunctionSchema:
    """parse_schema(schema: str, allow_typevars: bool = True) -> c10::FunctionSchema"""

def get_device(input: Tensor) -> _int: ...
def import_ir_module(
    cu: CompilationUnit, filename: str | Path, map_location: DeviceLikeType | None, extra_files: dict[str, Any]
) -> ScriptModule:
    """import_ir_module(arg0: torch._C.CompilationUnit, arg1: str, arg2: object, arg3: dict, arg4: bool) -> torch._C.ScriptModule"""

def import_ir_module_from_buffer(
    cu: CompilationUnit, buffer: IO[bytes], map_location: DeviceLikeType | None, extra_files: dict[str, Any]
) -> ScriptModule:
    """import_ir_module_from_buffer(arg0: torch._C.CompilationUnit, arg1: str, arg2: object, arg3: dict, arg4: bool) -> torch._C.ScriptModule"""

class GraphExecutorState: ...
class AliasDb: ...

class _InsertPoint:
    def __enter__(self) -> None: ...
    def __exit__(self, *exc_info: object) -> None: ...

class Use:
    @property
    def user(self) -> Node: ...
    @property
    def offset(self) -> _int: ...
    def isAfter(self, other: Use) -> _bool:
        """isAfter(self: torch._C.Use, arg0: torch._C.Use) -> bool"""

class Value:
    def type(self) -> JitType:
        """
        type(*args, **kwargs)
        Overloaded function.

        1. type(self: torch._C.Value) -> c10::Type

        2. type(self: torch._C.Value) -> c10::Type
        """
    def setType(self, t: JitType) -> Value:
        """setType(self: torch._C.Value, arg0: c10::Type) -> torch._C.Value"""
    def setTypeAs(self, other: Value) -> Value:
        """setTypeAs(self: torch._C.Value, arg0: torch._C.Value) -> torch._C.Value"""
    def inferTypeFrom(self, t: Tensor) -> None:
        """
        inferTypeFrom(*args, **kwargs)
        Overloaded function.

        1. inferTypeFrom(self: torch._C.Value, arg0: torch.Tensor) -> None

        2. inferTypeFrom(self: torch._C.Value, arg0: c10::ivalue::Object) -> None
        """
    def debugName(self) -> str:
        """debugName(self: torch._C.Value) -> str"""
    def setDebugName(self, name: str) -> None:
        """setDebugName(self: torch._C.Value, arg0: str) -> torch._C.Value"""
    def unique(self) -> _int:
        """unique(self: torch._C.Value) -> int"""
    def offset(self) -> _int:
        """offset(self: torch._C.Value) -> int"""
    def node(self) -> Node:
        """node(self: torch._C.Value) -> torch::jit::Node"""
    def uses(self) -> list[Use]:
        """uses(self: torch._C.Value) -> list[torch::jit::Use]"""
    def replaceAllUsesWith(self, val: Value) -> None:
        """replaceAllUsesWith(self: torch._C.Value, arg0: torch._C.Value) -> None"""
    def replaceAllUsesAfterNodeWith(self, node: Node, val: Value) -> None:
        """replaceAllUsesAfterNodeWith(self: torch._C.Value, arg0: torch::jit::Node, arg1: torch._C.Value) -> None"""
    def requires_grad(self) -> _bool:
        """requires_grad(self: torch._C.Value) -> bool"""
    def requiresGrad(self) -> _bool:
        """requiresGrad(self: torch._C.Value) -> bool | None"""
    def copyMetadata(self, other: Value) -> Value:
        """copyMetadata(self: torch._C.Value, arg0: torch._C.Value) -> torch._C.Value"""
    def isCompleteTensor(self) -> _bool:
        """isCompleteTensor(self: torch._C.Value) -> bool"""
    def toIValue(self) -> IValue:
        """toIValue(self: torch._C.Value) -> IValue | None"""

class Block:
    def inputs(self) -> Iterator[Value]:
        """inputs(self: torch._C.Block) -> collections.abc.Iterator[torch._C.Value]"""
    def outputs(self) -> Iterator[Value]:
        """outputs(self: torch._C.Block) -> collections.abc.Iterator[torch._C.Value]"""
    def nodes(self) -> Iterator[Node]:
        """nodes(self: torch._C.Block) -> collections.abc.Iterator[torch::jit::Node]"""
    def paramNode(self) -> Node:
        """paramNode(self: torch._C.Block) -> torch::jit::Node"""
    def returnNode(self) -> Node:
        """returnNode(self: torch._C.Block) -> torch::jit::Node"""
    def owningNode(self) -> Node:
        """owningNode(self: torch._C.Block) -> torch::jit::Node"""
    def registerOutput(self, n: Value) -> _int:
        """registerOutput(self: torch._C.Block, arg0: torch._C.Value) -> int"""
    def addNode(self, name: str, inputs: Sequence[Value]) -> Node:
        """addNode(self: torch._C.Block, arg0: str, arg1: collections.abc.Sequence[torch._C.Value]) -> torch::jit::Node"""

class Node:
    def __getitem__(self, key: str) -> Any: ...
    def schema(self) -> str:
        """schema(self: torch._C.Node) -> str"""
    def input(self) -> Value:
        """input(self: torch._C.Node) -> torch._C.Value"""
    def inputs(self) -> Iterator[Value]:
        """inputs(self: torch._C.Node) -> collections.abc.Iterator[torch._C.Value]"""
    def inputsAt(self, idx: _int) -> Value:
        """inputsAt(self: torch._C.Node, arg0: typing.SupportsInt) -> torch._C.Value"""
    def inputsSize(self) -> _int:
        """inputsSize(self: torch._C.Node) -> int"""
    def output(self) -> Value:
        """output(self: torch._C.Node) -> torch._C.Value"""
    def outputs(self) -> Iterator[Value]:
        """outputs(self: torch._C.Node) -> collections.abc.Iterator[torch._C.Value]"""
    def outputsAt(self, idx: _int) -> Value:
        """outputsAt(self: torch._C.Node, arg0: typing.SupportsInt) -> torch._C.Value"""
    def outputsSize(self) -> _int:
        """outputsSize(self: torch._C.Node) -> int"""
    def hasMultipleOutputs(self) -> _bool:
        """hasMultipleOutputs(self: torch._C.Node) -> bool"""
    def blocks(self) -> list[Block]:
        """blocks(self: torch._C.Node) -> collections.abc.Iterator[torch._C.Block]"""
    def addBlock(self) -> Block:
        """addBlock(self: torch._C.Node) -> torch._C.Block"""
    def mustBeNone(self) -> _bool:
        """mustBeNone(self: torch._C.Node) -> bool"""
    def matches(self, pattern: str) -> _bool:
        """matches(self: torch._C.Node, arg0: str) -> bool"""
    def kind(self) -> str:
        """kind(self: torch._C.Node) -> Symbol"""
    def kindOf(self, name: str) -> str:
        """kindOf(self: torch._C.Node, arg0: str) -> AttributeKind"""
    def addInput(self, name: str) -> Value:
        """addInput(self: torch._C.Node, arg0: torch._C.Value) -> torch._C.Value"""
    def replaceInput(self, i: _int, newValue: Value) -> Value:
        """replaceInput(self: torch._C.Node, arg0: typing.SupportsInt, arg1: torch._C.Value) -> torch._C.Value"""
    def replaceInputWith(self, from_: Value, to: Value) -> None:
        """replaceInputWith(self: torch._C.Node, arg0: torch._C.Value, arg1: torch._C.Value) -> None"""
    def replaceAllUsesWith(self, n: Node) -> None:
        """replaceAllUsesWith(self: torch._C.Node, arg0: torch._C.Node) -> None"""
    def insertBefore(self, n: Node) -> Node:
        """insertBefore(self: torch._C.Node, arg0: torch._C.Node) -> torch._C.Node"""
    def insertAfter(self, n: Node) -> Node:
        """insertAfter(self: torch._C.Node, arg0: torch._C.Node) -> torch._C.Node"""
    def isBefore(self, n: Node) -> _bool:
        """isBefore(self: torch._C.Node, arg0: torch._C.Node) -> bool"""
    def isAfter(self, n: Node) -> _bool:
        """isAfter(self: torch._C.Node, arg0: torch._C.Node) -> bool"""
    def moveBefore(self, n: Node) -> None:
        """moveBefore(self: torch._C.Node, arg0: torch._C.Node) -> None"""
    def moveAfter(self, n: Node) -> None:
        """moveAfter(self: torch._C.Node, arg0: torch._C.Node) -> None"""
    def removeInput(self, i: _int) -> None:
        """removeInput(self: torch._C.Node, arg0: typing.SupportsInt) -> None"""
    def removeAllInputs(self, i: _int) -> None:
        """removeAllInputs(self: torch._C.Node) -> None"""
    def hasUses(self) -> _bool:
        """hasUses(self: torch._C.Node) -> bool"""
    def eraseOutput(self, i: _int) -> None:
        """eraseOutput(self: torch._C.Node, arg0: typing.SupportsInt) -> None"""
    def addOutput(self) -> Value:
        """addOutput(self: torch._C.Node) -> torch._C.Value"""
    def scopeName(self) -> str:
        """scopeName(self: torch._C.Node) -> str"""
    def isNondeterministic(self) -> _bool:
        """isNondeterministic(self: torch._C.Node) -> bool"""
    def copyAttributes(self, rhs: Node) -> Node:
        """copyAttributes(self: torch._C.Node, arg0: torch._C.Node) -> torch._C.Node"""
    def copyMetadata(self, rhs: Node) -> Node:
        """copyMetadata(self: torch._C.Node, arg0: torch._C.Node) -> torch._C.Node"""
    def hasAttributes(self) -> _bool:
        """hasAttributes(self: torch._C.Node) -> bool"""
    def hasAttribute(self, name: str) -> _bool:
        """hasAttribute(self: torch._C.Node, arg0: str) -> bool"""
    def removeAttribute(self, attr: str) -> Node:
        """removeAttribute(self: torch._C.Node, arg0: str) -> torch._C.Node"""
    def namedInput(self, name: str) -> Value:
        """namedInput(self: torch._C.Node, arg0: str) -> torch._C.Value"""
    def sourceRange(self) -> SourceRange:
        """sourceRange(self: torch._C.Node) -> str"""
    def owningBlock(self) -> Block:
        """owningBlock(self: torch._C.Node) -> torch._C.Block"""
    def findNode(self, kind: str, recurse: _bool = ...) -> Node:
        """
        findNode(self: torch._C.Node, kind: str, recurse: bool = True) -> torch._C.Node

        Find Node
        """
    def findAllNodes(self, kind: str, recurse: _bool = ...) -> list[Node]:
        """
        findAllNodes(self: torch._C.Node, kind: str, recurse: bool = True) -> list[torch._C.Node]

        Find all nodes
        """
    def getModuleHierarchy(self) -> str:
        """getModuleHierarchy(self: torch._C.Node) -> str"""
    def prev(self) -> Node:
        """prev(self: torch._C.Node) -> torch._C.Node"""
    def destroy(self) -> None:
        """destroy(self: torch._C.Node) -> None"""
    def attributeNames(self) -> list[str]:
        """attributeNames(self: torch._C.Node) -> list[str]"""
    def f(self, name: str) -> _float:
        """f(self: torch._C.Node, arg0: str) -> float"""
    def f_(self, name: str, val: _float) -> Node:
        """f_(self: torch._C.Node, arg0: str, arg1: typing.SupportsFloat) -> torch._C.Node"""
    def fs(self, name: str) -> list[_float]:
        """fs(self: torch._C.Node, arg0: str) -> list[float]"""
    def fs_(self, name: str, val: list[_float]) -> Node:
        """fs_(self: torch._C.Node, arg0: str, arg1: collections.abc.Sequence[typing.SupportsFloat]) -> torch._C.Node"""
    def c(self, name: str) -> complex:
        """c(self: torch._C.Node, arg0: str) -> complex"""
    def c_(self, name: str, val: complex) -> Node:
        """c_(self: torch._C.Node, arg0: str, arg1: complex) -> torch._C.Node"""
    def s(self, name: str) -> str:
        """s(self: torch._C.Node, arg0: str) -> str"""
    def s_(self, name: str, val: str) -> Node:
        """s_(self: torch._C.Node, arg0: str, arg1: str) -> torch._C.Node"""
    def ss(self, name: str) -> list[str]:
        """ss(self: torch._C.Node, arg0: str) -> list[str]"""
    def ss_(self, name: str, val: list[str]) -> Node:
        """ss_(self: torch._C.Node, arg0: str, arg1: collections.abc.Sequence[str]) -> torch._C.Node"""
    def i(self, name: str) -> _int:
        """i(self: torch._C.Node, arg0: str) -> int"""
    def i_(self, name: str, val: _int) -> Node:
        """i_(self: torch._C.Node, arg0: str, arg1: typing.SupportsInt) -> torch._C.Node"""
    def g(self, name: str) -> Graph:
        """g(self: torch._C.Node, arg0: str) -> torch._C.Graph"""
    def g_(self, name: str, val: Graph) -> Node:
        """g_(self: torch._C.Node, arg0: str, arg1: torch._C.Graph) -> torch._C.Node"""
    def gs(self, name: str) -> list[Graph]:
        """gs(self: torch._C.Node, arg0: str) -> list[torch._C.Graph]"""
    def gs_(self, name: str, val: list[Graph]) -> Node:
        """gs_(self: torch._C.Node, arg0: str, arg1: collections.abc.Sequence[torch._C.Graph]) -> torch._C.Node"""
    def ival(self, name: str) -> IValue:
        """ival(self: torch._C.Node, arg0: str) -> IValue"""
    def ival_(self, name: str, val: IValue) -> Node:
        """ival_(self: torch._C.Node, arg0: str, arg1: IValue) -> torch._C.Node"""
    def t(self, name: str) -> Tensor:
        """t(self: torch._C.Node, arg0: str) -> torch.Tensor"""
    def t_(self, name: str, val: Tensor) -> Node:
        """t_(self: torch._C.Node, arg0: str, arg1: torch.Tensor) -> torch._C.Node"""
    def ts(self, name: str) -> list[Tensor]:
        """ts(self: torch._C.Node, arg0: str) -> list[torch.Tensor]"""
    def ts_(self, name: str, val: list[Tensor]) -> Node:
        """ts_(self: torch._C.Node, arg0: str, arg1: collections.abc.Sequence[torch.Tensor]) -> torch._C.Node"""
    def ty(self, name: str) -> JitType:
        """ty(self: torch._C.Node, arg0: str) -> c10::Type"""
    def ty_(self, name: str, val: JitType) -> Node:
        """ty_(self: torch._C.Node, arg0: str, arg1: c10::Type) -> torch._C.Node"""
    def tys(self, name: str) -> list[JitType]:
        """tys(self: torch._C.Node, arg0: str) -> list[c10::Type]"""
    def tys_(self, name: str, val: list[JitType]) -> Node:
        """tys_(self: torch._C.Node, arg0: str, arg1: collections.abc.Sequence[c10::Type]) -> torch._C.Node"""

class Graph:
    def inputs(self) -> Iterator[Value]:
        """inputs(self: torch._C.Graph) -> collections.abc.Iterator[torch::jit::Value]"""
    def outputs(self) -> Iterator[Value]:
        """outputs(self: torch._C.Graph) -> collections.abc.Iterator[torch::jit::Value]"""
    def nodes(self) -> Iterator[Node]:
        """nodes(self: torch._C.Graph) -> collections.abc.Iterator[torch::jit::Node]"""
    def param_node(self) -> Node:
        """param_node(self: torch._C.Graph) -> torch::jit::Node"""
    def return_node(self) -> Node:
        """return_node(self: torch._C.Graph) -> torch::jit::Node"""
    def addInput(self, name: str = ...) -> Value:
        """
        addInput(self: torch._C.Graph, name: str = '') -> torch::jit::Value

        Add input to graph with optional name seed
        """
    def eraseInput(self, i: _int) -> None:
        """eraseInput(self: torch._C.Graph, arg0: typing.SupportsInt) -> None"""
    def registerOutput(self, n: Value) -> _int:
        """registerOutput(self: torch._C.Graph, arg0: torch::jit::Value) -> int"""
    def eraseOutput(self, i: _int) -> None:
        """eraseOutput(self: torch._C.Graph, arg0: typing.SupportsInt) -> None"""
    def create(self, name: str, args, num_outputs: _int) -> Node:
        """
        create(*args, **kwargs)
        Overloaded function.

        1. create(self: torch._C.Graph, arg0: str) -> torch::jit::Node

        2. create(self: torch._C.Graph, arg0: str, arg1: typing.SupportsInt) -> torch::jit::Node

        3. create(self: torch._C.Graph, arg0: str, arg1: collections.abc.Sequence[torch::jit::Value]) -> torch::jit::Node

        4. create(self: torch._C.Graph, arg0: str, arg1: collections.abc.Sequence[torch::jit::Value], arg2: typing.SupportsInt) -> torch::jit::Node
        """
    def appendNode(self, n: Node) -> Node:
        """appendNode(self: torch._C.Graph, arg0: torch::jit::Node) -> torch::jit::Node"""
    def prependNode(self, n: Node) -> Node:
        """prependNode(self: torch._C.Graph, arg0: torch::jit::Node) -> torch::jit::Node"""
    def insertNode(self, n: Node) -> Node:
        """insertNode(self: torch._C.Graph, arg0: torch::jit::Node) -> torch::jit::Node"""
    def block(self) -> Block:
        """block(self: torch._C.Graph) -> torch::jit::Block"""
    def lint(self) -> None:
        """lint(self: torch._C.Graph) -> None"""
    def alias_db(self) -> AliasDb:
        """alias_db(self: torch._C.Graph, isFrozen: bool = False, descend_function_calls: bool = False) -> torch._C.AliasDb"""
    def setInsertPoint(self, n: Block | Node) -> None:
        """
        setInsertPoint(*args, **kwargs)
        Overloaded function.

        1. setInsertPoint(self: torch._C.Graph, arg0: torch::jit::Node) -> None

        2. setInsertPoint(self: torch._C.Graph, arg0: torch::jit::Block) -> None
        """
    def insert_point_guard(self, n: Block | Node) -> _InsertPoint:
        """
        insert_point_guard(*args, **kwargs)
        Overloaded function.

        1. insert_point_guard(self: torch._C.Graph, arg0: torch::jit::Node) -> object

        2. insert_point_guard(self: torch._C.Graph, arg0: torch::jit::Block) -> object
        """
    def insertPoint(self) -> Node:
        """insertPoint(self: torch._C.Graph) -> torch::jit::Node"""
    def insertGraph(self, callee: Graph, inputs: list[Value]) -> list[Value]:
        """
        insertGraph(*args, **kwargs)
        Overloaded function.

        1. insertGraph(self: torch._C.Graph, arg0: torch._C.Graph, arg1: collections.abc.Sequence[torch::jit::Value]) -> list[torch::jit::Value]

        2. insertGraph(self: torch._C.Graph, arg0: torch._C.Graph, arg1: collections.abc.Sequence[torch::jit::Value], arg2: collections.abc.Mapping[torch::jit::Value, torch::jit::Value]) -> list[torch::jit::Value]
        """
    def makeMultiOutputIntoTuple(self) -> None:
        """makeMultiOutputIntoTuple(self: torch._C.Graph) -> None"""
    def copy(self) -> Graph:
        """copy(self: torch._C.Graph) -> torch._C.Graph"""

class AliasInfo:
    is_write: _bool
    before_set: set[str]
    after_set: set[str]
    def __init__(self, is_write: _bool, before_set: set[str], after_set: set[str]) -> None: ...

class Argument:
    name: str
    type: JitType
    default_value: Any | None
    def has_default_value(self) -> _bool:
        """has_default_value(self: torch._C.Argument) -> bool"""

    kwarg_only: _bool
    is_out: _bool
    alias_info: AliasInfo | None
    is_write: _bool
    real_type: JitType
    def __init__(
        self,
        name: str,
        type: JitType,
        N: _int | None,
        defualt_value: Any | None,
        kwarg_only: _bool,
        alias_info: AliasInfo | None,
    ) -> None:
        """__init__(self: torch._C.Argument, arg0: str, arg1: c10::Type, arg2: typing.SupportsInt | None, arg3: IValue | None, arg4: bool, arg5: c10::AliasInfo | None) -> None"""

class FunctionSchema:
    arguments: list[Argument]
    returns: list[Argument]
    name: str
    overload_name: str
    is_mutable: _bool
    def __init__(
        self,
        name: str,
        overload_name: str,
        arguments: list[Argument],
        returns: list[Argument],
        is_vararg: _bool,
        is_varret: _bool,
    ) -> None:
        """__init__(self: torch._C.FunctionSchema, arg0: str, arg1: str, arg2: collections.abc.Sequence[c10::Argument], arg3: collections.abc.Sequence[c10::Argument], arg4: bool, arg5: bool) -> None"""

class _UpgraderEntry:
    bumped_at_version: _int
    upgrader_name: str
    old_schema: str
    def __init__(self, bumped_at_version: _int, upgrader_name: str, old_schema: str) -> None:
        """__init__(self: torch._C._UpgraderEntry, arg0: typing.SupportsInt, arg1: str, arg2: str) -> None"""

class _UpgraderRange:
    min_version: _int
    max_version: _int

class ScriptModuleSerializer:
    def __init__(self, export_writer: PyTorchFileWriter) -> None:
        """__init__(self: torch._C.ScriptModuleSerializer, arg0: torch._C.PyTorchFileWriter) -> None"""
    def serialize(self, model: ScriptModule, script_module_id: _int) -> None:
        """serialize(self: torch._C.ScriptModuleSerializer, arg0: torch::jit::Module, arg1: typing.SupportsInt) -> None"""
    def write_files(self) -> None:
        """write_files(self: torch._C.ScriptModuleSerializer, code_dir: str = '.data/ts_code/code/') -> None"""
    def storage_context(self) -> SerializationStorageContext:
        """storage_context(self: torch._C.ScriptModuleSerializer) -> torch::jit::SerializationStorageContext"""

class SerializationStorageContext:
    def __init__(self) -> None: ...
    def has_storage(self, storage: Storage) -> _bool:
        """has_storage(self: torch._C.SerializationStorageContext, arg0: torch.StorageBase) -> bool"""
    def get_or_add_storage(self, storage: Storage) -> _int:
        """get_or_add_storage(self: torch._C.SerializationStorageContext, arg0: torch.StorageBase) -> int"""

class DeserializationStorageContext:
    def __init__(self) -> None:
        """__init__(self: torch._C.DeserializationStorageContext) -> None"""
    def get_storage(self, name: str, dtype: _dtype) -> Tensor:
        """get_storage(self: torch._C.DeserializationStorageContext, arg0: str, arg1: object) -> torch.Tensor"""
    def has_storage(self, name: str) -> _bool:
        """has_storage(self: torch._C.DeserializationStorageContext, arg0: str) -> bool"""
    def add_storage(self, name: str, tensor: Tensor) -> _int:
        """add_storage(self: torch._C.DeserializationStorageContext, arg0: str, arg1: torch.Tensor) -> None"""

class ConcreteModuleTypeBuilder:
    def __init__(self, obj: Any) -> None:
        """__init__(self: torch._C.ConcreteModuleTypeBuilder, arg0: object) -> None"""
    def set_module_dict(self):
        """set_module_dict(self: torch._C.ConcreteModuleTypeBuilder) -> None"""
    def set_module_list(self):
        """set_module_list(self: torch._C.ConcreteModuleTypeBuilder) -> None"""
    def set_parameter_list(self):
        """set_parameter_list(self: torch._C.ConcreteModuleTypeBuilder) -> None"""
    def set_parameter_dict(self):
        """set_parameter_dict(self: torch._C.ConcreteModuleTypeBuilder) -> None"""
    def add_attribute(self, name: str, ty: JitType, is_param: _bool, is_buffer: _bool):
        """add_attribute(self: torch._C.ConcreteModuleTypeBuilder, arg0: str, arg1: torch._C.Type, arg2: bool, arg3: bool) -> None"""
    def add_module(self, name: str, meta: ConcreteModuleType):
        """add_module(self: torch._C.ConcreteModuleTypeBuilder, arg0: str, arg1: torch::jit::ConcreteModuleType) -> None"""
    def add_constant(self, name: str, value: Any):
        """add_constant(self: torch._C.ConcreteModuleTypeBuilder, arg0: str, arg1: object) -> None"""
    def add_overload(self, method_name: str, overloaded_method_names: list[str]):
        """add_overload(self: torch._C.ConcreteModuleTypeBuilder, arg0: str, arg1: collections.abc.Sequence[str]) -> None"""
    def add_builtin_function(self, name: str, symbol_name: str):
        """add_builtin_function(self: torch._C.ConcreteModuleTypeBuilder, arg0: str, arg1: str) -> None"""
    def add_failed_attribute(self, name: str, failure_reason: str):
        """add_failed_attribute(self: torch._C.ConcreteModuleTypeBuilder, arg0: str, arg1: str) -> None"""
    def add_function_attribute(self, name: str, ty: JitType, func: Callable[..., Any]):
        """add_function_attribute(self: torch._C.ConcreteModuleTypeBuilder, arg0: str, arg1: torch._C.Type, arg2: object) -> None"""
    def add_ignored_attribute(self, name: str):
        """add_ignored_attribute(self: torch._C.ConcreteModuleTypeBuilder, arg0: str) -> None"""
    def add_ignored_attributes(self, names: list[str]):
        """add_ignored_attributes(self: torch._C.ConcreteModuleTypeBuilder, arg0: collections.abc.Sequence[str]) -> None"""
    def add_forward_hook(self, hook: Callable[..., Any]):
        """add_forward_hook(self: torch._C.ConcreteModuleTypeBuilder, arg0: object) -> None"""
    def add_forward_pre_hook(self, pre_hook: Callable[..., Any]):
        """add_forward_pre_hook(self: torch._C.ConcreteModuleTypeBuilder, arg0: object) -> None"""

class ConcreteModuleType:
    def get_constants(self) -> dict[str, Any]:
        """get_constants(self: torch._C.ConcreteModuleType) -> dict[str, object]"""
    def equals(self, other: ConcreteModuleType) -> _bool:
        """
        equals(*args, **kwargs)
        Overloaded function.

        1. equals(self: torch._C.ConcreteModuleType, arg0: torch._C.ConcreteModuleType) -> bool

        2. equals(self: torch._C.ConcreteModuleType, arg0: torch._C.ConcreteModuleTypeBuilder) -> bool
        """
    @staticmethod
    def from_jit_type(ty: JitType) -> ConcreteModuleType:
        """from_jit_type(arg0: torch._C.Type) -> torch._C.ConcreteModuleType"""

class CallStack:
    def __init__(self, name: str, range: SourceRange) -> None:
        """__init__(self: torch._C.CallStack, arg0: str, arg1: torch._C._jit_tree_views.SourceRange) -> None"""

class ErrorReport:
    def __init__(self, range: SourceRange) -> None:
        """__init__(self: torch._C.ErrorReport, arg0: torch._C._jit_tree_views.SourceRange) -> None"""
    def what(self) -> str:
        """what(self: torch._C.ErrorReport) -> str"""
    @staticmethod
    def call_stack() -> str:
        """call_stack() -> str"""

class CompilationUnit:
    def __init__(self, lang: str = ..., _frames_up: _int = ...) -> None:
        """__init__(self: torch._C.CompilationUnit, lang: str = '', _frames_up: typing.SupportsInt = 0) -> None"""
    def find_function(self, name: str) -> ScriptFunction:
        """find_function(self: torch._C.CompilationUnit, arg0: str) -> torch::jit::StrongFunctionPtr | None"""
    def __getattr__(self, name: str) -> ScriptFunction:
        """__getattr__(self: torch._C.CompilationUnit, arg0: str) -> torch::jit::StrongFunctionPtr"""
    def define(self, script: str, rcb: ResolutionCallback = ..., _frames_up: _int = ...):
        """define(self: torch._C.CompilationUnit, src: str, rcb: collections.abc.Callable[[str], object] = None, _frames_up: typing.SupportsInt = 0) -> None"""
    def get_interface(self, name: str) -> InterfaceType:
        """get_interface(self: torch._C.CompilationUnit, arg0: str) -> torch._C.InterfaceType"""
    def get_functions(self) -> list[ScriptFunction]:
        """get_functions(self: torch._C.CompilationUnit) -> list[torch::jit::StrongFunctionPtr]"""
    def create_function(self, name: str, graph: Graph, shouldMangle: _bool = ...) -> ScriptFunction:
        """create_function(self: torch._C.CompilationUnit, qualified_name: str, graph: torch._C.Graph, should_mangle: bool = False) -> torch::jit::StrongFunctionPtr"""
    def get_class(self, name: str) -> ClassType:
        """get_class(self: torch._C.CompilationUnit, arg0: str) -> torch._C.ClassType"""

class ScriptObject:
    def setattr(self, name: str, value: Any):
        """setattr(self: torch._C.ScriptObject, arg0: str, arg1: object) -> None"""

class ScriptModule(ScriptObject): ...

class LiteScriptModule:
    def __call__(self, *input):
        """Call self as a function."""
    def find_method(self, method_name: str):
        """find_method(self: torch._C.LiteScriptModule, method_name: str) -> bool"""
    def forward(self, *input) -> list[str]:
        """forward(self: torch._C.LiteScriptModule, input_tuple: tuple) -> IValue"""
    def run_method(self, method_name: str, *input):
        """run_method(self: torch._C.LiteScriptModule, method_name: str, input_tuple: tuple) -> IValue"""

class ScriptFunction[P, R]:
    """
    Functionally equivalent to a :class:`ScriptModule`, but represents a single
    function and does not have any attributes or Parameters.
    """
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """__call__(*args, **kwargs) -> object"""
    def save(self, filename: str, _extra_files: dict[str, bytes]) -> None:
        """save(self: torch._C.ScriptFunction, filename: str, _extra_files: collections.abc.Mapping[str, str] = {}) -> None"""
    def save_to_buffer(self, _extra_files: dict[str, bytes]) -> bytes:
        """save_to_buffer(self: torch._C.ScriptFunction, _extra_files: collections.abc.Mapping[str, str] = {}) -> bytes"""
    @property
    def graph(self) -> Graph: ...
    def inlined_graph(self) -> Graph: ...
    def schema(self) -> FunctionSchema: ...
    def code(self) -> str: ...
    def name(self) -> str: ...
    @property
    def qualified_name(self) -> str: ...

class ScriptMethod[P, R]:
    graph: Graph
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """__call__(*args, **kwargs) -> object"""
    @property
    def owner(self) -> ScriptModule: ...
    @property
    def name(self) -> str: ...
    @property
    def schema(self) -> FunctionSchema: ...

class ScriptDict[K, T]:
    def __init__(self, dict: dict[K, T]) -> None:
        """__init__(self: torch._C.ScriptDict, arg0: dict) -> None"""
    def __len__(self) -> _int:
        """__len__(self: torch._C.ScriptDict) -> object"""
    def __contains__(self, key: K) -> _bool:
        """__contains__(self: torch._C.ScriptDict, arg0: object) -> object"""
    def __getitem__(self, key: K) -> T:
        """__getitem__(self: torch._C.ScriptDict, arg0: object) -> object"""
    def __setitem__(self, key: K, value: T) -> None:
        """__setitem__(self: torch._C.ScriptDict, arg0: object, arg1: object) -> None"""
    def __delitem__(self, key: K) -> None:
        """__delitem__(self: torch._C.ScriptDict, arg0: object) -> None"""
    def __iter__(self) -> Iterator[K]:
        """__iter__(self: torch._C.ScriptDict) -> torch._C.ScriptDictKeyIterator"""
    def items(self) -> Iterator[tuple[K, T]]:
        """items(self: torch._C.ScriptDict) -> torch._C.ScriptDictIterator"""
    def keys(self) -> Iterator[K]:
        """keys(self: torch._C.ScriptDict) -> torch._C.ScriptDictKeyIterator"""

class ScriptList[T]:
    def __init__(self, list: list[T]) -> None:
        """__init__(self: torch._C.ScriptList, arg0: list) -> None"""
    def __len__(self) -> _int:
        """__len__(self: torch._C.ScriptList) -> object"""
    def __contains__(self, item: T) -> _bool:
        """__contains__(self: torch._C.ScriptList, arg0: object) -> object"""
    @overload
    def __getitem__(self, idx: _int) -> T:
        """
        __getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: torch._C.ScriptList, arg0: typing.SupportsInt) -> object

        2. __getitem__(self: torch._C.ScriptList, arg0: slice) -> torch._C.ScriptList
        """
    @overload
    def __getitem__(self, idx: slice) -> ScriptList[T]:
        """
        __getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: torch._C.ScriptList, arg0: typing.SupportsInt) -> object

        2. __getitem__(self: torch._C.ScriptList, arg0: slice) -> torch._C.ScriptList
        """
    @overload
    def __setitem__(self, idx: _int, value: T) -> None:
        """
        __setitem__(*args, **kwargs)
        Overloaded function.

        1. __setitem__(self: torch._C.ScriptList, arg0: typing.SupportsInt, arg1: object) -> None

        2. __setitem__(self: torch._C.ScriptList, arg0: slice, arg1: list) -> None
        """
    @overload
    def __setitem__(self, idx: slice, value: list[T]) -> None:
        """
        __setitem__(*args, **kwargs)
        Overloaded function.

        1. __setitem__(self: torch._C.ScriptList, arg0: typing.SupportsInt, arg1: object) -> None

        2. __setitem__(self: torch._C.ScriptList, arg0: slice, arg1: list) -> None
        """
    def __delitem__(self, idx: _int) -> None:
        """__delitem__(self: torch._C.ScriptList, arg0: typing.SupportsInt) -> None"""
    def __iter__(self) -> Iterator[T]:
        """__iter__(self: torch._C.ScriptList) -> torch._C.ScriptListIterator"""
    def count(self, value: T) -> _int:
        """count(self: torch._C.ScriptList, arg0: object) -> int"""
    def remove(self, value: T) -> None:
        """remove(self: torch._C.ScriptList, arg0: object) -> None"""
    def append(self, value: T) -> None:
        """append(self: torch._C.ScriptList, arg0: object) -> None"""
    def clear(self) -> None:
        """clear(self: torch._C.ScriptList) -> None"""
    @overload
    def extend(self, values: list[T]) -> None:
        """
        extend(*args, **kwargs)
        Overloaded function.

        1. extend(self: torch._C.ScriptList, arg0: list) -> None

        2. extend(self: torch._C.ScriptList, arg0: collections.abc.Iterable) -> None
        """
    @overload
    def extend(self, values: Iterable[T]) -> None:
        """
        extend(*args, **kwargs)
        Overloaded function.

        1. extend(self: torch._C.ScriptList, arg0: list) -> None

        2. extend(self: torch._C.ScriptList, arg0: collections.abc.Iterable) -> None
        """
    @overload
    def pop(self) -> T:
        """
        pop(*args, **kwargs)
        Overloaded function.

        1. pop(self: torch._C.ScriptList) -> object

        2. pop(self: torch._C.ScriptList, arg0: typing.SupportsInt) -> object
        """
    @overload
    def pop(self, idx: _int) -> T:
        """
        pop(*args, **kwargs)
        Overloaded function.

        1. pop(self: torch._C.ScriptList) -> object

        2. pop(self: torch._C.ScriptList, arg0: typing.SupportsInt) -> object
        """

class ModuleDict:
    def __init__(self, mod: ScriptModule) -> None:
        """__init__(self: torch._C.ModuleDict, arg0: torch._C.ScriptModule) -> None"""
    def items(self) -> list[tuple[str, Any]]:
        """items(self: torch._C.ModuleDict) -> list[tuple[str, object]]"""

class ParameterDict:
    def __init__(self, mod: ScriptModule) -> None:
        """__init__(self: torch._C.ParameterDict, arg0: torch._C.ScriptModule) -> None"""

class BufferDict:
    def __init__(self, mod: ScriptModule) -> None:
        """__init__(self: torch._C.BufferDict, arg0: torch._C.ScriptModule) -> None"""

class Module: ...

def get_num_thread() -> _int: ...
def set_num_threads(nthreads: _int) -> None:
    """
    set_num_threads(int)

    Sets the number of threads used for intraop parallelism on CPU.

    .. warning::
        To ensure that the correct number of threads is used, set_num_threads
        must be called before running eager, JIT or autograd code.
    """

def get_num_interop_threads() -> _int:
    """
    get_num_interop_threads() -> int

    Returns the number of threads used for inter-op parallelism on CPU
    (e.g. in JIT interpreter)
    """

def set_num_interop_threads(nthreads: _int) -> None:
    """
    set_num_interop_threads(int)

    Sets the number of threads used for interop parallelism
    (e.g. in JIT interpreter) on CPU.

    .. warning::
        Can only be called once and before any inter-op parallel work
        is started (e.g. JIT execution).
    """

def set_flush_denormal(arg: _bool) -> _bool:
    """
    set_flush_denormal(mode) -> bool

    Disables denormal floating numbers on CPU.

    Returns ``True`` if your system supports flushing denormal numbers and it
    successfully configures flush denormal mode.  :meth:`~torch.set_flush_denormal`
    is supported on x86 architectures supporting SSE3 and AArch64 architecture.

    Args:
        mode (bool): Controls whether to enable flush denormal mode or not

    Example::

        >>> torch.set_flush_denormal(True)
        True
        >>> torch.tensor([1e-323], dtype=torch.float64)
        tensor([ 0.], dtype=torch.float64)
        >>> torch.set_flush_denormal(False)
        True
        >>> torch.tensor([1e-323], dtype=torch.float64)
        tensor(9.88131e-324 *
               [ 1.0000], dtype=torch.float64)
    """

def get_default_dtype() -> _dtype:
    """
    get_default_dtype() -> torch.dtype

    Get the current default floating point :class:`torch.dtype`.

    Example::

        >>> torch.get_default_dtype()  # initial default for floating point is torch.float32
        torch.float32
        >>> torch.set_default_dtype(torch.float64)
        >>> torch.get_default_dtype()  # default is now changed to torch.float64
        torch.float64
    """

class _LinalgBackend:
    """
    Members:

    Default

    Cusolver

    Magma
    """

    Default: _LinalgBackend
    Cusolver: _LinalgBackend
    Magma: _LinalgBackend

class BatchNormBackend(Enum): ...

class _BlasBackend:
    """
    Members:

    Default

    Cublas

    Cublaslt

    Ck
    """

    Default: _BlasBackend
    Cublas: _BlasBackend
    Cublaslt: _BlasBackend
    Ck: _BlasBackend

class _ROCmFABackend:
    """
    Members:

    Default

    AOTriton

    Ck
    """

    Default: _ROCmFABackend
    AOTriton: _ROCmFABackend
    Ck: _ROCmFABackend

class ConvBackend(Enum): ...

class Tag(Enum):
    """
    Members:

    core

    cudagraph_unsafe

    data_dependent_output

    dynamic_output_shape

    flexible_layout

    generated

    inplace_view

    maybe_aliasing_or_mutating

    needs_contiguous_strides

    needs_exact_strides

    needs_fixed_stride_order

    nondeterministic_bitwise

    nondeterministic_seeded

    pointwise

    pt2_compliant_tag

    view_copy
    """

    core = ...
    cudagraph_unsafe = ...
    data_dependent_output = ...
    dynamic_output_shape = ...
    flexible_layout = ...
    generated = ...
    inplace_view = ...
    maybe_aliasing_or_mutating = ...
    needs_contiguous_strides = ...
    needs_exact_strides = ...
    needs_fixed_stride_order = ...
    nondeterministic_bitwise = ...
    nondeterministic_seeded = ...
    pointwise = ...
    pt2_compliant_tag = ...
    view_copy = ...

has_openmp: _bool
has_mkl: _bool
_has_kleidiai: _bool
_has_mps: _bool
has_lapack: _bool
_has_cuda: _bool
_has_magma: _bool
_has_xpu: _bool
_has_mkldnn: _bool
_has_cudnn: _bool
_has_cusparselt: _bool
has_spectral: _bool
_GLIBCXX_USE_CXX11_ABI: _bool
default_generator: Generator

def is_grad_enabled() -> _bool:
    """
    is_grad_enabled() -> (bool)

    Returns True if grad mode is currently enabled.
    """

def is_inference_mode_enabled() -> _bool:
    """
    is_inference_mode_enabled() -> (bool)

    Returns True if inference mode is currently enabled.
    """

@overload
def set_autocast_enabled(device_type: str, enabled: _bool) -> None: ...
@overload
def set_autocast_enabled(enabled: _bool) -> None: ...
@overload
def is_autocast_enabled(device_type: str) -> _bool: ...
@overload
def is_autocast_enabled() -> _bool: ...
def set_autocast_dtype(device_type: str, dtype: _dtype) -> None: ...
def get_autocast_dtype(device_type: str) -> _dtype: ...
def clear_autocast_cache() -> None: ...
def set_autocast_cpu_enabled(enabled: _bool) -> None: ...
def is_autocast_cpu_enabled() -> _bool: ...
def set_autocast_cpu_dtype(dtype: _dtype) -> None: ...
def set_autocast_gpu_dtype(dtype: _dtype) -> None: ...
def get_autocast_cpu_dtype() -> _dtype: ...
def get_autocast_gpu_dtype() -> _dtype: ...
def autocast_increment_nesting() -> _int: ...
def autocast_decrement_nesting() -> _int: ...
def is_autocast_cache_enabled() -> _bool: ...
def set_autocast_cache_enabled(enabled: _bool) -> None: ...
def set_anomaly_enabled(enabled: _bool, check_nan: _bool = ...) -> None: ...
def is_anomaly_enabled() -> _bool: ...
def is_anomaly_check_nan_enabled() -> _bool: ...

class _DisableTorchDispatch:
    def __init__(self) -> None:
        """__init__(self: torch._C._DisableTorchDispatch) -> None"""
    def __enter__(self):
        """__enter__(self: torch._C._DisableTorchDispatch) -> None"""
    def __exit__(self, *exc_info: object) -> None:
        """__exit__(self: torch._C._DisableTorchDispatch, arg0: object, arg1: object, arg2: object) -> None"""

class _EnableTorchFunction:
    def __init__(self) -> None:
        """__init__(self: torch._C._EnableTorchFunction) -> None"""
    def __enter__(self):
        """__enter__(self: torch._C._EnableTorchFunction) -> None"""
    def __exit__(self, *exc_info: object) -> None:
        """__exit__(self: torch._C._EnableTorchFunction, arg0: object, arg1: object, arg2: object) -> None"""

class _EnablePythonDispatcher:
    def __init__(self) -> None:
        """__init__(self: torch._C._EnablePythonDispatcher) -> None"""
    def __enter__(self):
        """__enter__(self: torch._C._EnablePythonDispatcher) -> None"""
    def __exit__(self, *exc_info: object) -> None:
        """__exit__(self: torch._C._EnablePythonDispatcher, arg0: object, arg1: object, arg2: object) -> None"""

class _DisablePythonDispatcher:
    def __init__(self) -> None:
        """__init__(self: torch._C._DisablePythonDispatcher) -> None"""
    def __enter__(self):
        """__enter__(self: torch._C._DisablePythonDispatcher) -> None"""
    def __exit__(self, *exc_info: object) -> None:
        """__exit__(self: torch._C._DisablePythonDispatcher, arg0: object, arg1: object, arg2: object) -> None"""

class _EnablePreDispatch:
    def __init__(self) -> None:
        """__init__(self: torch._C._EnablePreDispatch) -> None"""
    def __enter__(self):
        """__enter__(self: torch._C._EnablePreDispatch) -> None"""
    def __exit__(self, *exc_info: object) -> None:
        """__exit__(self: torch._C._EnablePreDispatch, arg0: object, arg1: object, arg2: object) -> None"""

class _DisableFuncTorch:
    def __init__(self) -> None:
        """__init__(self: torch._C._DisableFuncTorch) -> None"""
    def __enter__(self):
        """__enter__(self: torch._C._DisableFuncTorch) -> None"""
    def __exit__(self, *exc_info: object) -> None:
        """__exit__(self: torch._C._DisableFuncTorch, arg0: object, arg1: object, arg2: object) -> None"""

class _DisableAutocast:
    def __init__(self) -> None:
        """__init__(self: torch._C._DisableAutocast) -> None"""
    def __enter__(self):
        """__enter__(self: torch._C._DisableAutocast) -> None"""
    def __exit__(self, *exc_info: object) -> None:
        """__exit__(self: torch._C._DisableAutocast, arg0: object, arg1: object, arg2: object) -> None"""

class _InferenceMode:
    def __init__(self, enabled: _bool) -> None:
        """__init__(self: torch._C._InferenceMode, arg0: bool) -> None"""
    def __enter__(self):
        """__enter__(self: torch._C._InferenceMode) -> None"""
    def __exit__(self, *exc_info: object) -> None:
        """__exit__(self: torch._C._InferenceMode, arg0: object, arg1: object, arg2: object) -> None"""

class LoggerBase: ...
class NoopLogger(LoggerBase): ...
class LockingLogger(LoggerBase): ...

class AggregationType(Enum):
    """
    Members:

    SUM

    AVG
    """

    SUM = ...
    AVG = ...

class FileCheck:
    def run(self, test_string: str) -> None:
        """
        run(*args, **kwargs)
        Overloaded function.

        1. run(self: torch._C.FileCheck, arg0: str) -> None

        2. run(self: torch._C.FileCheck, arg0: torch._C.Graph) -> None

        3. run(self: torch._C.FileCheck, checks_file: str, test_file: str) -> None

        Run

        4. run(self: torch._C.FileCheck, checks_file: str, graph: torch._C.Graph) -> None

        Run
        """
    def check(self, test_string: str) -> FileCheck:
        """check(self: torch._C.FileCheck, arg0: str) -> torch._C.FileCheck"""
    def check_not(self, test_string: str) -> FileCheck:
        """check_not(self: torch._C.FileCheck, arg0: str) -> torch._C.FileCheck"""
    def check_same(self, test_string: str) -> FileCheck:
        """check_same(self: torch._C.FileCheck, arg0: str) -> torch._C.FileCheck"""
    def check_next(self, test_string: str) -> FileCheck:
        """check_next(self: torch._C.FileCheck, arg0: str) -> torch._C.FileCheck"""
    def check_count(self, test_string: str, count: _int, exactly: _bool = ...) -> FileCheck:
        """
        check_count(*args, **kwargs)
        Overloaded function.

        1. check_count(self: torch._C.FileCheck, arg0: str, arg1: typing.SupportsInt, arg2: bool) -> torch._C.FileCheck

        2. check_count(self: torch._C.FileCheck, str: str, count: typing.SupportsInt, exactly: bool = False) -> torch._C.FileCheck

        Check Count
        """
    def check_dag(self, test_string: str) -> FileCheck:
        """check_dag(self: torch._C.FileCheck, arg0: str) -> torch._C.FileCheck"""
    def check_source_highlighted(self, test_string: str) -> FileCheck:
        """check_source_highlighted(self: torch._C.FileCheck, arg0: str) -> torch._C.FileCheck"""
    def check_regex(self, test_string: str) -> FileCheck:
        """check_regex(self: torch._C.FileCheck, arg0: str) -> torch._C.FileCheck"""

class PyTorchFileReader:
    @overload
    def __init__(self, name: str) -> None:
        """
        __init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: torch._C.PyTorchFileReader, arg0: str) -> None

        2. __init__(self: torch._C.PyTorchFileReader, arg0: object) -> None
        """
    @overload
    def __init__(self, buffer: IO[bytes]) -> None:
        """
        __init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: torch._C.PyTorchFileReader, arg0: str) -> None

        2. __init__(self: torch._C.PyTorchFileReader, arg0: object) -> None
        """
    def get_record(self, name: str) -> bytes:
        """get_record(self: torch._C.PyTorchFileReader, arg0: str) -> bytes"""
    def get_all_records(self) -> list[str]:
        """get_all_records(self: torch._C.PyTorchFileReader) -> list[str]"""
    def serialization_id(self) -> str:
        """serialization_id(self: torch._C.PyTorchFileReader) -> str"""

class PyTorchFileWriter:
    @overload
    def __init__(self, name: str, compute_crc32: _bool = ..., storage_alignment: _int = ...) -> None:
        """
        __init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: torch._C.PyTorchFileWriter, file_name: str, compute_crc32: bool = True, storage_alignment: typing.SupportsInt = 64) -> None

        2. __init__(self: torch._C.PyTorchFileWriter, buffer: object, compute_crc32: bool = True, storage_alignment: typing.SupportsInt = 64) -> None

        3. __init__(self: torch._C.PyTorchFileWriter, writer_func: collections.abc.Callable[[types.CapsuleType, typing.SupportsInt], int], compute_crc32: bool = True, storage_alignment: typing.SupportsInt = 64) -> None
        """
    @overload
    def __init__(self, buffer: IO[bytes], compute_crc32: _bool = ..., storage_alignment: _int = ...) -> None:
        """
        __init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: torch._C.PyTorchFileWriter, file_name: str, compute_crc32: bool = True, storage_alignment: typing.SupportsInt = 64) -> None

        2. __init__(self: torch._C.PyTorchFileWriter, buffer: object, compute_crc32: bool = True, storage_alignment: typing.SupportsInt = 64) -> None

        3. __init__(self: torch._C.PyTorchFileWriter, writer_func: collections.abc.Callable[[types.CapsuleType, typing.SupportsInt], int], compute_crc32: bool = True, storage_alignment: typing.SupportsInt = 64) -> None
        """
    def write_record(self, name: str, data: Storage | bytes | _int, size: _int) -> None:
        """
        write_record(*args, **kwargs)
        Overloaded function.

        1. write_record(self: torch._C.PyTorchFileWriter, arg0: str, arg1: str, arg2: typing.SupportsInt) -> None

        2. write_record(self: torch._C.PyTorchFileWriter, arg0: str, arg1: bytes, arg2: typing.SupportsInt) -> None

        3. write_record(self: torch._C.PyTorchFileWriter, arg0: str, arg1: torch.StorageBase, arg2: typing.SupportsInt) -> None

        4. write_record(self: torch._C.PyTorchFileWriter, arg0: str, arg1: typing.SupportsInt, arg2: typing.SupportsInt) -> None
        """
    def write_end_of_file(self) -> None:
        """write_end_of_file(self: torch._C.PyTorchFileWriter) -> None"""
    def set_min_version(self, version: _int) -> None:
        """set_min_version(self: torch._C.PyTorchFileWriter, arg0: typing.SupportsInt) -> None"""
    def get_all_written_records(self) -> list[str]:
        """get_all_written_records(self: torch._C.PyTorchFileWriter) -> set[str]"""
    def archive_name(self) -> str:
        """archive_name(self: torch._C.PyTorchFileWriter) -> str"""
    def serialization_id(self) -> str:
        """serialization_id(self: torch._C.PyTorchFileWriter) -> str"""

class Generator:
    """
    Generator(device='cpu') -> Generator

    Creates and returns a generator object that manages the state of the algorithm which
    produces pseudo random numbers. Used as a keyword argument in many :ref:`inplace-random-sampling`
    functions.

    Arguments:
        device (:class:`torch.device`, optional): the desired device for the generator.

    Returns:
        Generator: An torch.Generator object.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> g_cpu = torch.Generator()
        >>> g_cuda = torch.Generator(device='cuda')
    """

    device: _device
    def __init__(self, device: DeviceLikeType | None = ...) -> None: ...
    def __reduce__(self) -> tuple[type[Generator], tuple[_device], tuple[_int, _int | None, Tensor]]: ...
    def __setstate__(self, state: tuple[_int, _int | None, Tensor]) -> None: ...
    def get_state(self) -> Tensor:
        """
        Generator.get_state() -> Tensor

        Returns the Generator state as a ``torch.ByteTensor``.

        Returns:
            Tensor: A ``torch.ByteTensor`` which contains all the necessary bits
            to restore a Generator to a specific point in time.

        Example::

            >>> g_cpu = torch.Generator()
            >>> g_cpu.get_state()
        """
    def set_state(self, _new_state: Tensor) -> Generator:
        """
        Generator.set_state(new_state) -> void

        Sets the Generator state.

        Arguments:
            new_state (torch.ByteTensor): The desired state.

        Example::

            >>> g_cpu = torch.Generator()
            >>> g_cpu_other = torch.Generator()
            >>> g_cpu.set_state(g_cpu_other.get_state())
        """
    def clone_state(self) -> Generator:
        """
        Generator.clone_state() -> torch.Generator

        Clones the current state of the generator and returns a new generator pointing to this cloned state.
        This method is beneficial for preserving a particular state of a generator to restore at a later point.

        Returns:
            torch.Generator: A Generator pointing to the newly cloned state.

        Example:
            >>> g_cuda = torch.Generator(device='cuda')
            >>> cloned_state = g_cuda.clone_state()
        """
    def graphsafe_get_state(self) -> Generator:
        """
        Generator.graphsafe_get_state() -> torch.Generator

        Retrieves the current state of the generator in a manner that is safe for graph capture.
        This method is crucial for ensuring that the generator's state can be captured in the CUDA graph.

        Returns:
            torch.Generator: A Generator point to the current state of the generator

        Example:
            >>> g_cuda = torch.Generator(device='cuda')
            >>> current_state = g_cuda.graphsafe_get_state()
        """
    def graphsafe_set_state(self, _new_state: Generator) -> Generator:
        """
        Generator.graphsafe_set_state(state) -> None

        Sets the state of the generator to the specified state in a manner that is safe for use in graph capture.
        This method is crucial for ensuring that the generator's state can be captured in the CUDA graph.

        Arguments:
            state (torch.Generator): A Generator point to the new state for the generator, typically obtained from `graphsafe_get_state`.

        Example:
            >>> g_cuda = torch.Generator(device='cuda')
            >>> g_cuda_other = torch.Generator(device='cuda')
            >>> current_state = g_cuda_other.graphsafe_get_state()
            >>> g_cuda.graphsafe_set_state(current_state)
        """
    def set_offset(self, offset: _int) -> Generator: ...
    def get_offset(self) -> _int: ...
    def manual_seed(self, seed: _int) -> Generator:
        """
        Generator.manual_seed(seed) -> Generator

        Sets the seed for generating random numbers. Returns a `torch.Generator` object. Any 32-bit integer is a valid seed.

        Arguments:
            seed (int): The desired seed. Value must be within the inclusive range
                `[-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]`. Otherwise, a RuntimeError
                is raised. Negative inputs are remapped to positive values with the formula
                `0xffff_ffff_ffff_ffff + seed`.

        Returns:
            Generator: An torch.Generator object.

        Example::

            >>> g_cpu = torch.Generator()
            >>> g_cpu.manual_seed(2147483647)
        """
    def seed(self) -> _int:
        """
        Generator.seed() -> int

        Gets a non-deterministic random number from std::random_device or the current
        time and uses it to seed a Generator.

        Example::

            >>> g_cpu = torch.Generator()
            >>> g_cpu.seed()
            1516516984916
        """
    def initial_seed(self) -> _int:
        """
        Generator.initial_seed() -> int

        Returns the initial seed for generating random numbers.

        Example::

            >>> g_cpu = torch.Generator()
            >>> g_cpu.initial_seed()
            2147483647
        """

class _DispatchOperatorHandle:
    def schema(self) -> FunctionSchema:
        """schema(self: torch._C._DispatchOperatorHandle) -> torch._C.FunctionSchema"""
    def debug(self) -> str:
        """debug(self: torch._C._DispatchOperatorHandle) -> str"""
    def redispatch_boxed(self, keyset: DispatchKeySet, *args, **kwargs) -> Any:
        """redispatch_boxed(self: object, arg0: c10::DispatchKeySet, *args, **kwargs) -> object"""

class _DispatchModule:
    def reset(self) -> None:
        """reset(self: object) -> None"""
    def def_(self, schema: str, alias: str = ...) -> _DispatchModule:
        """def_(self: object, schema: str, alias: str = '') -> object"""
    def def_legacy(self, schema: str) -> _DispatchModule:
        """def_legacy(self: object, schema: str) -> object"""
    def def_name_t_t(self, name: str, dispatch: str, debug: str = ...) -> _DispatchModule:
        """def_name_t_t(self: object, name: str, dispatch: str = '', debug: str = 'default_def_name_t_t') -> object"""
    def def_schema_t_t(self, schema: str, dispatch: str, alias: str, debug: str = ...) -> _DispatchModule:
        """def_schema_t_t(self: object, name: str, dispatch: str = '', alias: str = '', debug: str = 'default_def_schema_t_t') -> object"""
    def impl_t_t(self, name: str, dispatch: str, debug: str = ...) -> _DispatchModule:
        """impl_t_t(self: object, name: str, dispatch: str = '', debug: str = 'impl_t_t') -> object"""
    def impl_with_aoti_compile(self, ns: str, op_name_with_overload: str, dispatch: _dispatchkey) -> None:
        """impl_with_aoti_compile(self: object, ns: str, op_name_with_overload: str, dispatch: c10::DispatchKey) -> None"""
    def impl(self, name: str, dispatch: _dispatchkey, func: Callable) -> None:
        """impl(self: object, name: str, dispatch: c10::DispatchKey, func: object, with_keyset: bool = False) -> None"""
    def define(self, schema: str, alias: str = ...) -> str:
        """define(self: object, schema: str, alias_analysis: str = '', tags: collections.abc.Sequence[torch._C.Tag] = []) -> str"""
    def fallback_fallthrough(self, dispatch: str = ...) -> _DispatchModule:
        """fallback_fallthrough(self: object, dispatch: str = '') -> object"""
    def fallback(self, dispatch: _dispatchkey, func: Callable, with_keyset: _bool = ...) -> None:
        """fallback(self: object, dispatch: c10::DispatchKey, func: object, with_keyset: bool = False) -> None"""

_after_ADInplaceOrView_keyset: DispatchKeySet
_after_autograd_keyset: DispatchKeySet

class _SafeKernelFunction:
    def call_boxed(self, keyset: DispatchKeySet, *args, **kwargs) -> Any:
        """call_boxed(self: torch._C._SafeKernelFunction, arg0: c10::DispatchKeySet, *args, **kwargs) -> object"""
    @property
    def op_handle(self) -> _DispatchOperatorHandle: ...

class DispatchKey(Enum):
    """
    Members:

    Undefined

    CompositeExplicitAutogradNonFunctional

    CompositeExplicitAutograd

    CompositeImplicitAutogradNestedTensor

    CompositeImplicitAutograd

    AutogradNestedTensor

    AutogradOther

    Autograd

    Conjugate

    ZeroTensor

    Negative

    BackendSelect

    ADInplaceOrView

    PythonTLSSnapshot

    Python

    FuncTorchDynamicLayerFrontMode

    FuncTorchDynamicLayerBackMode

    FuncTorchBatchedDecomposition

    FuncTorchBatched

    FuncTorchVmapMode

    FuncTorchGradWrapper

    PythonDispatcher

    PreDispatch

    Functionalize

    AutocastCPU

    AutocastMPS

    AutocastXPU

    AutocastHPU

    AutocastIPU

    AutocastCUDA

    AutocastPrivateUse1

    Dense

    StartOfDenseBackends

    CPU

    CUDA

    HIP

    XLA

    MPS

    IPU

    XPU

    HPU

    VE

    Lazy

    MTIA

    MAIA

    PrivateUse1

    PrivateUse2

    PrivateUse3

    Meta

    EndOfDenseBackends

    Quantized

    StartOfQuantizedBackends

    QuantizedCPU

    QuantizedCUDA

    QuantizedHIP

    QuantizedXLA

    QuantizedMPS

    QuantizedIPU

    QuantizedXPU

    QuantizedHPU

    QuantizedVE

    QuantizedLazy

    QuantizedMTIA

    QuantizedMAIA

    QuantizedPrivateUse1

    QuantizedPrivateUse2

    QuantizedPrivateUse3

    QuantizedMeta

    EndOfQuantizedBackends

    Sparse

    StartOfSparseBackends

    SparseCPU

    SparseCUDA

    SparseHIP

    SparseXLA

    SparseMPS

    SparseIPU

    SparseXPU

    SparseHPU

    SparseVE

    SparseLazy

    SparseMTIA

    SparseMAIA

    SparsePrivateUse1

    SparsePrivateUse2

    SparsePrivateUse3

    SparseMeta

    EndOfSparseBackends

    SparseCsr

    StartOfSparseCsrBackends

    SparseCsrCPU

    SparseCsrCUDA

    SparseCsrHIP

    SparseCsrXLA

    SparseCsrMPS

    SparseCsrIPU

    SparseCsrXPU

    SparseCsrHPU

    SparseCsrVE

    SparseCsrLazy

    SparseCsrMTIA

    SparseCsrMAIA

    SparseCsrPrivateUse1

    SparseCsrPrivateUse2

    SparseCsrPrivateUse3

    SparseCsrMeta

    EndOfSparseCsrBackends

    NestedTensor

    StartOfNestedTensorBackends

    NestedTensorCPU

    NestedTensorCUDA

    NestedTensorHIP

    NestedTensorXLA

    NestedTensorMPS

    NestedTensorIPU

    NestedTensorXPU

    NestedTensorHPU

    NestedTensorVE

    NestedTensorLazy

    NestedTensorMTIA

    NestedTensorMAIA

    NestedTensorPrivateUse1

    NestedTensorPrivateUse2

    NestedTensorPrivateUse3

    NestedTensorMeta

    EndOfNestedTensorBackends

    AutogradFunctionality

    StartOfAutogradFunctionalityBackends

    AutogradCPU

    AutogradCUDA

    AutogradHIP

    AutogradXLA

    AutogradMPS

    AutogradIPU

    AutogradXPU

    AutogradHPU

    AutogradVE

    AutogradLazy

    AutogradMTIA

    AutogradMAIA

    AutogradPrivateUse1

    AutogradPrivateUse2

    AutogradPrivateUse3

    AutogradMeta

    EndOfAutogradFunctionalityBackends
    """

    Undefined = ...
    FPGA = ...
    MAIA = ...
    Vulkan = ...
    Metal = ...
    MKLDNN = ...
    OpenGL = ...
    OpenCL = ...
    IDEEP = ...
    CustomRNGKeyId = ...
    MkldnnCPU = ...
    Sparse = ...
    SparseCsr = ...
    NestedTensor = ...
    Dense = ...
    PythonTLSSnapshot = ...
    PreDispatch = ...
    PythonDispatcher = ...
    Python = ...
    FuncTorchDynamicLayerBackMode = ...
    ZeroTensor = ...
    Conjugate = ...
    Negative = ...
    BackendSelect = ...
    Named = ...
    AutogradOther = ...
    AutogradFunctionality = ...
    AutogradNestedTensor = ...
    Tracer = ...
    Autocast = ...
    AutocastCPU = ...
    AutocastCUDA = ...
    Batched = ...
    VmapMode = ...
    FuncTorchGradWrapper = ...
    FuncTorchBatched = ...
    BatchedNestedTensor = ...
    FuncTorchVmapMode = ...
    FuncTorchDynamicLayerFrontMode = ...
    Functionalize = ...
    TESTING_ONLY_GenericWrapper = ...
    TESTING_ONLY_GenericMode = ...
    ADInplaceOrView = ...
    Autograd = ...
    CompositeImplicitAutograd = ...
    CompositeImplicitAutogradNestedTensor = ...
    CompositeExplicitAutograd = ...
    CompositeExplicitAutogradNonFunctional = ...
    FuncTorchBatchedDecomposition = ...
    CPU = ...
    CUDA = ...
    HIP = ...
    XLA = ...
    MTIA = ...
    MPS = ...
    IPU = ...
    XPU = ...
    HPU = ...
    VE = ...
    Lazy = ...
    Meta = ...
    PrivateUse1 = ...
    PrivateUse2 = ...
    PrivateUse3 = ...
    QuantizedCPU = ...
    QuantizedCUDA = ...
    QuantizedHIP = ...
    QuantizedXLA = ...
    QuantizedMTIA = ...
    QuantizedMPS = ...
    QuantizedIPU = ...
    QuantizedXPU = ...
    QuantizedHPU = ...
    QuantizedVE = ...
    QuantizedLazy = ...
    QuantizedMeta = ...
    QuantizedPrivateUse1 = ...
    QuantizedPrivateUse2 = ...
    QuantizedPrivateUse3 = ...
    SparseCPU = ...
    SparseCUDA = ...
    SparseHIP = ...
    SparseXLA = ...
    SparseMTIA = ...
    SparseMPS = ...
    SparseIPU = ...
    SparseXPU = ...
    SparseHPU = ...
    SparseVE = ...
    SparseLazy = ...
    SparseMeta = ...
    SparsePrivateUse1 = ...
    SparsePrivateUse2 = ...
    SparsePrivateUse3 = ...
    SparseCsrCPU = ...
    SparseCsrCUDA = ...
    SparseCsrHIP = ...
    SparseCsrXLA = ...
    SparseCsrMTIA = ...
    SparseCsrMPS = ...
    SparseCsrIPU = ...
    SparseCsrXPU = ...
    SparseCsrHPU = ...
    SparseCsrVE = ...
    SparseCsrLazy = ...
    SparseCsrMeta = ...
    SparseCsrPrivateUse1 = ...
    SparseCsrPrivateUse2 = ...
    SparseCsrPrivateUse3 = ...
    NestedTensorCPU = ...
    NestedTensorCUDA = ...
    NestedTensorHIP = ...
    NestedTensorXLA = ...
    NestedTensorMTIA = ...
    NestedTensorMPS = ...
    NestedTensorIPU = ...
    NestedTensorXPU = ...
    NestedTensorHPU = ...
    NestedTensorVE = ...
    NestedTensorLazy = ...
    NestedTensorMeta = ...
    NestedTensorPrivateUse1 = ...
    NestedTensorPrivateUse2 = ...
    NestedTensorPrivateUse3 = ...
    AutogradCPU = ...
    AutogradCUDA = ...
    AutogradHIP = ...
    AutogradXLA = ...
    AutogradMTIA = ...
    AutogradMPS = ...
    AutogradIPU = ...
    AutogradXPU = ...
    AutogradHPU = ...
    AutogradVE = ...
    AutogradLazy = ...
    AutogradMeta = ...
    AutogradPrivateUse1 = ...
    AutogradPrivateUse2 = ...
    AutogradPrivateUse3 = ...

class DispatchKeySet:
    def __init__(self, key: DispatchKey) -> None:
        """__init__(self: torch._C.DispatchKeySet, arg0: torch._C.DispatchKey) -> None"""
    def __or__(self, other: DispatchKeySet) -> DispatchKeySet:
        """__or__(self: torch._C.DispatchKeySet, arg0: torch._C.DispatchKeySet) -> torch._C.DispatchKeySet"""
    def __sub__(self, other: DispatchKeySet) -> DispatchKeySet:
        """__sub__(self: torch._C.DispatchKeySet, arg0: torch._C.DispatchKeySet) -> torch._C.DispatchKeySet"""
    def __and__(self, other: DispatchKeySet) -> DispatchKeySet:
        """__and__(self: torch._C.DispatchKeySet, arg0: torch._C.DispatchKeySet) -> torch._C.DispatchKeySet"""
    def raw_repr(self) -> _int:
        """raw_repr(self: torch._C.DispatchKeySet) -> int"""
    @staticmethod
    def from_raw_repr(raw: _int) -> DispatchKeySet:
        """from_raw_repr(arg0: typing.SupportsInt) -> torch._C.DispatchKeySet"""
    def highestPriorityTypeId(self) -> DispatchKey:
        """highestPriorityTypeId(self: torch._C.DispatchKeySet) -> torch._C.DispatchKey"""
    def has(self, k: _dispatchkey) -> _bool:
        """has(self: torch._C.DispatchKeySet, arg0: torch._C.DispatchKey) -> bool"""
    def add(self, k: _dispatchkey) -> DispatchKeySet:
        """add(self: torch._C.DispatchKeySet, arg0: torch._C.DispatchKey) -> torch._C.DispatchKeySet"""
    def remove(self, k: _dispatchkey) -> DispatchKeySet:
        """remove(self: torch._C.DispatchKeySet, arg0: torch._C.DispatchKey) -> torch._C.DispatchKeySet"""

_dispatch_autogradother_backends: DispatchKeySet
_additional_keys_to_prop_for_wrapper_tensors: DispatchKeySet

class _ExcludeDispatchKeyGuard:
    def __init__(self, keyset: DispatchKeySet) -> None:
        """__init__(self: torch._C._ExcludeDispatchKeyGuard, arg0: torch._C.DispatchKeySet) -> None"""
    def __enter__(self):
        """__enter__(self: torch._C._ExcludeDispatchKeyGuard) -> None"""
    def __exit__(self, *exc_info: object) -> None:
        """__exit__(self: torch._C._ExcludeDispatchKeyGuard, arg0: object, arg1: object, arg2: object) -> None"""

class _IncludeDispatchKeyGuard:
    def __init__(self, k: DispatchKey) -> None:
        """__init__(self: torch._C._IncludeDispatchKeyGuard, arg0: torch._C.DispatchKey) -> None"""
    def __enter__(self):
        """__enter__(self: torch._C._IncludeDispatchKeyGuard) -> None"""
    def __exit__(self, *exc_info: object) -> None:
        """__exit__(self: torch._C._IncludeDispatchKeyGuard, arg0: object, arg1: object, arg2: object) -> None"""

class _ForceDispatchKeyGuard:
    def __init__(self, include: DispatchKeySet, exclude: DispatchKeySet) -> None:
        """__init__(self: torch._C._ForceDispatchKeyGuard, arg0: torch._C.DispatchKeySet, arg1: torch._C.DispatchKeySet) -> None"""
    def __enter__(self):
        """__enter__(self: torch._C._ForceDispatchKeyGuard) -> None"""
    def __exit__(self, *exc_info: object) -> None:
        """__exit__(self: torch._C._ForceDispatchKeyGuard, arg0: object, arg1: object, arg2: object) -> None"""

class _PreserveDispatchKeyGuard:
    def __init__(self) -> None:
        """__init__(self: torch._C._PreserveDispatchKeyGuard) -> None"""
    def __enter__(self):
        """__enter__(self: torch._C._PreserveDispatchKeyGuard) -> None"""
    def __exit__(self, *exc_info: object) -> None:
        """__exit__(self: torch._C._PreserveDispatchKeyGuard, arg0: object, arg1: object, arg2: object) -> None"""

class _AutoDispatchBelowAutograd:
    def __init__(self) -> None:
        """__init__(self: torch._C._AutoDispatchBelowAutograd) -> None"""
    def __enter__(self):
        """__enter__(self: torch._C._AutoDispatchBelowAutograd) -> None"""
    def __exit__(self, *exc_info: object) -> None:
        """__exit__(self: torch._C._AutoDispatchBelowAutograd, arg0: object, arg1: object, arg2: object) -> None"""

class _AutoDispatchBelowADInplaceOrView:
    def __init__(self) -> None:
        """__init__(self: torch._C._AutoDispatchBelowADInplaceOrView) -> None"""
    def __enter__(self):
        """__enter__(self: torch._C._AutoDispatchBelowADInplaceOrView) -> None"""
    def __exit__(self, *exc_info: object) -> None:
        """__exit__(self: torch._C._AutoDispatchBelowADInplaceOrView, arg0: object, arg1: object, arg2: object) -> None"""

class _TorchDispatchModeKey(Enum):
    """
    Members:

    FUNCTIONAL

    PROXY

    FAKE
    """

    FAKE = ...
    PROXY = ...
    FUNCTIONAL = ...

class _SetExcludeDispatchKeyGuard:
    def __init__(self, k: DispatchKey, enabled: _bool) -> None:
        """__init__(self: torch._C._SetExcludeDispatchKeyGuard, arg0: torch._C.DispatchKey, arg1: bool) -> None"""
    def __enter__(self):
        """__enter__(self: torch._C._SetExcludeDispatchKeyGuard) -> None"""
    def __exit__(self, *exc_info: object) -> None:
        """__exit__(self: torch._C._SetExcludeDispatchKeyGuard, arg0: object, arg1: object, arg2: object) -> None"""

class _SchemaInfo:
    def __init__(self, schema: FunctionSchema) -> None:
        """__init__(self: torch._C._SchemaInfo, arg0: c10::FunctionSchema) -> None"""
    @overload
    def is_mutable(self) -> _bool:
        """
        is_mutable(*args, **kwargs)
        Overloaded function.

        1. is_mutable(self: torch._C._SchemaInfo) -> bool

        2. is_mutable(self: torch._C._SchemaInfo, arg0: torch._C._SchemaArgument) -> bool

        3. is_mutable(self: torch._C._SchemaInfo, arg0: str) -> bool
        """
    @overload
    def is_mutable(self, name: str) -> _bool:
        """
        is_mutable(*args, **kwargs)
        Overloaded function.

        1. is_mutable(self: torch._C._SchemaInfo) -> bool

        2. is_mutable(self: torch._C._SchemaInfo, arg0: torch._C._SchemaArgument) -> bool

        3. is_mutable(self: torch._C._SchemaInfo, arg0: str) -> bool
        """
    def has_argument(self, name: str) -> _bool:
        """has_argument(self: torch._C._SchemaInfo, arg0: str) -> bool"""

class BenchmarkConfig:
    num_calling_threads: _int
    num_worker_threads: _int
    num_warmup_iters: _int
    num_iters: _int
    profiler_output_path: str

class BenchmarkExecutionStats:
    latency_avg_ms: _float
    num_iters: _int

class ThroughputBenchmark:
    def __init__(self, module: Any) -> None:
        """
        __init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: torch._C.ThroughputBenchmark, arg0: torch._C.ScriptModule) -> None

        2. __init__(self: torch._C.ThroughputBenchmark, arg0: object) -> None
        """
    def add_input(self, *args: Any, **kwargs: Any) -> None:
        """add_input(self: torch._C.ThroughputBenchmark, *args, **kwargs) -> None"""
    def run_once(self, *args: Any, **kwargs: Any) -> Any:
        """run_once(self: torch._C.ThroughputBenchmark, *args, **kwargs) -> object"""
    def benchmark(self, config: BenchmarkConfig) -> BenchmarkExecutionStats:
        """benchmark(self: torch._C.ThroughputBenchmark, arg0: torch._C.BenchmarkConfig) -> torch._C.BenchmarkExecutionStats"""

class StorageBase: ...
class DoubleTensor(Tensor): ...
class FloatTensor(Tensor): ...
class BFloat16Tensor(Tensor): ...
class LongTensor(Tensor): ...
class IntTensor(Tensor): ...
class ShortTensor(Tensor): ...
class HalfTensor(Tensor): ...
class CharTensor(Tensor): ...
class ByteTensor(Tensor): ...
class BoolTensor(Tensor): ...

class _ImperativeEngine:
    def queue_callback(self, callback: Callable[[], None]) -> None: ...
    def run_backward(self, *args: Any, **kwargs: Any) -> tuple[Tensor, ...]: ...
    def is_checkpoint_valid(self) -> _bool: ...

class _TensorMeta(type): ...

type _Index = (
    SupportsIndex
    | _bool
    | _int
    | slice
    | EllipsisType
    | Tensor
    | None
    | _NestedSequence[_bool | _int | slice | EllipsisType | Tensor | None]
)

class TensorBase(metaclass=_TensorMeta):
    requires_grad: _bool
    retains_grad: _bool
    shape: Size
    data: Tensor
    names: list[str]
    device: _device
    dtype: _dtype
    layout: _layout
    real: Tensor
    imag: Tensor
    T: Tensor
    H: Tensor
    mT: Tensor
    mH: Tensor
    ndim: _int
    output_nr: _int
    _version: _int
    _base: Tensor | None
    _cdata: _int
    grad_fn: _Node | None
    _grad_fn: Any
    _grad: Tensor | None
    grad: Tensor | None
    _backward_hooks: dict[_int, Callable[[Tensor], Tensor | None]] | None
    nbytes: _int
    itemsize: _int
    _has_symbolic_sizes_strides: _bool
    def __abs__(self) -> Tensor: ...
    def __add__(self, other: Tensor | Number | _complex) -> Tensor: ...
    @overload
    def __and__(self, other: Tensor) -> Tensor: ...
    @overload
    def __and__(self, other: Number | _complex) -> Tensor: ...
    @overload
    def __and__(self, other: Tensor | _int) -> Tensor: ...
    def __bool__(self) -> _bool: ...
    def __complex__(self) -> _complex: ...
    def __contains__(self, item: Any, /) -> _bool: ...
    def __div__(self, other: Tensor | Number | _complex) -> Tensor: ...
    @overload
    def __eq__(self, other: Tensor | Number | _complex) -> Tensor: ...
    @overload
    def __eq__(self, other: object) -> _bool: ...
    def __float__(self) -> _float: ...
    def __floordiv__(self, other: Tensor | Number | _complex) -> Tensor: ...
    def __ge__(self, other: Tensor | Number | _complex) -> Tensor: ...
    def __getitem__(self, indices: _Index | tuple[_Index, ...], /) -> Tensor:
        """Return self[key]."""
    def __gt__(self, other: Tensor | Number | _complex) -> Tensor: ...
    def __iadd__(self, other: Tensor | Number | _complex) -> Self: ...
    @overload
    def __iand__(self, other: Tensor) -> Tensor: ...
    @overload
    def __iand__(self, other: Number | _complex) -> Tensor: ...
    @overload
    def __iand__(self, other: Tensor | _int) -> Tensor: ...
    def __idiv__(self, other: Tensor | Number | _complex) -> Tensor: ...
    def __ifloordiv__(self, other: Tensor | Number | _complex) -> Self: ...
    @overload
    def __ilshift__(self, other: Tensor) -> Tensor: ...
    @overload
    def __ilshift__(self, other: Number | _complex) -> Tensor: ...
    @overload
    def __ilshift__(self, other: Tensor | _int) -> Tensor: ...
    def __imod__(self, other: Tensor | Number | _complex) -> Self: ...
    def __imul__(self, other: Tensor | Number | _complex) -> Self: ...
    def __index__(self) -> _int: ...
    @overload
    def __init__(self, *args: Any, device: DeviceLikeType | None = ...) -> None: ...
    @overload
    def __init__(self, storage: Storage) -> None: ...
    @overload
    def __init__(self, other: Tensor) -> None: ...
    @overload
    def __init__(self, size: _size, *, device: DeviceLikeType | None = ...) -> None: ...
    def __int__(self) -> _int: ...
    def __invert__(self) -> Tensor: ...
    @overload
    def __ior__(self, other: Tensor) -> Tensor: ...
    @overload
    def __ior__(self, other: Number | _complex) -> Tensor: ...
    @overload
    def __ior__(self, other: Tensor | _int) -> Tensor: ...
    @overload
    def __irshift__(self, other: Tensor) -> Tensor: ...
    @overload
    def __irshift__(self, other: Number | _complex) -> Tensor: ...
    @overload
    def __irshift__(self, other: Tensor | _int) -> Tensor: ...
    def __isub__(self, other: Tensor | Number | _complex) -> Self: ...
    @overload
    def __ixor__(self, other: Tensor) -> Tensor: ...
    @overload
    def __ixor__(self, other: Number | _complex) -> Tensor: ...
    @overload
    def __ixor__(self, other: Tensor | _int) -> Tensor: ...
    def __le__(self, other: Tensor | Number | _complex) -> Tensor: ...
    def __long__(self) -> _int: ...
    @overload
    def __lshift__(self, other: Tensor) -> Tensor: ...
    @overload
    def __lshift__(self, other: Number | _complex) -> Tensor: ...
    @overload
    def __lshift__(self, other: Tensor | _int) -> Tensor: ...
    def __lt__(self, other: Tensor | Number | _complex) -> Tensor: ...
    def __matmul__(self, other: Tensor | Number | _complex) -> Tensor: ...
    def __mod__(self, other: Tensor | Number | _complex) -> Tensor: ...
    def __mul__(self, other: Tensor | Number | _complex) -> Tensor: ...
    @overload
    def __ne__(self, other: Tensor | Number | _complex) -> Tensor: ...
    @overload
    def __ne__(self, other: object) -> _bool: ...
    def __neg__(self) -> Tensor: ...
    def __new__(cls, *args, **kwargs) -> Self: ...
    def __nonzero__(self) -> _bool: ...
    @overload
    def __or__(self, other: Tensor) -> Tensor: ...
    @overload
    def __or__(self, other: Number | _complex) -> Tensor: ...
    @overload
    def __or__(self, other: Tensor | _int) -> Tensor: ...
    def __pow__(self, other: Tensor | Number | _complex) -> Tensor: ...
    def __radd__(self, other: Tensor | Number | _complex) -> Tensor: ...
    def __rand__(self, other: Tensor | _int) -> Tensor: ...
    def __rfloordiv__(self, other: Tensor | Number | _complex) -> Tensor: ...
    def __rmul__(self, other: Tensor | Number | _complex) -> Tensor: ...
    def __ror__(self, other: Tensor | _int) -> Tensor: ...
    def __rpow__(self, other: Tensor | Number | _complex) -> Tensor: ...
    @overload
    def __rshift__(self, other: Tensor) -> Tensor: ...
    @overload
    def __rshift__(self, other: Number | _complex) -> Tensor: ...
    @overload
    def __rshift__(self, other: Tensor | _int) -> Tensor: ...
    def __rsub__(self, other: Tensor | Number | _complex) -> Tensor: ...
    def __rtruediv__(self, other: Tensor | Number | _complex) -> Tensor: ...
    def __rxor__(self, other: Tensor | _int) -> Tensor: ...
    def __setitem__(self, indices: _Index | tuple[_Index, ...], value: Tensor | Number, /) -> None:
        """Set self[key] to value."""
    def __sub__(self, other: Tensor | Number | _complex) -> Tensor: ...
    def __truediv__(self, other: Tensor | Number | _complex) -> Tensor: ...
    @overload
    def __xor__(self, other: Tensor) -> Tensor: ...
    @overload
    def __xor__(self, other: Number | _complex) -> Tensor: ...
    @overload
    def __xor__(self, other: Tensor | _int) -> Tensor: ...
    def abs(self) -> Tensor:
        """
        abs() -> Tensor

        See :func:`torch.abs`
        """
    def abs_(self) -> Tensor:
        """
        abs_() -> Tensor

        In-place version of :meth:`~Tensor.abs`
        """
    def absolute(self) -> Tensor:
        """
        absolute() -> Tensor

        Alias for :func:`abs`
        """
    def absolute_(self) -> Tensor:
        """
        absolute_() -> Tensor

        In-place version of :meth:`~Tensor.absolute`
        Alias for :func:`abs_`
        """
    def acos(self) -> Tensor:
        """
        acos() -> Tensor

        See :func:`torch.acos`
        """
    def acos_(self) -> Tensor:
        """
        acos_() -> Tensor

        In-place version of :meth:`~Tensor.acos`
        """
    def acosh(self) -> Tensor:
        """
        acosh() -> Tensor

        See :func:`torch.acosh`
        """
    def acosh_(self) -> Tensor:
        """
        acosh_() -> Tensor

        In-place version of :meth:`~Tensor.acosh`
        """
    def add(
        self,
        other: Tensor | Number | _complex | torch.SymInt | torch.SymFloat,
        *,
        alpha: Number | _complex | None = ...,
        out: Tensor | None = ...,
    ) -> Tensor:
        """
        add(other, *, alpha=1) -> Tensor

        Add a scalar or tensor to :attr:`self` tensor. If both :attr:`alpha`
        and :attr:`other` are specified, each element of :attr:`other` is scaled by
        :attr:`alpha` before being used.

        When :attr:`other` is a tensor, the shape of :attr:`other` must be
        :ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying
        tensor

        See :func:`torch.add`
        """
    def add_(
        self,
        other: Tensor | Number | _complex | torch.SymInt | torch.SymFloat,
        *,
        alpha: Number | _complex | None = ...,
    ) -> Tensor:
        """
        add_(other, *, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.add`
        """
    def addbmm(
        self, batch1: Tensor, batch2: Tensor, *, beta: Number | _complex = ..., alpha: Number | _complex = ...
    ) -> Tensor:
        """
        addbmm(batch1, batch2, *, beta=1, alpha=1) -> Tensor

        See :func:`torch.addbmm`
        """
    def addbmm_(
        self, batch1: Tensor, batch2: Tensor, *, beta: Number | _complex = ..., alpha: Number | _complex = ...
    ) -> Tensor:
        """
        addbmm_(batch1, batch2, *, beta=1, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.addbmm`
        """
    def addcdiv(self, tensor1: Tensor, tensor2: Tensor, *, value: Number | _complex = ...) -> Tensor:
        """
        addcdiv(tensor1, tensor2, *, value=1) -> Tensor

        See :func:`torch.addcdiv`
        """
    def addcdiv_(self, tensor1: Tensor, tensor2: Tensor, *, value: Number | _complex = ...) -> Tensor:
        """
        addcdiv_(tensor1, tensor2, *, value=1) -> Tensor

        In-place version of :meth:`~Tensor.addcdiv`
        """
    def addcmul(self, tensor1: Tensor, tensor2: Tensor, *, value: Number | _complex = ...) -> Tensor:
        """
        addcmul(tensor1, tensor2, *, value=1) -> Tensor

        See :func:`torch.addcmul`
        """
    def addcmul_(self, tensor1: Tensor, tensor2: Tensor, *, value: Number | _complex = ...) -> Tensor:
        """
        addcmul_(tensor1, tensor2, *, value=1) -> Tensor

        In-place version of :meth:`~Tensor.addcmul`
        """
    def addmm(
        self, mat1: Tensor, mat2: Tensor, *, beta: Number | _complex = ..., alpha: Number | _complex = ...
    ) -> Tensor:
        """
        addmm(mat1, mat2, *, beta=1, alpha=1) -> Tensor

        See :func:`torch.addmm`
        """
    def addmm_(
        self, mat1: Tensor, mat2: Tensor, *, beta: Number | _complex = ..., alpha: Number | _complex = ...
    ) -> Tensor:
        """
        addmm_(mat1, mat2, *, beta=1, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.addmm`
        """
    def addmv(
        self, mat: Tensor, vec: Tensor, *, beta: Number | _complex = ..., alpha: Number | _complex = ...
    ) -> Tensor:
        """
        addmv(mat, vec, *, beta=1, alpha=1) -> Tensor

        See :func:`torch.addmv`
        """
    def addmv_(
        self, mat: Tensor, vec: Tensor, *, beta: Number | _complex = ..., alpha: Number | _complex = ...
    ) -> Tensor:
        """
        addmv_(mat, vec, *, beta=1, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.addmv`
        """
    def addr(
        self, vec1: Tensor, vec2: Tensor, *, beta: Number | _complex = ..., alpha: Number | _complex = ...
    ) -> Tensor:
        """
        addr(vec1, vec2, *, beta=1, alpha=1) -> Tensor

        See :func:`torch.addr`
        """
    def addr_(
        self, vec1: Tensor, vec2: Tensor, *, beta: Number | _complex = ..., alpha: Number | _complex = ...
    ) -> Tensor:
        """
        addr_(vec1, vec2, *, beta=1, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.addr`
        """
    def adjoint(self) -> Tensor:
        """
        adjoint() -> Tensor

        Alias for :func:`adjoint`
        """
    def align_as(self, other: Tensor) -> Tensor:
        """
        align_as(other) -> Tensor

        Permutes the dimensions of the :attr:`self` tensor to match the dimension order
        in the :attr:`other` tensor, adding size-one dims for any new names.

        This operation is useful for explicit broadcasting by names (see examples).

        All of the dims of :attr:`self` must be named in order to use this method.
        The resulting tensor is a view on the original tensor.

        All dimension names of :attr:`self` must be present in ``other.names``.
        :attr:`other` may contain named dimensions that are not in ``self.names``;
        the output tensor has a size-one dimension for each of those new names.

        To align a tensor to a specific order, use :meth:`~Tensor.align_to`.

        Examples::

            # Example 1: Applying a mask
            >>> mask = torch.randint(2, [127, 128], dtype=torch.bool).refine_names('W', 'H')
            >>> imgs = torch.randn(32, 128, 127, 3, names=('N', 'H', 'W', 'C'))
            >>> imgs.masked_fill_(mask.align_as(imgs), 0)


            # Example 2: Applying a per-channel-scale
            >>> def scale_channels(input, scale):
            >>>    scale = scale.refine_names('C')
            >>>    return input * scale.align_as(input)

            >>> num_channels = 3
            >>> scale = torch.randn(num_channels, names=('C',))
            >>> imgs = torch.rand(32, 128, 128, num_channels, names=('N', 'H', 'W', 'C'))
            >>> more_imgs = torch.rand(32, num_channels, 128, 128, names=('N', 'C', 'H', 'W'))
            >>> videos = torch.randn(3, num_channels, 128, 128, 128, names=('N', 'C', 'H', 'W', 'D'))

            # scale_channels is agnostic to the dimension order of the input
            >>> scale_channels(imgs, scale)
            >>> scale_channels(more_imgs, scale)
            >>> scale_channels(videos, scale)

        .. warning::
            The named tensor API is experimental and subject to change.
        """
    @overload
    def align_to(self, order: Sequence[str | EllipsisType | None], ellipsis_idx: _int) -> Tensor: ...
    @overload
    def align_to(self, names: Sequence[str | EllipsisType | None]) -> Tensor: ...
    @overload
    def all(self) -> Tensor:
        """
        all(dim=None, keepdim=False) -> Tensor

        See :func:`torch.all`
        """
    @overload
    def all(self, dim: _size | None = ..., keepdim: _bool = ...) -> Tensor:
        """
        all(dim=None, keepdim=False) -> Tensor

        See :func:`torch.all`
        """
    @overload
    def all(self, dim: _int, keepdim: _bool = ...) -> Tensor:
        """
        all(dim=None, keepdim=False) -> Tensor

        See :func:`torch.all`
        """
    @overload
    def all(self, dim: str | EllipsisType | None, keepdim: _bool = ...) -> Tensor:
        """
        all(dim=None, keepdim=False) -> Tensor

        See :func:`torch.all`
        """
    def allclose(self, other: Tensor, rtol: _float = ..., atol: _float = ..., equal_nan: _bool = ...) -> _bool:
        """
        allclose(other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor

        See :func:`torch.allclose`
        """
    def amax(self, dim: _int | _size = ..., keepdim: _bool = ...) -> Tensor:
        """
        amax(dim=None, keepdim=False) -> Tensor

        See :func:`torch.amax`
        """
    def amin(self, dim: _int | _size = ..., keepdim: _bool = ...) -> Tensor:
        """
        amin(dim=None, keepdim=False) -> Tensor

        See :func:`torch.amin`
        """
    def aminmax(self, *, dim: _int | None = ..., keepdim: _bool = ...) -> torch.return_types.aminmax:
        """
        aminmax(*, dim=None, keepdim=False) -> (Tensor min, Tensor max)

        See :func:`torch.aminmax`
        """
    def angle(self) -> Tensor:
        """
        angle() -> Tensor

        See :func:`torch.angle`
        """
    @overload
    def any(self) -> Tensor:
        """
        any(dim=None, keepdim=False) -> Tensor

        See :func:`torch.any`
        """
    @overload
    def any(self, dim: _size | None = ..., keepdim: _bool = ...) -> Tensor:
        """
        any(dim=None, keepdim=False) -> Tensor

        See :func:`torch.any`
        """
    @overload
    def any(self, dim: _int, keepdim: _bool = ...) -> Tensor:
        """
        any(dim=None, keepdim=False) -> Tensor

        See :func:`torch.any`
        """
    @overload
    def any(self, dim: str | EllipsisType | None, keepdim: _bool = ...) -> Tensor:
        """
        any(dim=None, keepdim=False) -> Tensor

        See :func:`torch.any`
        """
    def apply_(self, callable: Callable) -> Tensor:
        """
        apply_(callable) -> Tensor

        Applies the function :attr:`callable` to each element in the tensor, replacing
        each element with the value returned by :attr:`callable`.

        .. note::

            This function only works with CPU tensors and should not be used in code
            sections that require high performance.
        """
    def arccos(self) -> Tensor:
        """
        arccos() -> Tensor

        See :func:`torch.arccos`
        """
    def arccos_(self) -> Tensor:
        """
        arccos_() -> Tensor

        In-place version of :meth:`~Tensor.arccos`
        """
    def arccosh(self) -> Tensor:
        """
        acosh() -> Tensor

        See :func:`torch.arccosh`
        """
    def arccosh_(self) -> Tensor:
        """
        acosh_() -> Tensor

        In-place version of :meth:`~Tensor.arccosh`
        """
    def arcsin(self) -> Tensor:
        """
        arcsin() -> Tensor

        See :func:`torch.arcsin`
        """
    def arcsin_(self) -> Tensor:
        """
        arcsin_() -> Tensor

        In-place version of :meth:`~Tensor.arcsin`
        """
    def arcsinh(self) -> Tensor:
        """
        arcsinh() -> Tensor

        See :func:`torch.arcsinh`
        """
    def arcsinh_(self) -> Tensor:
        """
        arcsinh_() -> Tensor

        In-place version of :meth:`~Tensor.arcsinh`
        """
    def arctan(self) -> Tensor:
        """
        arctan() -> Tensor

        See :func:`torch.arctan`
        """
    def arctan2(self, other: Tensor) -> Tensor:
        """
        arctan2(other) -> Tensor

        See :func:`torch.arctan2`
        """
    def arctan2_(self, other: Tensor) -> Tensor:
        """
        atan2_(other) -> Tensor

        In-place version of :meth:`~Tensor.arctan2`
        """
    def arctan_(self) -> Tensor:
        """
        arctan_() -> Tensor

        In-place version of :meth:`~Tensor.arctan`
        """
    def arctanh(self) -> Tensor:
        """
        arctanh() -> Tensor

        See :func:`torch.arctanh`
        """
    def arctanh_(self) -> Tensor:
        """
        arctanh_(other) -> Tensor

        In-place version of :meth:`~Tensor.arctanh`
        """
    def argmax(self, dim: _int | None = ..., keepdim: _bool = ...) -> Tensor:
        """
        argmax(dim=None, keepdim=False) -> LongTensor

        See :func:`torch.argmax`
        """
    def argmin(self, dim: _int | None = ..., keepdim: _bool = ...) -> Tensor:
        """
        argmin(dim=None, keepdim=False) -> LongTensor

        See :func:`torch.argmin`
        """
    @overload
    def argsort(self, *, stable: _bool, dim: _int = ..., descending: _bool = ...) -> Tensor:
        """
        argsort(dim=-1, descending=False) -> LongTensor

        See :func:`torch.argsort`
        """
    @overload
    def argsort(self, dim: _int = ..., descending: _bool = ...) -> Tensor:
        """
        argsort(dim=-1, descending=False) -> LongTensor

        See :func:`torch.argsort`
        """
    @overload
    def argsort(self, dim: str | EllipsisType | None, descending: _bool = ...) -> Tensor:
        """
        argsort(dim=-1, descending=False) -> LongTensor

        See :func:`torch.argsort`
        """
    def argwhere(self) -> Tensor:
        """
        argwhere() -> Tensor

        See :func:`torch.argwhere`
        """
    def as_strided(
        self, size: Sequence[_int | SymInt], stride: Sequence[_int | SymInt], storage_offset: _int | SymInt | None = ...
    ) -> Tensor:
        """
        as_strided(size, stride, storage_offset=None) -> Tensor

        See :func:`torch.as_strided`
        """
    def as_strided_(
        self, size: Sequence[_int | SymInt], stride: Sequence[_int | SymInt], storage_offset: _int | SymInt | None = ...
    ) -> Tensor:
        """
        as_strided_(size, stride, storage_offset=None) -> Tensor

        In-place version of :meth:`~Tensor.as_strided`
        """
    def as_strided_scatter(
        self,
        src: Tensor,
        size: Sequence[_int | SymInt],
        stride: Sequence[_int | SymInt],
        storage_offset: _int | SymInt | None = ...,
    ) -> Tensor:
        """
        as_strided_scatter(src, size, stride, storage_offset=None) -> Tensor

        See :func:`torch.as_strided_scatter`
        """
    def as_subclass(self, cls: type[S]) -> S:
        """
        as_subclass(cls) -> Tensor

        Makes a ``cls`` instance with the same data pointer as ``self``. Changes
        in the output mirror changes in ``self``, and the output stays attached
        to the autograd graph. ``cls`` must be a subclass of ``Tensor``.
        """
    def asin(self) -> Tensor:
        """
        asin() -> Tensor

        See :func:`torch.asin`
        """
    def asin_(self) -> Tensor:
        """
        asin_() -> Tensor

        In-place version of :meth:`~Tensor.asin`
        """
    def asinh(self) -> Tensor:
        """
        asinh() -> Tensor

        See :func:`torch.asinh`
        """
    def asinh_(self) -> Tensor:
        """
        asinh_() -> Tensor

        In-place version of :meth:`~Tensor.asinh`
        """
    def atan(self) -> Tensor:
        """
        atan() -> Tensor

        See :func:`torch.atan`
        """
    def atan2(self, other: Tensor) -> Tensor:
        """
        atan2(other) -> Tensor

        See :func:`torch.atan2`
        """
    def atan2_(self, other: Tensor) -> Tensor:
        """
        atan2_(other) -> Tensor

        In-place version of :meth:`~Tensor.atan2`
        """
    def atan_(self) -> Tensor:
        """
        atan_() -> Tensor

        In-place version of :meth:`~Tensor.atan`
        """
    def atanh(self) -> Tensor:
        """
        atanh() -> Tensor

        See :func:`torch.atanh`
        """
    def atanh_(self) -> Tensor:
        """
        atanh_(other) -> Tensor

        In-place version of :meth:`~Tensor.atanh`
        """
    def baddbmm(
        self, batch1: Tensor, batch2: Tensor, *, beta: Number | _complex = ..., alpha: Number | _complex = ...
    ) -> Tensor:
        """
        baddbmm(batch1, batch2, *, beta=1, alpha=1) -> Tensor

        See :func:`torch.baddbmm`
        """
    def baddbmm_(
        self, batch1: Tensor, batch2: Tensor, *, beta: Number | _complex = ..., alpha: Number | _complex = ...
    ) -> Tensor:
        """
        baddbmm_(batch1, batch2, *, beta=1, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.baddbmm`
        """
    @overload
    def bernoulli(self, *, generator: Generator | None = ...) -> Tensor:
        r"""
        bernoulli(*, generator=None) -> Tensor

        Returns a result tensor where each :math:`\texttt{result[i]}` is independently
        sampled from :math:`\text{Bernoulli}(\texttt{self[i]})`. :attr:`self` must have
        floating point ``dtype``, and the result will have the same ``dtype``.

        See :func:`torch.bernoulli`
        """
    @overload
    def bernoulli(self, p: _float, *, generator: Generator | None = ...) -> Tensor:
        r"""
        bernoulli(*, generator=None) -> Tensor

        Returns a result tensor where each :math:`\texttt{result[i]}` is independently
        sampled from :math:`\text{Bernoulli}(\texttt{self[i]})`. :attr:`self` must have
        floating point ``dtype``, and the result will have the same ``dtype``.

        See :func:`torch.bernoulli`
        """
    @overload
    def bernoulli_(self, p: Tensor, *, generator: Generator | None = ...) -> Tensor:
        r"""
        bernoulli_(p=0.5, *, generator=None) -> Tensor

        Fills each location of :attr:`self` with an independent sample from
        :math:`\text{Bernoulli}(\texttt{p})`. :attr:`self` can have integral
        ``dtype``.

        :attr:`p` should either be a scalar or tensor containing probabilities to be
        used for drawing the binary random number.

        If it is a tensor, the :math:`\text{i}^{th}` element of :attr:`self` tensor
        will be set to a value sampled from
        :math:`\text{Bernoulli}(\texttt{p\_tensor[i]})`. In this case `p` must have
        floating point ``dtype``.

        See also :meth:`~Tensor.bernoulli` and :func:`torch.bernoulli`
        """
    @overload
    def bernoulli_(self, p: _float = ..., *, generator: Generator | None = ...) -> Tensor:
        r"""
        bernoulli_(p=0.5, *, generator=None) -> Tensor

        Fills each location of :attr:`self` with an independent sample from
        :math:`\text{Bernoulli}(\texttt{p})`. :attr:`self` can have integral
        ``dtype``.

        :attr:`p` should either be a scalar or tensor containing probabilities to be
        used for drawing the binary random number.

        If it is a tensor, the :math:`\text{i}^{th}` element of :attr:`self` tensor
        will be set to a value sampled from
        :math:`\text{Bernoulli}(\texttt{p\_tensor[i]})`. In this case `p` must have
        floating point ``dtype``.

        See also :meth:`~Tensor.bernoulli` and :func:`torch.bernoulli`
        """
    def bfloat16(self) -> Tensor:
        """
        bfloat16(memory_format=torch.preserve_format) -> Tensor
        ``self.bfloat16()`` is equivalent to ``self.to(torch.bfloat16)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
    def bincount(self, weights: Tensor | None = ..., minlength: _int | SymInt = ...) -> Tensor:
        """
        bincount(weights=None, minlength=0) -> Tensor

        See :func:`torch.bincount`
        """
    @overload
    def bitwise_and(self, other: Tensor) -> Tensor:
        """
        bitwise_and() -> Tensor

        See :func:`torch.bitwise_and`
        """
    @overload
    def bitwise_and(self, other: Number | _complex) -> Tensor:
        """
        bitwise_and() -> Tensor

        See :func:`torch.bitwise_and`
        """
    @overload
    def bitwise_and_(self, other: Tensor) -> Tensor:
        """
        bitwise_and_() -> Tensor

        In-place version of :meth:`~Tensor.bitwise_and`
        """
    @overload
    def bitwise_and_(self, other: Number | _complex) -> Tensor:
        """
        bitwise_and_() -> Tensor

        In-place version of :meth:`~Tensor.bitwise_and`
        """
    @overload
    def bitwise_left_shift(self, other: Tensor) -> Tensor:
        """
        bitwise_left_shift(other) -> Tensor

        See :func:`torch.bitwise_left_shift`
        """
    @overload
    def bitwise_left_shift(self, other: Number | _complex) -> Tensor:
        """
        bitwise_left_shift(other) -> Tensor

        See :func:`torch.bitwise_left_shift`
        """
    @overload
    def bitwise_left_shift_(self, other: Tensor) -> Tensor:
        """
        bitwise_left_shift_(other) -> Tensor

        In-place version of :meth:`~Tensor.bitwise_left_shift`
        """
    @overload
    def bitwise_left_shift_(self, other: Number | _complex) -> Tensor:
        """
        bitwise_left_shift_(other) -> Tensor

        In-place version of :meth:`~Tensor.bitwise_left_shift`
        """
    def bitwise_not(self) -> Tensor:
        """
        bitwise_not() -> Tensor

        See :func:`torch.bitwise_not`
        """
    def bitwise_not_(self) -> Tensor:
        """
        bitwise_not_() -> Tensor

        In-place version of :meth:`~Tensor.bitwise_not`
        """
    @overload
    def bitwise_or(self, other: Tensor) -> Tensor:
        """
        bitwise_or() -> Tensor

        See :func:`torch.bitwise_or`
        """
    @overload
    def bitwise_or(self, other: Number | _complex) -> Tensor:
        """
        bitwise_or() -> Tensor

        See :func:`torch.bitwise_or`
        """
    @overload
    def bitwise_or_(self, other: Tensor) -> Tensor:
        """
        bitwise_or_() -> Tensor

        In-place version of :meth:`~Tensor.bitwise_or`
        """
    @overload
    def bitwise_or_(self, other: Number | _complex) -> Tensor:
        """
        bitwise_or_() -> Tensor

        In-place version of :meth:`~Tensor.bitwise_or`
        """
    @overload
    def bitwise_right_shift(self, other: Tensor) -> Tensor:
        """
        bitwise_right_shift(other) -> Tensor

        See :func:`torch.bitwise_right_shift`
        """
    @overload
    def bitwise_right_shift(self, other: Number | _complex) -> Tensor:
        """
        bitwise_right_shift(other) -> Tensor

        See :func:`torch.bitwise_right_shift`
        """
    @overload
    def bitwise_right_shift_(self, other: Tensor) -> Tensor:
        """
        bitwise_right_shift_(other) -> Tensor

        In-place version of :meth:`~Tensor.bitwise_right_shift`
        """
    @overload
    def bitwise_right_shift_(self, other: Number | _complex) -> Tensor:
        """
        bitwise_right_shift_(other) -> Tensor

        In-place version of :meth:`~Tensor.bitwise_right_shift`
        """
    @overload
    def bitwise_xor(self, other: Tensor) -> Tensor:
        """
        bitwise_xor() -> Tensor

        See :func:`torch.bitwise_xor`
        """
    @overload
    def bitwise_xor(self, other: Number | _complex) -> Tensor:
        """
        bitwise_xor() -> Tensor

        See :func:`torch.bitwise_xor`
        """
    @overload
    def bitwise_xor_(self, other: Tensor) -> Tensor:
        """
        bitwise_xor_() -> Tensor

        In-place version of :meth:`~Tensor.bitwise_xor`
        """
    @overload
    def bitwise_xor_(self, other: Number | _complex) -> Tensor:
        """
        bitwise_xor_() -> Tensor

        In-place version of :meth:`~Tensor.bitwise_xor`
        """
    def bmm(self, mat2: Tensor) -> Tensor:
        """
        bmm(batch2) -> Tensor

        See :func:`torch.bmm`
        """
    def bool(self) -> Tensor:
        """
        bool(memory_format=torch.preserve_format) -> Tensor

        ``self.bool()`` is equivalent to ``self.to(torch.bool)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
    @overload
    def broadcast_to(self, size: Sequence[_int | SymInt]) -> Tensor:
        """
        broadcast_to(shape) -> Tensor

        See :func:`torch.broadcast_to`.
        """
    @overload
    def broadcast_to(self, *size: _int | SymInt) -> Tensor:
        """
        broadcast_to(shape) -> Tensor

        See :func:`torch.broadcast_to`.
        """
    def byte(self) -> Tensor:
        """
        byte(memory_format=torch.preserve_format) -> Tensor

        ``self.byte()`` is equivalent to ``self.to(torch.uint8)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
    def cauchy_(self, median: _float = ..., sigma: _float = ..., *, generator: Generator | None = ...) -> Tensor:
        r"""
        cauchy_(median=0, sigma=1, *, generator=None) -> Tensor

        Fills the tensor with numbers drawn from the Cauchy distribution:

        .. math::

            f(x) = \dfrac{1}{\pi} \dfrac{\sigma}{(x - \text{median})^2 + \sigma^2}

        .. note::
          Sigma (:math:`\sigma`) is used to denote the scale parameter in Cauchy distribution.
        """
    def ccol_indices(self) -> Tensor: ...
    def ceil(self) -> Tensor:
        """
        ceil() -> Tensor

        See :func:`torch.ceil`
        """
    def ceil_(self) -> Tensor:
        """
        ceil_() -> Tensor

        In-place version of :meth:`~Tensor.ceil`
        """
    def chalf(self, *, memory_format: memory_format | None = ...) -> Tensor:
        """
        chalf(memory_format=torch.preserve_format) -> Tensor

        ``self.chalf()`` is equivalent to ``self.to(torch.complex32)``. See :func:`to`.

        Args:
             memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.

        """
    def char(self) -> Tensor:
        """
        char(memory_format=torch.preserve_format) -> Tensor

        ``self.char()`` is equivalent to ``self.to(torch.int8)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
    def cholesky(self, upper: _bool = ...) -> Tensor:
        """
        cholesky(upper=False) -> Tensor

        See :func:`torch.cholesky`
        """
    def cholesky_inverse(self, upper: _bool = ...) -> Tensor:
        """
        cholesky_inverse(upper=False) -> Tensor

        See :func:`torch.cholesky_inverse`
        """
    def cholesky_solve(self, input2: Tensor, upper: _bool = ...) -> Tensor:
        """
        cholesky_solve(input2, upper=False) -> Tensor

        See :func:`torch.cholesky_solve`
        """
    def chunk(self, chunks: _int, dim: _int = ...) -> tuple[Tensor, ...]:
        """
        chunk(chunks, dim=0) -> List of Tensors

        See :func:`torch.chunk`
        """
    @overload
    def clamp(self, min: Tensor | None = ..., max: Tensor | None = ...) -> Tensor:
        """
        clamp(min=None, max=None) -> Tensor

        See :func:`torch.clamp`
        """
    @overload
    def clamp(self, min: Number | _complex | None = ..., max: Number | _complex | None = ...) -> Tensor:
        """
        clamp(min=None, max=None) -> Tensor

        See :func:`torch.clamp`
        """
    @overload
    def clamp_(self, min: Tensor | None = ..., max: Tensor | None = ...) -> Tensor:
        """
        clamp_(min=None, max=None) -> Tensor

        In-place version of :meth:`~Tensor.clamp`
        """
    @overload
    def clamp_(self, min: Number | _complex | None = ..., max: Number | _complex | None = ...) -> Tensor:
        """
        clamp_(min=None, max=None) -> Tensor

        In-place version of :meth:`~Tensor.clamp`
        """
    @overload
    def clamp_max(self, max: Tensor) -> Tensor: ...
    @overload
    def clamp_max(self, max: Number | _complex) -> Tensor: ...
    @overload
    def clamp_max_(self, max: Tensor) -> Tensor: ...
    @overload
    def clamp_max_(self, max: Number | _complex) -> Tensor: ...
    @overload
    def clamp_min(self, min: Tensor) -> Tensor: ...
    @overload
    def clamp_min(self, min: Number | _complex) -> Tensor: ...
    @overload
    def clamp_min_(self, min: Tensor) -> Tensor: ...
    @overload
    def clamp_min_(self, min: Number | _complex) -> Tensor: ...
    @overload
    def clip(self, min: Tensor | None = ..., max: Tensor | None = ...) -> Tensor:
        """
        clip(min=None, max=None) -> Tensor

        Alias for :meth:`~Tensor.clamp`.
        """
    @overload
    def clip(self, min: Number | _complex | None = ..., max: Number | _complex | None = ...) -> Tensor:
        """
        clip(min=None, max=None) -> Tensor

        Alias for :meth:`~Tensor.clamp`.
        """
    @overload
    def clip_(self, min: Tensor | None = ..., max: Tensor | None = ...) -> Tensor:
        """
        clip_(min=None, max=None) -> Tensor

        Alias for :meth:`~Tensor.clamp_`.
        """
    @overload
    def clip_(self, min: Number | _complex | None = ..., max: Number | _complex | None = ...) -> Tensor:
        """
        clip_(min=None, max=None) -> Tensor

        Alias for :meth:`~Tensor.clamp_`.
        """
    def clone(self, *, memory_format: memory_format | None = ...) -> Tensor:
        """
        clone(*, memory_format=torch.preserve_format) -> Tensor

        See :func:`torch.clone`
        """
    def coalesce(self) -> Tensor:
        """
        coalesce() -> Tensor

        Returns a coalesced copy of :attr:`self` if :attr:`self` is an
        :ref:`uncoalesced tensor <sparse-uncoalesced-coo-docs>`.

        Returns :attr:`self` if :attr:`self` is a coalesced tensor.

        .. warning::
          Throws an error if :attr:`self` is not a sparse COO tensor.
        """
    def col_indices(self) -> Tensor:
        """
        col_indices() -> IntTensor

        Returns the tensor containing the column indices of the :attr:`self`
        tensor when :attr:`self` is a sparse CSR tensor of layout ``sparse_csr``.
        The ``col_indices`` tensor is strictly of shape (:attr:`self`.nnz())
        and of type ``int32`` or ``int64``.  When using MKL routines such as sparse
        matrix multiplication, it is necessary to use ``int32`` indexing in order
        to avoid downcasting and potentially losing information.

        Example::

            >>> csr = torch.eye(5,5).to_sparse_csr()
            >>> csr.col_indices()
            tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        """
    def conj(self) -> Tensor:
        """
        conj() -> Tensor

        See :func:`torch.conj`
        """
    def conj_physical(self) -> Tensor:
        """
        conj_physical() -> Tensor

        See :func:`torch.conj_physical`
        """
    def conj_physical_(self) -> Tensor:
        """
        conj_physical_() -> Tensor

        In-place version of :meth:`~Tensor.conj_physical`
        """
    def contiguous(self, memory_format: torch.memory_format = ...) -> Tensor:
        """
        contiguous(memory_format=torch.contiguous_format) -> Tensor

        Returns a contiguous in memory tensor containing the same data as :attr:`self` tensor. If
        :attr:`self` tensor is already in the specified memory format, this function returns the
        :attr:`self` tensor.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.contiguous_format``.
        """
    def copy_(self, other: Tensor, non_blocking: _bool = ...) -> Tensor:
        """
        copy_(src, non_blocking=False) -> Tensor

        Copies the elements from :attr:`src` into :attr:`self` tensor and returns
        :attr:`self`.

        The :attr:`src` tensor must be :ref:`broadcastable <broadcasting-semantics>`
        with the :attr:`self` tensor. It may be of a different data type or reside on a
        different device.

        Args:
            src (Tensor): the source tensor to copy from
            non_blocking (bool, optional): if ``True`` and this copy is between CPU and GPU,
                the copy may occur asynchronously with respect to the host. For other
                cases, this argument has no effect. Default: ``False``
        """
    @overload
    def copysign(self, other: Tensor) -> Tensor:
        """
        copysign(other) -> Tensor

        See :func:`torch.copysign`
        """
    @overload
    def copysign(self, other: Number | _complex) -> Tensor:
        """
        copysign(other) -> Tensor

        See :func:`torch.copysign`
        """
    @overload
    def copysign_(self, other: Tensor) -> Tensor:
        """
        copysign_(other) -> Tensor

        In-place version of :meth:`~Tensor.copysign`
        """
    @overload
    def copysign_(self, other: Number | _complex) -> Tensor:
        """
        copysign_(other) -> Tensor

        In-place version of :meth:`~Tensor.copysign`
        """
    def corrcoef(self) -> Tensor:
        """
        corrcoef() -> Tensor

        See :func:`torch.corrcoef`
        """
    def cos(self) -> Tensor:
        """
        cos() -> Tensor

        See :func:`torch.cos`
        """
    def cos_(self) -> Tensor:
        """
        cos_() -> Tensor

        In-place version of :meth:`~Tensor.cos`
        """
    def cosh(self) -> Tensor:
        """
        cosh() -> Tensor

        See :func:`torch.cosh`
        """
    def cosh_(self) -> Tensor:
        """
        cosh_() -> Tensor

        In-place version of :meth:`~Tensor.cosh`
        """
    @overload
    def count_nonzero(self, dim: _int | None = ...) -> Tensor:
        """
        count_nonzero(dim=None) -> Tensor

        See :func:`torch.count_nonzero`
        """
    @overload
    def count_nonzero(self, dim: _size) -> Tensor:
        """
        count_nonzero(dim=None) -> Tensor

        See :func:`torch.count_nonzero`
        """
    @overload
    def count_nonzero(self, *dim: _int) -> Tensor:
        """
        count_nonzero(dim=None) -> Tensor

        See :func:`torch.count_nonzero`
        """
    def cov(self, *, correction: _int = ..., fweights: Tensor | None = ..., aweights: Tensor | None = ...) -> Tensor:
        """
        cov(*, correction=1, fweights=None, aweights=None) -> Tensor

        See :func:`torch.cov`
        """
    def cpu(self, memory_format: torch.memory_format = ...) -> Tensor:
        """
        cpu(memory_format=torch.preserve_format) -> Tensor

        Returns a copy of this object in CPU memory.

        If this object is already in CPU memory,
        then no copy is performed and the original object is returned.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
    def cross(self, other: Tensor, dim: _int | None = ...) -> Tensor:
        """
        cross(other, dim=None) -> Tensor

        See :func:`torch.cross`
        """
    def crow_indices(self) -> Tensor:
        """
        crow_indices() -> IntTensor

        Returns the tensor containing the compressed row indices of the :attr:`self`
        tensor when :attr:`self` is a sparse CSR tensor of layout ``sparse_csr``.
        The ``crow_indices`` tensor is strictly of shape (:attr:`self`.size(0) + 1)
        and of type ``int32`` or ``int64``. When using MKL routines such as sparse
        matrix multiplication, it is necessary to use ``int32`` indexing in order
        to avoid downcasting and potentially losing information.

        Example::

            >>> csr = torch.eye(5,5).to_sparse_csr()
            >>> csr.crow_indices()
            tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
        """
    def cuda(
        self,
        device: _device | _int | str | None = ...,
        non_blocking: _bool = ...,
        memory_format: torch.memory_format = ...,
    ) -> Tensor:
        """
        cuda(device=None, non_blocking=False, memory_format=torch.preserve_format) -> Tensor

        Returns a copy of this object in CUDA memory.

        If this object is already in CUDA memory and on the correct device,
        then no copy is performed and the original object is returned.

        Args:
            device (:class:`torch.device`, optional): The destination GPU device.
                Defaults to the current CUDA device.
            non_blocking (bool, optional): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host.
                Otherwise, the argument has no effect. Default: ``False``.
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
    @overload
    def cummax(self, dim: _int) -> torch.return_types.cummax:
        """
        cummax(dim) -> (Tensor, Tensor)

        See :func:`torch.cummax`
        """
    @overload
    def cummax(self, dim: str | EllipsisType | None) -> torch.return_types.cummax:
        """
        cummax(dim) -> (Tensor, Tensor)

        See :func:`torch.cummax`
        """
    @overload
    def cummin(self, dim: _int) -> torch.return_types.cummin:
        """
        cummin(dim) -> (Tensor, Tensor)

        See :func:`torch.cummin`
        """
    @overload
    def cummin(self, dim: str | EllipsisType | None) -> torch.return_types.cummin:
        """
        cummin(dim) -> (Tensor, Tensor)

        See :func:`torch.cummin`
        """
    @overload
    def cumprod(self, dim: _int, *, dtype: _dtype | None = ...) -> Tensor:
        """
        cumprod(dim, dtype=None) -> Tensor

        See :func:`torch.cumprod`
        """
    @overload
    def cumprod(self, dim: str | EllipsisType | None, *, dtype: _dtype | None = ...) -> Tensor:
        """
        cumprod(dim, dtype=None) -> Tensor

        See :func:`torch.cumprod`
        """
    @overload
    def cumprod_(self, dim: _int, *, dtype: _dtype | None = ...) -> Tensor:
        """
        cumprod_(dim, dtype=None) -> Tensor

        In-place version of :meth:`~Tensor.cumprod`
        """
    @overload
    def cumprod_(self, dim: str | EllipsisType | None, *, dtype: _dtype | None = ...) -> Tensor:
        """
        cumprod_(dim, dtype=None) -> Tensor

        In-place version of :meth:`~Tensor.cumprod`
        """
    @overload
    def cumsum(self, dim: _int, *, dtype: _dtype | None = ...) -> Tensor:
        """
        cumsum(dim, dtype=None) -> Tensor

        See :func:`torch.cumsum`
        """
    @overload
    def cumsum(self, dim: str | EllipsisType | None, *, dtype: _dtype | None = ...) -> Tensor:
        """
        cumsum(dim, dtype=None) -> Tensor

        See :func:`torch.cumsum`
        """
    @overload
    def cumsum_(self, dim: _int, *, dtype: _dtype | None = ...) -> Tensor:
        """
        cumsum_(dim, dtype=None) -> Tensor

        In-place version of :meth:`~Tensor.cumsum`
        """
    @overload
    def cumsum_(self, dim: str | EllipsisType | None, *, dtype: _dtype | None = ...) -> Tensor:
        """
        cumsum_(dim, dtype=None) -> Tensor

        In-place version of :meth:`~Tensor.cumsum`
        """
    def data_ptr(self) -> _int:
        """
        data_ptr() -> int

        Returns the address of the first element of :attr:`self` tensor.
        """
    def deg2rad(self) -> Tensor:
        """
        deg2rad() -> Tensor

        See :func:`torch.deg2rad`
        """
    def deg2rad_(self) -> Tensor:
        """
        deg2rad_() -> Tensor

        In-place version of :meth:`~Tensor.deg2rad`
        """
    def dense_dim(self) -> _int:
        """
        dense_dim() -> int

        Return the number of dense dimensions in a :ref:`sparse tensor <sparse-docs>` :attr:`self`.

        .. note::
          Returns ``len(self.shape)`` if :attr:`self` is not a sparse tensor.

        See also :meth:`Tensor.sparse_dim` and :ref:`hybrid tensors <sparse-hybrid-coo-docs>`.
        """
    def dequantize(self) -> Tensor:
        """
        dequantize() -> Tensor

        Given a quantized Tensor, dequantize it and return the dequantized float Tensor.
        """
    def det(self) -> Tensor:
        """
        det() -> Tensor

        See :func:`torch.det`
        """
    def detach(self) -> Tensor:
        """
        Returns a new Tensor, detached from the current graph.

        The result will never require gradient.

        This method also affects forward mode AD gradients and the result will never
        have forward mode AD gradients.

        .. note::

          Returned Tensor shares the same storage with the original one.
          In-place modifications on either of them will be seen, and may trigger
          errors in correctness checks.
        """
    def detach_(self) -> Tensor:
        """
        Detaches the Tensor from the graph that created it, making it a leaf.
        Views cannot be detached in-place.

        This method also affects forward mode AD gradients and the result will never
        have forward mode AD gradients.
        """
    def diag(self, diagonal: _int = ...) -> Tensor:
        """
        diag(diagonal=0) -> Tensor

        See :func:`torch.diag`
        """
    def diag_embed(self, offset: _int = ..., dim1: _int = ..., dim2: _int = ...) -> Tensor:
        """
        diag_embed(offset=0, dim1=-2, dim2=-1) -> Tensor

        See :func:`torch.diag_embed`
        """
    def diagflat(self, offset: _int = ...) -> Tensor:
        """
        diagflat(offset=0) -> Tensor

        See :func:`torch.diagflat`
        """
    @overload
    def diagonal(
        self,
        *,
        outdim: str | EllipsisType | None,
        dim1: str | EllipsisType | None,
        dim2: str | EllipsisType | None,
        offset: _int = ...,
    ) -> Tensor:
        """
        diagonal(offset=0, dim1=0, dim2=1) -> Tensor

        See :func:`torch.diagonal`
        """
    @overload
    def diagonal(self, offset: _int = ..., dim1: _int = ..., dim2: _int = ...) -> Tensor:
        """
        diagonal(offset=0, dim1=0, dim2=1) -> Tensor

        See :func:`torch.diagonal`
        """
    def diagonal_scatter(self, src: Tensor, offset: _int = ..., dim1: _int = ..., dim2: _int = ...) -> Tensor:
        """
        diagonal_scatter(src, offset=0, dim1=0, dim2=1) -> Tensor

        See :func:`torch.diagonal_scatter`
        """
    def diff(self, n: _int = ..., dim: _int = ..., prepend: Tensor | None = ..., append: Tensor | None = ...) -> Tensor:
        """
        diff(n=1, dim=-1, prepend=None, append=None) -> Tensor

        See :func:`torch.diff`
        """
    def digamma(self) -> Tensor:
        """
        digamma() -> Tensor

        See :func:`torch.digamma`
        """
    def digamma_(self) -> Tensor:
        """
        digamma_() -> Tensor

        In-place version of :meth:`~Tensor.digamma`
        """
    def dim(self) -> _int:
        """
        dim() -> int

        Returns the number of dimensions of :attr:`self` tensor.
        """
    def dist(self, other: Tensor, p: Number | _complex = ...) -> Tensor:
        """
        dist(other, p=2) -> Tensor

        See :func:`torch.dist`
        """
    def div(self, other: Tensor | Number, *, rounding_mode: str | None = ...) -> Tensor:
        """
        div(value, *, rounding_mode=None) -> Tensor

        See :func:`torch.div`
        """
    def div_(self, other: Tensor | Number, *, rounding_mode: str | None = ...) -> Tensor:
        """
        div_(value, *, rounding_mode=None) -> Tensor

        In-place version of :meth:`~Tensor.div`
        """
    @overload
    def divide(self, other: Tensor) -> Tensor:
        """
        divide(value, *, rounding_mode=None) -> Tensor

        See :func:`torch.divide`
        """
    @overload
    def divide(self, other: Tensor, *, rounding_mode: str | None) -> Tensor:
        """
        divide(value, *, rounding_mode=None) -> Tensor

        See :func:`torch.divide`
        """
    @overload
    def divide(self, other: Number | _complex, *, rounding_mode: str | None) -> Tensor:
        """
        divide(value, *, rounding_mode=None) -> Tensor

        See :func:`torch.divide`
        """
    @overload
    def divide(self, other: Number | _complex) -> Tensor:
        """
        divide(value, *, rounding_mode=None) -> Tensor

        See :func:`torch.divide`
        """
    @overload
    def divide_(self, other: Tensor) -> Tensor:
        """
        divide_(value, *, rounding_mode=None) -> Tensor

        In-place version of :meth:`~Tensor.divide`
        """
    @overload
    def divide_(self, other: Tensor, *, rounding_mode: str | None) -> Tensor:
        """
        divide_(value, *, rounding_mode=None) -> Tensor

        In-place version of :meth:`~Tensor.divide`
        """
    @overload
    def divide_(self, other: Number | _complex, *, rounding_mode: str | None) -> Tensor:
        """
        divide_(value, *, rounding_mode=None) -> Tensor

        In-place version of :meth:`~Tensor.divide`
        """
    @overload
    def divide_(self, other: Number | _complex) -> Tensor:
        """
        divide_(value, *, rounding_mode=None) -> Tensor

        In-place version of :meth:`~Tensor.divide`
        """
    def dot(self, tensor: Tensor) -> Tensor:
        """
        dot(other) -> Tensor

        See :func:`torch.dot`
        """
    def double(self) -> Tensor:
        """
        double(memory_format=torch.preserve_format) -> Tensor

        ``self.double()`` is equivalent to ``self.to(torch.float64)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
    @overload
    def dsplit(self, sections: _int) -> tuple[Tensor, ...]:
        """
        dsplit(split_size_or_sections) -> List of Tensors

        See :func:`torch.dsplit`
        """
    @overload
    def dsplit(self, indices: _size) -> tuple[Tensor, ...]:
        """
        dsplit(split_size_or_sections) -> List of Tensors

        See :func:`torch.dsplit`
        """
    @overload
    def dsplit(self, *indices: _int) -> tuple[Tensor, ...]:
        """
        dsplit(split_size_or_sections) -> List of Tensors

        See :func:`torch.dsplit`
        """
    def element_size(self) -> _int:
        """
        element_size() -> int

        Returns the size in bytes of an individual element.

        Example::

            >>> torch.tensor([]).element_size()
            4
            >>> torch.tensor([], dtype=torch.uint8).element_size()
            1
        """
    @overload
    def eq(self, other: Tensor) -> Tensor:
        """
        eq(other) -> Tensor

        See :func:`torch.eq`
        """
    @overload
    def eq(self, other: Number | _complex) -> Tensor:
        """
        eq(other) -> Tensor

        See :func:`torch.eq`
        """
    @overload
    def eq_(self, other: Tensor) -> Tensor:
        """
        eq_(other) -> Tensor

        In-place version of :meth:`~Tensor.eq`
        """
    @overload
    def eq_(self, other: Number | _complex) -> Tensor:
        """
        eq_(other) -> Tensor

        In-place version of :meth:`~Tensor.eq`
        """
    def equal(self, other: Tensor) -> _bool:
        """
        equal(other) -> bool

        See :func:`torch.equal`
        """
    def erf(self) -> Tensor:
        """
        erf() -> Tensor

        See :func:`torch.erf`
        """
    def erf_(self) -> Tensor:
        """
        erf_() -> Tensor

        In-place version of :meth:`~Tensor.erf`
        """
    def erfc(self) -> Tensor:
        """
        erfc() -> Tensor

        See :func:`torch.erfc`
        """
    def erfc_(self) -> Tensor:
        """
        erfc_() -> Tensor

        In-place version of :meth:`~Tensor.erfc`
        """
    def erfinv(self) -> Tensor:
        """
        erfinv() -> Tensor

        See :func:`torch.erfinv`
        """
    def erfinv_(self) -> Tensor:
        """
        erfinv_() -> Tensor

        In-place version of :meth:`~Tensor.erfinv`
        """
    def exp(self) -> Tensor:
        """
        exp() -> Tensor

        See :func:`torch.exp`
        """
    def exp2(self) -> Tensor:
        """
        exp2() -> Tensor

        See :func:`torch.exp2`
        """
    def exp2_(self) -> Tensor:
        """
        exp2_() -> Tensor

        In-place version of :meth:`~Tensor.exp2`
        """
    def exp_(self) -> Tensor:
        """
        exp_() -> Tensor

        In-place version of :meth:`~Tensor.exp`
        """
    @overload
    def expand(self, size: Sequence[_int | SymInt], *, implicit: _bool = ...) -> Tensor:
        """
        expand(*sizes) -> Tensor

        Returns a new view of the :attr:`self` tensor with singleton dimensions expanded
        to a larger size.

        Passing -1 as the size for a dimension means not changing the size of
        that dimension.

        Tensor can be also expanded to a larger number of dimensions, and the
        new ones will be appended at the front. For the new dimensions, the
        size cannot be set to -1.

        Expanding a tensor does not allocate new memory, but only creates a
        new view on the existing tensor where a dimension of size one is
        expanded to a larger size by setting the ``stride`` to 0. Any dimension
        of size 1 can be expanded to an arbitrary value without allocating new
        memory.

        Args:
            *sizes (torch.Size or int...): the desired expanded size

        .. warning::

            More than one element of an expanded tensor may refer to a single
            memory location. As a result, in-place operations (especially ones that
            are vectorized) may result in incorrect behavior. If you need to write
            to the tensors, please clone them first.

        Example::

            >>> x = torch.tensor([[1], [2], [3]])
            >>> x.size()
            torch.Size([3, 1])
            >>> x.expand(3, 4)
            tensor([[ 1,  1,  1,  1],
                    [ 2,  2,  2,  2],
                    [ 3,  3,  3,  3]])
            >>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
            tensor([[ 1,  1,  1,  1],
                    [ 2,  2,  2,  2],
                    [ 3,  3,  3,  3]])
        """
    @overload
    def expand(self, *size: _int | SymInt, implicit: _bool = ...) -> Tensor:
        """
        expand(*sizes) -> Tensor

        Returns a new view of the :attr:`self` tensor with singleton dimensions expanded
        to a larger size.

        Passing -1 as the size for a dimension means not changing the size of
        that dimension.

        Tensor can be also expanded to a larger number of dimensions, and the
        new ones will be appended at the front. For the new dimensions, the
        size cannot be set to -1.

        Expanding a tensor does not allocate new memory, but only creates a
        new view on the existing tensor where a dimension of size one is
        expanded to a larger size by setting the ``stride`` to 0. Any dimension
        of size 1 can be expanded to an arbitrary value without allocating new
        memory.

        Args:
            *sizes (torch.Size or int...): the desired expanded size

        .. warning::

            More than one element of an expanded tensor may refer to a single
            memory location. As a result, in-place operations (especially ones that
            are vectorized) may result in incorrect behavior. If you need to write
            to the tensors, please clone them first.

        Example::

            >>> x = torch.tensor([[1], [2], [3]])
            >>> x.size()
            torch.Size([3, 1])
            >>> x.expand(3, 4)
            tensor([[ 1,  1,  1,  1],
                    [ 2,  2,  2,  2],
                    [ 3,  3,  3,  3]])
            >>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
            tensor([[ 1,  1,  1,  1],
                    [ 2,  2,  2,  2],
                    [ 3,  3,  3,  3]])
        """
    def expand_as(self, other: Tensor) -> Tensor:
        """
        expand_as(other) -> Tensor

        Expand this tensor to the same size as :attr:`other`.
        ``self.expand_as(other)`` is equivalent to ``self.expand(other.size())``.

        Please see :meth:`~Tensor.expand` for more information about ``expand``.

        Args:
            other (:class:`torch.Tensor`): The result tensor has the same size
                as :attr:`other`.
        """
    def expm1(self) -> Tensor:
        """
        expm1() -> Tensor

        See :func:`torch.expm1`
        """
    def expm1_(self) -> Tensor:
        """
        expm1_() -> Tensor

        In-place version of :meth:`~Tensor.expm1`
        """
    def exponential_(self, lambd: _float = ..., *, generator: Generator | None = ...) -> Tensor:
        r"""
        exponential_(lambd=1, *, generator=None) -> Tensor

        Fills :attr:`self` tensor with elements drawn from the PDF (probability density function):

        .. math::

            f(x) = \lambda e^{-\lambda x}, x > 0

        .. note::
          In probability theory, exponential distribution is supported on interval [0, :math:`\inf`) (i.e., :math:`x >= 0`)
          implying that zero can be sampled from the exponential distribution.
          However, :func:`torch.Tensor.exponential_` does not sample zero,
          which means that its actual support is the interval (0, :math:`\inf`).

          Note that :func:`torch.distributions.exponential.Exponential` is supported on the interval [0, :math:`\inf`) and can sample zero.
        """
    @overload
    def fill_(self, value: Tensor) -> Tensor:
        """
        fill_(value) -> Tensor

        Fills :attr:`self` tensor with the specified value.
        """
    @overload
    def fill_(self, value: Number | _complex) -> Tensor:
        """
        fill_(value) -> Tensor

        Fills :attr:`self` tensor with the specified value.
        """
    def fill_diagonal_(self, fill_value: Number | _complex, wrap: _bool = ...) -> Tensor:
        """
        fill_diagonal_(fill_value, wrap=False) -> Tensor

        Fill the main diagonal of a tensor that has at least 2-dimensions.
        When dims>2, all dimensions of input must be of equal length.
        This function modifies the input tensor in-place, and returns the input tensor.

        Arguments:
            fill_value (Scalar): the fill value
            wrap (bool, optional): the diagonal 'wrapped' after N columns for tall matrices. Default: ``False``

        Example::

            >>> a = torch.zeros(3, 3)
            >>> a.fill_diagonal_(5)
            tensor([[5., 0., 0.],
                    [0., 5., 0.],
                    [0., 0., 5.]])
            >>> b = torch.zeros(7, 3)
            >>> b.fill_diagonal_(5)
            tensor([[5., 0., 0.],
                    [0., 5., 0.],
                    [0., 0., 5.],
                    [0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]])
            >>> c = torch.zeros(7, 3)
            >>> c.fill_diagonal_(5, wrap=True)
            tensor([[5., 0., 0.],
                    [0., 5., 0.],
                    [0., 0., 5.],
                    [0., 0., 0.],
                    [5., 0., 0.],
                    [0., 5., 0.],
                    [0., 0., 5.]])
        """
    def fix(self) -> Tensor:
        """
        fix() -> Tensor

        See :func:`torch.fix`.
        """
    def fix_(self) -> Tensor:
        """
        fix_() -> Tensor

        In-place version of :meth:`~Tensor.fix`
        """
    @overload
    def flatten(self, start_dim: _int, end_dim: _int, out_dim: str | EllipsisType | None) -> Tensor:
        """
        flatten(start_dim=0, end_dim=-1) -> Tensor

        See :func:`torch.flatten`
        """
    @overload
    def flatten(self, start_dim: _int = ..., end_dim: _int = ...) -> Tensor:
        """
        flatten(start_dim=0, end_dim=-1) -> Tensor

        See :func:`torch.flatten`
        """
    @overload
    def flatten(
        self,
        start_dim: str | EllipsisType | None,
        end_dim: str | EllipsisType | None,
        out_dim: str | EllipsisType | None,
    ) -> Tensor:
        """
        flatten(start_dim=0, end_dim=-1) -> Tensor

        See :func:`torch.flatten`
        """
    @overload
    def flatten(self, dims: Sequence[str | EllipsisType | None], out_dim: str | EllipsisType | None) -> Tensor:
        """
        flatten(start_dim=0, end_dim=-1) -> Tensor

        See :func:`torch.flatten`
        """
    @overload
    def flip(self, dims: _size) -> Tensor:
        """
        flip(dims) -> Tensor

        See :func:`torch.flip`
        """
    @overload
    def flip(self, *dims: _int) -> Tensor:
        """
        flip(dims) -> Tensor

        See :func:`torch.flip`
        """
    def fliplr(self) -> Tensor:
        """
        fliplr() -> Tensor

        See :func:`torch.fliplr`
        """
    def flipud(self) -> Tensor:
        """
        flipud() -> Tensor

        See :func:`torch.flipud`
        """
    def float(self) -> Tensor:
        """
        float(memory_format=torch.preserve_format) -> Tensor

        ``self.float()`` is equivalent to ``self.to(torch.float32)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
    @overload
    def float_power(self, exponent: Tensor) -> Tensor:
        """
        float_power(exponent) -> Tensor

        See :func:`torch.float_power`
        """
    @overload
    def float_power(self, exponent: Number | _complex) -> Tensor:
        """
        float_power(exponent) -> Tensor

        See :func:`torch.float_power`
        """
    @overload
    def float_power_(self, exponent: Tensor) -> Tensor:
        """
        float_power_(exponent) -> Tensor

        In-place version of :meth:`~Tensor.float_power`
        """
    @overload
    def float_power_(self, exponent: Number | _complex) -> Tensor:
        """
        float_power_(exponent) -> Tensor

        In-place version of :meth:`~Tensor.float_power`
        """
    def floor(self) -> Tensor:
        """
        floor() -> Tensor

        See :func:`torch.floor`
        """
    def floor_(self) -> Tensor:
        """
        floor_() -> Tensor

        In-place version of :meth:`~Tensor.floor`
        """
    def floor_divide(
        self, other: Tensor | Number | torch.SymInt | torch.SymFloat, *, out: Tensor | None = ...
    ) -> Tensor:
        """
        floor_divide(value) -> Tensor

        See :func:`torch.floor_divide`
        """
    def floor_divide_(self, other: Tensor | Number | torch.SymInt | torch.SymFloat) -> Tensor:
        """
        floor_divide_(value) -> Tensor

        In-place version of :meth:`~Tensor.floor_divide`
        """
    def fmax(self, other: Tensor) -> Tensor:
        """
        fmax(other) -> Tensor

        See :func:`torch.fmax`
        """
    def fmin(self, other: Tensor) -> Tensor:
        """
        fmin(other) -> Tensor

        See :func:`torch.fmin`
        """
    @overload
    def fmod(self, other: Tensor) -> Tensor:
        """
        fmod(divisor) -> Tensor

        See :func:`torch.fmod`
        """
    @overload
    def fmod(self, other: Number | _complex) -> Tensor:
        """
        fmod(divisor) -> Tensor

        See :func:`torch.fmod`
        """
    @overload
    def fmod_(self, other: Tensor) -> Tensor:
        """
        fmod_(divisor) -> Tensor

        In-place version of :meth:`~Tensor.fmod`
        """
    @overload
    def fmod_(self, other: Number | _complex) -> Tensor:
        """
        fmod_(divisor) -> Tensor

        In-place version of :meth:`~Tensor.fmod`
        """
    def frac(self) -> Tensor:
        """
        frac() -> Tensor

        See :func:`torch.frac`
        """
    def frac_(self) -> Tensor:
        """
        frac_() -> Tensor

        In-place version of :meth:`~Tensor.frac`
        """
    def frexp(self) -> torch.return_types.frexp:
        """
        frexp(input) -> (Tensor mantissa, Tensor exponent)

        See :func:`torch.frexp`
        """
    @overload
    def gather(self, dim: _int, index: Tensor, *, sparse_grad: _bool = ...) -> Tensor:
        """
        gather(dim, index) -> Tensor

        See :func:`torch.gather`
        """
    @overload
    def gather(self, dim: str | EllipsisType | None, index: Tensor, *, sparse_grad: _bool = ...) -> Tensor:
        """
        gather(dim, index) -> Tensor

        See :func:`torch.gather`
        """
    def gcd(self, other: Tensor) -> Tensor:
        """
        gcd(other) -> Tensor

        See :func:`torch.gcd`
        """
    def gcd_(self, other: Tensor) -> Tensor:
        """
        gcd_(other) -> Tensor

        In-place version of :meth:`~Tensor.gcd`
        """
    @overload
    def ge(self, other: Tensor) -> Tensor:
        """
        ge(other) -> Tensor

        See :func:`torch.ge`.
        """
    @overload
    def ge(self, other: Number | _complex) -> Tensor:
        """
        ge(other) -> Tensor

        See :func:`torch.ge`.
        """
    @overload
    def ge_(self, other: Tensor) -> Tensor:
        """
        ge_(other) -> Tensor

        In-place version of :meth:`~Tensor.ge`.
        """
    @overload
    def ge_(self, other: Number | _complex) -> Tensor:
        """
        ge_(other) -> Tensor

        In-place version of :meth:`~Tensor.ge`.
        """
    def geometric_(self, p: _float, *, generator: Generator | None = ...) -> Tensor:
        r"""
        geometric_(p, *, generator=None) -> Tensor

        Fills :attr:`self` tensor with elements drawn from the geometric distribution:

        .. math::

            P(X=k) = (1 - p)^{k - 1} p, k = 1, 2, ...

        .. note::
          :func:`torch.Tensor.geometric_` `k`-th trial is the first success hence draws samples in :math:`\{1, 2, \ldots\}`, whereas
          :func:`torch.distributions.geometric.Geometric` :math:`(k+1)`-th trial is the first success
          hence draws samples in :math:`\{0, 1, \ldots\}`.
        """
    def geqrf(self) -> torch.return_types.geqrf:
        """
        geqrf() -> (Tensor, Tensor)

        See :func:`torch.geqrf`
        """
    def ger(self, vec2: Tensor) -> Tensor:
        """
        ger(vec2) -> Tensor

        See :func:`torch.ger`
        """
    def get_device(self) -> _int:
        """
        get_device() -> Device ordinal (Integer)

        For CUDA tensors, this function returns the device ordinal of the GPU on which the tensor resides.
        For CPU tensors, this function returns `-1`.

        Example::

            >>> x = torch.randn(3, 4, 5, device='cuda:0')
            >>> x.get_device()
            0
            >>> x.cpu().get_device()
            -1
        """
    @overload
    def greater(self, other: Tensor) -> Tensor:
        """
        greater(other) -> Tensor

        See :func:`torch.greater`.
        """
    @overload
    def greater(self, other: Number | _complex) -> Tensor:
        """
        greater(other) -> Tensor

        See :func:`torch.greater`.
        """
    @overload
    def greater_(self, other: Tensor) -> Tensor:
        """
        greater_(other) -> Tensor

        In-place version of :meth:`~Tensor.greater`.
        """
    @overload
    def greater_(self, other: Number | _complex) -> Tensor:
        """
        greater_(other) -> Tensor

        In-place version of :meth:`~Tensor.greater`.
        """
    @overload
    def greater_equal(self, other: Tensor) -> Tensor:
        """
        greater_equal(other) -> Tensor

        See :func:`torch.greater_equal`.
        """
    @overload
    def greater_equal(self, other: Number | _complex) -> Tensor:
        """
        greater_equal(other) -> Tensor

        See :func:`torch.greater_equal`.
        """
    @overload
    def greater_equal_(self, other: Tensor) -> Tensor:
        """
        greater_equal_(other) -> Tensor

        In-place version of :meth:`~Tensor.greater_equal`.
        """
    @overload
    def greater_equal_(self, other: Number | _complex) -> Tensor:
        """
        greater_equal_(other) -> Tensor

        In-place version of :meth:`~Tensor.greater_equal`.
        """
    @overload
    def gt(self, other: Tensor) -> Tensor:
        """
        gt(other) -> Tensor

        See :func:`torch.gt`.
        """
    @overload
    def gt(self, other: Number | _complex) -> Tensor:
        """
        gt(other) -> Tensor

        See :func:`torch.gt`.
        """
    @overload
    def gt_(self, other: Tensor) -> Tensor:
        """
        gt_(other) -> Tensor

        In-place version of :meth:`~Tensor.gt`.
        """
    @overload
    def gt_(self, other: Number | _complex) -> Tensor:
        """
        gt_(other) -> Tensor

        In-place version of :meth:`~Tensor.gt`.
        """
    def half(self) -> Tensor:
        """
        half(memory_format=torch.preserve_format) -> Tensor

        ``self.half()`` is equivalent to ``self.to(torch.float16)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
    def hardshrink(self, lambd: Number | _complex = ...) -> Tensor:
        """
        hardshrink(lambd=0.5) -> Tensor

        See :func:`torch.nn.functional.hardshrink`
        """
    def has_names(self) -> _bool:
        """Is ``True`` if any of this tensor's dimensions are named. Otherwise, is ``False``."""
    @overload
    def hash_tensor(self, dim: _int | _size = ..., *, keepdim: _bool = ..., mode: _int = ...) -> Tensor: ...
    @overload
    def hash_tensor(self, *dim: _int, keepdim: _bool = ..., mode: _int = ...) -> Tensor: ...
    def heaviside(self, values: Tensor) -> Tensor:
        """
        heaviside(values) -> Tensor

        See :func:`torch.heaviside`
        """
    def heaviside_(self, values: Tensor) -> Tensor:
        """
        heaviside_(values) -> Tensor

        In-place version of :meth:`~Tensor.heaviside`
        """
    def histc(self, bins: _int = ..., min: Number | _complex = ..., max: Number | _complex = ...) -> Tensor:
        """
        histc(bins=100, min=0, max=0) -> Tensor

        See :func:`torch.histc`
        """
    @overload
    def histogram(
        self, bins: Tensor, *, weight: Tensor | None = ..., density: _bool = ...
    ) -> torch.return_types.histogram:
        """
        histogram(input, bins, *, range=None, weight=None, density=False) -> (Tensor, Tensor)

        See :func:`torch.histogram`
        """
    @overload
    def histogram(
        self,
        bins: _int = ...,
        *,
        range: Sequence[_float] | None = ...,
        weight: Tensor | None = ...,
        density: _bool = ...,
    ) -> torch.return_types.histogram:
        """
        histogram(input, bins, *, range=None, weight=None, density=False) -> (Tensor, Tensor)

        See :func:`torch.histogram`
        """
    @overload
    def hsplit(self, sections: _int) -> tuple[Tensor, ...]:
        """
        hsplit(split_size_or_sections) -> List of Tensors

        See :func:`torch.hsplit`
        """
    @overload
    def hsplit(self, indices: _size) -> tuple[Tensor, ...]:
        """
        hsplit(split_size_or_sections) -> List of Tensors

        See :func:`torch.hsplit`
        """
    @overload
    def hsplit(self, *indices: _int) -> tuple[Tensor, ...]:
        """
        hsplit(split_size_or_sections) -> List of Tensors

        See :func:`torch.hsplit`
        """
    def hypot(self, other: Tensor) -> Tensor:
        """
        hypot(other) -> Tensor

        See :func:`torch.hypot`
        """
    def hypot_(self, other: Tensor) -> Tensor:
        """
        hypot_(other) -> Tensor

        In-place version of :meth:`~Tensor.hypot`
        """
    def i0(self) -> Tensor:
        """
        i0() -> Tensor

        See :func:`torch.i0`
        """
    def i0_(self) -> Tensor:
        """
        i0_() -> Tensor

        In-place version of :meth:`~Tensor.i0`
        """
    def igamma(self, other: Tensor) -> Tensor:
        """
        igamma(other) -> Tensor

        See :func:`torch.igamma`
        """
    def igamma_(self, other: Tensor) -> Tensor:
        """
        igamma_(other) -> Tensor

        In-place version of :meth:`~Tensor.igamma`
        """
    def igammac(self, other: Tensor) -> Tensor:
        """
        igammac(other) -> Tensor
        See :func:`torch.igammac`
        """
    def igammac_(self, other: Tensor) -> Tensor:
        """
        igammac_(other) -> Tensor
        In-place version of :meth:`~Tensor.igammac`
        """
    @overload
    def index_add(self, dim: _int, index: Tensor, source: Tensor, *, alpha: Number | _complex = ...) -> Tensor:
        """
        index_add(dim, index, source, *, alpha=1) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_add_`.
        """
    @overload
    def index_add(
        self, dim: str | EllipsisType | None, index: Tensor, source: Tensor, *, alpha: Number | _complex = ...
    ) -> Tensor:
        """
        index_add(dim, index, source, *, alpha=1) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_add_`.
        """
    def index_add_(self, dim: _int, index: Tensor, source: Tensor, *, alpha: Number | _complex = ...) -> Tensor:
        r"""
        index_add_(dim, index, source, *, alpha=1) -> Tensor

        Accumulate the elements of :attr:`alpha` times ``source`` into the :attr:`self`
        tensor by adding to the indices in the order given in :attr:`index`. For example,
        if ``dim == 0``, ``index[i] == j``, and ``alpha=-1``, then the ``i``\ th row of
        ``source`` is subtracted from the ``j``\ th row of :attr:`self`.

        The :attr:`dim`\ th dimension of ``source`` must have the same size as the
        length of :attr:`index` (which must be a vector), and all other dimensions must
        match :attr:`self`, or an error will be raised.

        For a 3-D tensor the output is given as::

            self[index[i], :, :] += alpha * src[i, :, :]  # if dim == 0
            self[:, index[i], :] += alpha * src[:, i, :]  # if dim == 1
            self[:, :, index[i]] += alpha * src[:, :, i]  # if dim == 2

        Note:
            This operation may behave nondeterministically when given tensors on a CUDA device. See :doc:`/notes/randomness` for more information.

        Args:
            dim (int): dimension along which to index
            index (Tensor): indices of ``source`` to select from,
                    should have dtype either `torch.int64` or `torch.int32`
            source (Tensor): the tensor containing values to add

        Keyword args:
            alpha (Number): the scalar multiplier for ``source``

        Example::

            >>> x = torch.ones(5, 3)
            >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
            >>> index = torch.tensor([0, 4, 2])
            >>> x.index_add_(0, index, t)
            tensor([[  2.,   3.,   4.],
                    [  1.,   1.,   1.],
                    [  8.,   9.,  10.],
                    [  1.,   1.,   1.],
                    [  5.,   6.,   7.]])
            >>> x.index_add_(0, index, t, alpha=-1)
            tensor([[  1.,   1.,   1.],
                    [  1.,   1.,   1.],
                    [  1.,   1.,   1.],
                    [  1.,   1.,   1.],
                    [  1.,   1.,   1.]])
        """
    @overload
    def index_copy(self, dim: _int, index: Tensor, source: Tensor) -> Tensor:
        """
        index_copy(dim, index, tensor2) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_copy_`.
        """
    @overload
    def index_copy(self, dim: str | EllipsisType | None, index: Tensor, source: Tensor) -> Tensor:
        """
        index_copy(dim, index, tensor2) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_copy_`.
        """
    @overload
    def index_copy_(self, dim: _int, index: Tensor, source: Tensor) -> Tensor:
        r"""
        index_copy_(dim, index, tensor) -> Tensor

        Copies the elements of :attr:`tensor` into the :attr:`self` tensor by selecting
        the indices in the order given in :attr:`index`. For example, if ``dim == 0``
        and ``index[i] == j``, then the ``i``\ th row of :attr:`tensor` is copied to the
        ``j``\ th row of :attr:`self`.

        The :attr:`dim`\ th dimension of :attr:`tensor` must have the same size as the
        length of :attr:`index` (which must be a vector), and all other dimensions must
        match :attr:`self`, or an error will be raised.

        .. note::
            If :attr:`index` contains duplicate entries, multiple elements from
            :attr:`tensor` will be copied to the same index of :attr:`self`. The result
            is nondeterministic since it depends on which copy occurs last.

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of :attr:`tensor` to select from
            tensor (Tensor): the tensor containing values to copy

        Example::

            >>> x = torch.zeros(5, 3)
            >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
            >>> index = torch.tensor([0, 4, 2])
            >>> x.index_copy_(0, index, t)
            tensor([[ 1.,  2.,  3.],
                    [ 0.,  0.,  0.],
                    [ 7.,  8.,  9.],
                    [ 0.,  0.,  0.],
                    [ 4.,  5.,  6.]])
        """
    @overload
    def index_copy_(self, dim: str | EllipsisType | None, index: Tensor, source: Tensor) -> Tensor:
        r"""
        index_copy_(dim, index, tensor) -> Tensor

        Copies the elements of :attr:`tensor` into the :attr:`self` tensor by selecting
        the indices in the order given in :attr:`index`. For example, if ``dim == 0``
        and ``index[i] == j``, then the ``i``\ th row of :attr:`tensor` is copied to the
        ``j``\ th row of :attr:`self`.

        The :attr:`dim`\ th dimension of :attr:`tensor` must have the same size as the
        length of :attr:`index` (which must be a vector), and all other dimensions must
        match :attr:`self`, or an error will be raised.

        .. note::
            If :attr:`index` contains duplicate entries, multiple elements from
            :attr:`tensor` will be copied to the same index of :attr:`self`. The result
            is nondeterministic since it depends on which copy occurs last.

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of :attr:`tensor` to select from
            tensor (Tensor): the tensor containing values to copy

        Example::

            >>> x = torch.zeros(5, 3)
            >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
            >>> index = torch.tensor([0, 4, 2])
            >>> x.index_copy_(0, index, t)
            tensor([[ 1.,  2.,  3.],
                    [ 0.,  0.,  0.],
                    [ 7.,  8.,  9.],
                    [ 0.,  0.,  0.],
                    [ 4.,  5.,  6.]])
        """
    @overload
    def index_fill(self, dim: _int, index: Tensor, value: Tensor) -> Tensor:
        """
        index_fill(dim, index, value) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_fill_`.
        """
    @overload
    def index_fill(self, dim: str | EllipsisType | None, index: Tensor, value: Tensor) -> Tensor:
        """
        index_fill(dim, index, value) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_fill_`.
        """
    @overload
    def index_fill(self, dim: _int, index: Tensor, value: Number | _complex) -> Tensor:
        """
        index_fill(dim, index, value) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_fill_`.
        """
    @overload
    def index_fill(self, dim: str | EllipsisType | None, index: Tensor, value: Number | _complex) -> Tensor:
        """
        index_fill(dim, index, value) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.index_fill_`.
        """
    @overload
    def index_fill_(self, dim: _int, index: Tensor, value: Tensor) -> Tensor:
        """
        index_fill_(dim, index, value) -> Tensor

        Fills the elements of the :attr:`self` tensor with value :attr:`value` by
        selecting the indices in the order given in :attr:`index`.

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of :attr:`self` tensor to fill in
            value (float): the value to fill with

        Example::

            >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
            >>> index = torch.tensor([0, 2])
            >>> x.index_fill_(1, index, -1)
            tensor([[-1.,  2., -1.],
                    [-1.,  5., -1.],
                    [-1.,  8., -1.]])
        """
    @overload
    def index_fill_(self, dim: str | EllipsisType | None, index: Tensor, value: Tensor) -> Tensor:
        """
        index_fill_(dim, index, value) -> Tensor

        Fills the elements of the :attr:`self` tensor with value :attr:`value` by
        selecting the indices in the order given in :attr:`index`.

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of :attr:`self` tensor to fill in
            value (float): the value to fill with

        Example::

            >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
            >>> index = torch.tensor([0, 2])
            >>> x.index_fill_(1, index, -1)
            tensor([[-1.,  2., -1.],
                    [-1.,  5., -1.],
                    [-1.,  8., -1.]])
        """
    @overload
    def index_fill_(self, dim: _int, index: Tensor, value: Number | _complex) -> Tensor:
        """
        index_fill_(dim, index, value) -> Tensor

        Fills the elements of the :attr:`self` tensor with value :attr:`value` by
        selecting the indices in the order given in :attr:`index`.

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of :attr:`self` tensor to fill in
            value (float): the value to fill with

        Example::

            >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
            >>> index = torch.tensor([0, 2])
            >>> x.index_fill_(1, index, -1)
            tensor([[-1.,  2., -1.],
                    [-1.,  5., -1.],
                    [-1.,  8., -1.]])
        """
    @overload
    def index_fill_(self, dim: str | EllipsisType | None, index: Tensor, value: Number | _complex) -> Tensor:
        """
        index_fill_(dim, index, value) -> Tensor

        Fills the elements of the :attr:`self` tensor with value :attr:`value` by
        selecting the indices in the order given in :attr:`index`.

        Args:
            dim (int): dimension along which to index
            index (LongTensor): indices of :attr:`self` tensor to fill in
            value (float): the value to fill with

        Example::

            >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
            >>> index = torch.tensor([0, 2])
            >>> x.index_fill_(1, index, -1)
            tensor([[-1.,  2., -1.],
                    [-1.,  5., -1.],
                    [-1.,  8., -1.]])
        """
    def index_put(
        self, indices: tuple[Tensor, ...] | list[Tensor] | None, values: Tensor, accumulate: _bool = ...
    ) -> Tensor:
        """
        index_put(indices, values, accumulate=False) -> Tensor

        Out-place version of :meth:`~Tensor.index_put_`.
        """
    def index_put_(
        self, indices: tuple[Tensor, ...] | list[Tensor] | None, values: Tensor, accumulate: _bool = ...
    ) -> Tensor:
        """
        index_put_(indices, values, accumulate=False) -> Tensor

        Puts values from the tensor :attr:`values` into the tensor :attr:`self` using
        the indices specified in :attr:`indices` (which is a tuple of Tensors). The
        expression ``tensor.index_put_(indices, values)`` is equivalent to
        ``tensor[indices] = values``. Returns :attr:`self`.

        If :attr:`accumulate` is ``True``, the elements in :attr:`values` are added to
        :attr:`self`. If accumulate is ``False``, the behavior is undefined if indices
        contain duplicate elements.

        Args:
            indices (tuple of LongTensor): tensors used to index into `self`.
            values (Tensor): tensor of same dtype as `self`.
            accumulate (bool): whether to accumulate into self
        """
    def index_reduce(
        self, dim: _int, index: Tensor, source: Tensor, reduce: str, *, include_self: _bool = ...
    ) -> Tensor: ...
    def index_reduce_(
        self, dim: _int, index: Tensor, source: Tensor, reduce: str, *, include_self: _bool = ...
    ) -> Tensor:
        r"""
        index_reduce_(dim, index, source, reduce, *, include_self=True) -> Tensor

        Accumulate the elements of ``source`` into the :attr:`self`
        tensor by accumulating to the indices in the order given in :attr:`index`
        using the reduction given by the ``reduce`` argument. For example, if ``dim == 0``,
        ``index[i] == j``, ``reduce == prod`` and ``include_self == True`` then the ``i``\ th
        row of ``source`` is multiplied by the ``j``\ th row of :attr:`self`. If
        :obj:`include_self="True"`, the values in the :attr:`self` tensor are included
        in the reduction, otherwise, rows in the :attr:`self` tensor that are accumulated
        to are treated as if they were filled with the reduction identities.

        The :attr:`dim`\ th dimension of ``source`` must have the same size as the
        length of :attr:`index` (which must be a vector), and all other dimensions must
        match :attr:`self`, or an error will be raised.

        For a 3-D tensor with :obj:`reduce="prod"` and :obj:`include_self=True` the
        output is given as::

            self[index[i], :, :] *= src[i, :, :]  # if dim == 0
            self[:, index[i], :] *= src[:, i, :]  # if dim == 1
            self[:, :, index[i]] *= src[:, :, i]  # if dim == 2

        Note:
            This operation may behave nondeterministically when given tensors on a CUDA device. See :doc:`/notes/randomness` for more information.

        .. note::

            This function only supports floating point tensors.

        .. warning::

            This function is in beta and may change in the near future.

        Args:
            dim (int): dimension along which to index
            index (Tensor): indices of ``source`` to select from,
                should have dtype either `torch.int64` or `torch.int32`
            source (FloatTensor): the tensor containing values to accumulate
            reduce (str): the reduction operation to apply
                (:obj:`"prod"`, :obj:`"mean"`, :obj:`"amax"`, :obj:`"amin"`)

        Keyword args:
            include_self (bool): whether the elements from the ``self`` tensor are
                included in the reduction

        Example::

            >>> x = torch.empty(5, 3).fill_(2)
            >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float)
            >>> index = torch.tensor([0, 4, 2, 0])
            >>> x.index_reduce_(0, index, t, 'prod')
            tensor([[20., 44., 72.],
                    [ 2.,  2.,  2.],
                    [14., 16., 18.],
                    [ 2.,  2.,  2.],
                    [ 8., 10., 12.]])
            >>> x = torch.empty(5, 3).fill_(2)
            >>> x.index_reduce_(0, index, t, 'prod', include_self=False)
            tensor([[10., 22., 36.],
                    [ 2.,  2.,  2.],
                    [ 7.,  8.,  9.],
                    [ 2.,  2.,  2.],
                    [ 4.,  5.,  6.]])
        """
    @overload
    def index_select(self, dim: _int, index: Tensor) -> Tensor:
        """
        index_select(dim, index) -> Tensor

        See :func:`torch.index_select`
        """
    @overload
    def index_select(self, dim: str | EllipsisType | None, index: Tensor) -> Tensor:
        """
        index_select(dim, index) -> Tensor

        See :func:`torch.index_select`
        """
    def indices(self) -> Tensor:
        """
        indices() -> Tensor

        Return the indices tensor of a :ref:`sparse COO tensor <sparse-coo-docs>`.

        .. warning::
          Throws an error if :attr:`self` is not a sparse COO tensor.

        See also :meth:`Tensor.values`.

        .. note::
          This method can only be called on a coalesced sparse tensor. See
          :meth:`Tensor.coalesce` for details.
        """
    def inner(self, other: Tensor) -> Tensor:
        """
        inner(other) -> Tensor

        See :func:`torch.inner`.
        """
    def int(self) -> Tensor:
        """
        int(memory_format=torch.preserve_format) -> Tensor

        ``self.int()`` is equivalent to ``self.to(torch.int32)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
    def int_repr(self) -> Tensor:
        """
        int_repr() -> Tensor

        Given a quantized Tensor,
        ``self.int_repr()`` returns a CPU Tensor with uint8_t as data type that stores the
        underlying uint8_t values of the given Tensor.
        """
    def inverse(self) -> Tensor:
        """
        inverse() -> Tensor

        See :func:`torch.inverse`
        """
    def is_coalesced(self) -> _bool:
        """
        is_coalesced() -> bool

        Returns ``True`` if :attr:`self` is a :ref:`sparse COO tensor
        <sparse-coo-docs>` that is coalesced, ``False`` otherwise.

        .. warning::
          Throws an error if :attr:`self` is not a sparse COO tensor.

        See :meth:`coalesce` and :ref:`uncoalesced tensors <sparse-uncoalesced-coo-docs>`.
        """
    def is_complex(self) -> _bool:
        """
        is_complex() -> bool

        Returns True if the data type of :attr:`self` is a complex data type.
        """
    def is_conj(self) -> _bool:
        """
        is_conj() -> bool

        Returns True if the conjugate bit of :attr:`self` is set to true.
        """
    def is_contiguous(self, memory_format: torch.memory_format = ...) -> _bool:
        """
        is_contiguous(memory_format=torch.contiguous_format) -> bool

        Returns True if :attr:`self` tensor is contiguous in memory in the order specified
        by memory format.

        Args:
            memory_format (:class:`torch.memory_format`, optional): Specifies memory allocation
                order. Default: ``torch.contiguous_format``.
        """

    is_cpu: _bool
    is_cuda: _bool
    def is_distributed(self) -> _bool: ...
    def is_floating_point(self) -> _bool:
        """
        is_floating_point() -> bool

        Returns True if the data type of :attr:`self` is a floating point data type.
        """
    def is_inference(self) -> _bool:
        """
        is_inference() -> bool

        See :func:`torch.is_inference`
        """

    is_ipu: _bool
    is_leaf: _bool
    is_maia: _bool
    is_meta: _bool
    is_mkldnn: _bool
    is_mps: _bool
    is_mtia: _bool
    def is_neg(self) -> _bool:
        """
        is_neg() -> bool

        Returns True if the negative bit of :attr:`self` is set to true.
        """

    is_nested: _bool
    def is_nonzero(self) -> _bool: ...
    def is_pinned(self, device: DeviceLikeType | None = ...) -> _bool:
        """
        Returns true if this tensor resides in pinned memory.
        By default, the device pinned memory on will be the current :ref:`accelerator<accelerators>`.
        """

    is_quantized: _bool
    def is_same_size(self, other: Tensor) -> _bool: ...
    def is_set_to(self, tensor: Tensor) -> _bool:
        """
        is_set_to(tensor) -> bool

        Returns True if both tensors are pointing to the exact same memory (same
        storage, offset, size and stride).
        """
    def is_signed(self) -> _bool:
        """
        is_signed() -> bool

        Returns True if the data type of :attr:`self` is a signed data type.
        """

    is_sparse: _bool
    is_sparse_csr: _bool
    is_vulkan: _bool
    is_xpu: _bool
    def isclose(self, other: Tensor, rtol: _float = ..., atol: _float = ..., equal_nan: _bool = ...) -> Tensor:
        """
        isclose(other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor

        See :func:`torch.isclose`
        """
    def isfinite(self) -> Tensor:
        """
        isfinite() -> Tensor

        See :func:`torch.isfinite`
        """
    def isinf(self) -> Tensor:
        """
        isinf() -> Tensor

        See :func:`torch.isinf`
        """
    def isnan(self) -> Tensor:
        """
        isnan() -> Tensor

        See :func:`torch.isnan`
        """
    def isneginf(self) -> Tensor:
        """
        isneginf() -> Tensor

        See :func:`torch.isneginf`
        """
    def isposinf(self) -> Tensor:
        """
        isposinf() -> Tensor

        See :func:`torch.isposinf`
        """
    def isreal(self) -> Tensor:
        """
        isreal() -> Tensor

        See :func:`torch.isreal`
        """
    def istft(
        self,
        n_fft: _int,
        hop_length: _int | None = ...,
        win_length: _int | None = ...,
        window: Tensor | None = ...,
        center: _bool = ...,
        normalized: _bool = ...,
        onesided: _bool | None = ...,
        length: _int | None = ...,
        return_complex: _bool = ...,
    ) -> Tensor:
        """
        istft(n_fft, hop_length=None, win_length=None, window=None,
         center=True, normalized=False, onesided=True, length=None) -> Tensor

        See :func:`torch.istft`
        """
    def item(self) -> Number:
        """
        item() -> number

        Returns the value of this tensor as a standard Python number. This only works
        for tensors with one element. For other cases, see :meth:`~Tensor.tolist`.

        This operation is not differentiable.

        Example::

            >>> x = torch.tensor([1.0])
            >>> x.item()
            1.0
        """
    def kron(self, other: Tensor) -> Tensor:
        """
        kron(other) -> Tensor

        See :func:`torch.kron`
        """
    @overload
    def kthvalue(self, k: _int | SymInt, dim: _int = ..., keepdim: _bool = ...) -> torch.return_types.kthvalue:
        """
        kthvalue(k, dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.kthvalue`
        """
    @overload
    def kthvalue(
        self, k: _int | SymInt, dim: str | EllipsisType | None, keepdim: _bool = ...
    ) -> torch.return_types.kthvalue:
        """
        kthvalue(k, dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.kthvalue`
        """
    def lcm(self, other: Tensor) -> Tensor:
        """
        lcm(other) -> Tensor

        See :func:`torch.lcm`
        """
    def lcm_(self, other: Tensor) -> Tensor:
        """
        lcm_(other) -> Tensor

        In-place version of :meth:`~Tensor.lcm`
        """
    def ldexp(self, other: Tensor) -> Tensor:
        """
        ldexp(other) -> Tensor

        See :func:`torch.ldexp`
        """
    def ldexp_(self, other: Tensor) -> Tensor:
        """
        ldexp_(other) -> Tensor

        In-place version of :meth:`~Tensor.ldexp`
        """
    @overload
    def le(self, other: Tensor) -> Tensor:
        """
        le(other) -> Tensor

        See :func:`torch.le`.
        """
    @overload
    def le(self, other: Number | _complex) -> Tensor:
        """
        le(other) -> Tensor

        See :func:`torch.le`.
        """
    @overload
    def le_(self, other: Tensor) -> Tensor:
        """
        le_(other) -> Tensor

        In-place version of :meth:`~Tensor.le`.
        """
    @overload
    def le_(self, other: Number | _complex) -> Tensor:
        """
        le_(other) -> Tensor

        In-place version of :meth:`~Tensor.le`.
        """
    @overload
    def lerp(self, end: Tensor, weight: Tensor) -> Tensor:
        """
        lerp(end, weight) -> Tensor

        See :func:`torch.lerp`
        """
    @overload
    def lerp(self, end: Tensor, weight: Number | _complex) -> Tensor:
        """
        lerp(end, weight) -> Tensor

        See :func:`torch.lerp`
        """
    @overload
    def lerp_(self, end: Tensor, weight: Tensor) -> Tensor:
        """
        lerp_(end, weight) -> Tensor

        In-place version of :meth:`~Tensor.lerp`
        """
    @overload
    def lerp_(self, end: Tensor, weight: Number | _complex) -> Tensor:
        """
        lerp_(end, weight) -> Tensor

        In-place version of :meth:`~Tensor.lerp`
        """
    @overload
    def less(self, other: Tensor) -> Tensor:
        """
        lt(other) -> Tensor

        See :func:`torch.less`.
        """
    @overload
    def less(self, other: Number | _complex) -> Tensor:
        """
        lt(other) -> Tensor

        See :func:`torch.less`.
        """
    @overload
    def less_(self, other: Tensor) -> Tensor:
        """
        less_(other) -> Tensor

        In-place version of :meth:`~Tensor.less`.
        """
    @overload
    def less_(self, other: Number | _complex) -> Tensor:
        """
        less_(other) -> Tensor

        In-place version of :meth:`~Tensor.less`.
        """
    @overload
    def less_equal(self, other: Tensor) -> Tensor:
        """
        less_equal(other) -> Tensor

        See :func:`torch.less_equal`.
        """
    @overload
    def less_equal(self, other: Number | _complex) -> Tensor:
        """
        less_equal(other) -> Tensor

        See :func:`torch.less_equal`.
        """
    @overload
    def less_equal_(self, other: Tensor) -> Tensor:
        """
        less_equal_(other) -> Tensor

        In-place version of :meth:`~Tensor.less_equal`.
        """
    @overload
    def less_equal_(self, other: Number | _complex) -> Tensor:
        """
        less_equal_(other) -> Tensor

        In-place version of :meth:`~Tensor.less_equal`.
        """
    def lgamma(self) -> Tensor:
        """
        lgamma() -> Tensor

        See :func:`torch.lgamma`
        """
    def lgamma_(self) -> Tensor:
        """
        lgamma_() -> Tensor

        In-place version of :meth:`~Tensor.lgamma`
        """
    def log(self) -> Tensor:
        """
        log() -> Tensor

        See :func:`torch.log`
        """
    def log10(self) -> Tensor:
        """
        log10() -> Tensor

        See :func:`torch.log10`
        """
    def log10_(self) -> Tensor:
        """
        log10_() -> Tensor

        In-place version of :meth:`~Tensor.log10`
        """
    def log1p(self) -> Tensor:
        """
        log1p() -> Tensor

        See :func:`torch.log1p`
        """
    def log1p_(self) -> Tensor:
        """
        log1p_() -> Tensor

        In-place version of :meth:`~Tensor.log1p`
        """
    def log2(self) -> Tensor:
        """
        log2() -> Tensor

        See :func:`torch.log2`
        """
    def log2_(self) -> Tensor:
        """
        log2_() -> Tensor

        In-place version of :meth:`~Tensor.log2`
        """
    def log_(self) -> Tensor:
        """
        log_() -> Tensor

        In-place version of :meth:`~Tensor.log`
        """
    def log_normal_(self, mean: _float = ..., std: _float = ..., *, generator: Generator | None = ...) -> Tensor:
        r"""
        log_normal_(mean=1, std=2, *, generator=None)

        Fills :attr:`self` tensor with numbers samples from the log-normal distribution
        parameterized by the given mean :math:`\mu` and standard deviation
        :math:`\sigma`. Note that :attr:`mean` and :attr:`std` are the mean and
        standard deviation of the underlying normal distribution, and not of the
        returned distribution:

        .. math::

            f(x) = \dfrac{1}{x \sigma \sqrt{2\pi}}\ e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}
        """
    @overload
    def log_softmax(self, dim: _int, dtype: _dtype | None = ...) -> Tensor: ...
    @overload
    def log_softmax(self, dim: str | EllipsisType | None, *, dtype: _dtype | None = ...) -> Tensor: ...
    def logaddexp(self, other: Tensor) -> Tensor:
        """
        logaddexp(other) -> Tensor

        See :func:`torch.logaddexp`
        """
    def logaddexp2(self, other: Tensor) -> Tensor:
        """
        logaddexp2(other) -> Tensor

        See :func:`torch.logaddexp2`
        """
    @overload
    def logcumsumexp(self, dim: _int) -> Tensor:
        """
        logcumsumexp(dim) -> Tensor

        See :func:`torch.logcumsumexp`
        """
    @overload
    def logcumsumexp(self, dim: str | EllipsisType | None) -> Tensor:
        """
        logcumsumexp(dim) -> Tensor

        See :func:`torch.logcumsumexp`
        """
    def logdet(self) -> Tensor:
        """
        logdet() -> Tensor

        See :func:`torch.logdet`
        """
    def logical_and(self, other: Tensor) -> Tensor:
        """
        logical_and() -> Tensor

        See :func:`torch.logical_and`
        """
    def logical_and_(self, other: Tensor) -> Tensor:
        """
        logical_and_() -> Tensor

        In-place version of :meth:`~Tensor.logical_and`
        """
    def logical_not(self) -> Tensor:
        """
        logical_not() -> Tensor

        See :func:`torch.logical_not`
        """
    def logical_not_(self) -> Tensor:
        """
        logical_not_() -> Tensor

        In-place version of :meth:`~Tensor.logical_not`
        """
    def logical_or(self, other: Tensor) -> Tensor:
        """
        logical_or() -> Tensor

        See :func:`torch.logical_or`
        """
    def logical_or_(self, other: Tensor) -> Tensor:
        """
        logical_or_() -> Tensor

        In-place version of :meth:`~Tensor.logical_or`
        """
    def logical_xor(self, other: Tensor) -> Tensor:
        """
        logical_xor() -> Tensor

        See :func:`torch.logical_xor`
        """
    def logical_xor_(self, other: Tensor) -> Tensor:
        """
        logical_xor_() -> Tensor

        In-place version of :meth:`~Tensor.logical_xor`
        """
    def logit(self, eps: _float | None = ...) -> Tensor:
        """
        logit() -> Tensor

        See :func:`torch.logit`
        """
    def logit_(self, eps: _float | None = ...) -> Tensor:
        """
        logit_() -> Tensor

        In-place version of :meth:`~Tensor.logit`
        """
    @overload
    def logsumexp(self, dim: _int | _size, keepdim: _bool = ...) -> Tensor:
        """
        logsumexp(dim, keepdim=False) -> Tensor

        See :func:`torch.logsumexp`
        """
    @overload
    def logsumexp(self, dim: Sequence[str | EllipsisType | None], keepdim: _bool = ...) -> Tensor:
        """
        logsumexp(dim, keepdim=False) -> Tensor

        See :func:`torch.logsumexp`
        """
    def long(self) -> Tensor:
        """
        long(memory_format=torch.preserve_format) -> Tensor

        ``self.long()`` is equivalent to ``self.to(torch.int64)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
    @overload
    def lt(self, other: Tensor) -> Tensor:
        """
        lt(other) -> Tensor

        See :func:`torch.lt`.
        """
    @overload
    def lt(self, other: Number | _complex) -> Tensor:
        """
        lt(other) -> Tensor

        See :func:`torch.lt`.
        """
    @overload
    def lt_(self, other: Tensor) -> Tensor:
        """
        lt_(other) -> Tensor

        In-place version of :meth:`~Tensor.lt`.
        """
    @overload
    def lt_(self, other: Number | _complex) -> Tensor:
        """
        lt_(other) -> Tensor

        In-place version of :meth:`~Tensor.lt`.
        """
    def lu_solve(self, LU_data: Tensor, LU_pivots: Tensor) -> Tensor:
        """
        lu_solve(LU_data, LU_pivots) -> Tensor

        See :func:`torch.lu_solve`
        """
    def map2_(self, x: Tensor, y: Tensor, callable: Callable) -> Tensor: ...
    def map_(self, other: Tensor, callable: Callable) -> Tensor:
        """
        map_(tensor, callable)

        Applies :attr:`callable` for each element in :attr:`self` tensor and the given
        :attr:`tensor` and stores the results in :attr:`self` tensor. :attr:`self` tensor and
        the given :attr:`tensor` must be :ref:`broadcastable <broadcasting-semantics>`.

        The :attr:`callable` should have the signature::

            def callable(a, b) -> number
        """
    @overload
    def masked_fill(self, mask: Tensor, value: Tensor) -> Tensor:
        """
        masked_fill(mask, value) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.masked_fill_`
        """
    @overload
    def masked_fill(self, mask: Tensor, value: Number | _complex) -> Tensor:
        """
        masked_fill(mask, value) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.masked_fill_`
        """
    @overload
    def masked_fill_(self, mask: Tensor, value: Tensor) -> Tensor:
        """
        masked_fill_(mask, value)

        Fills elements of :attr:`self` tensor with :attr:`value` where :attr:`mask` is
        True. The shape of :attr:`mask` must be
        :ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying
        tensor.

        Args:
            mask (BoolTensor): the boolean mask
            value (float): the value to fill in with
        """
    @overload
    def masked_fill_(self, mask: Tensor, value: Number | _complex) -> Tensor:
        """
        masked_fill_(mask, value)

        Fills elements of :attr:`self` tensor with :attr:`value` where :attr:`mask` is
        True. The shape of :attr:`mask` must be
        :ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying
        tensor.

        Args:
            mask (BoolTensor): the boolean mask
            value (float): the value to fill in with
        """
    def masked_scatter(self, mask: Tensor, source: Tensor) -> Tensor:
        """
        masked_scatter(mask, tensor) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.masked_scatter_`

        .. note::

            The inputs :attr:`self` and :attr:`mask`
            :ref:`broadcast <broadcasting-semantics>`.

        Example:

            >>> self = torch.tensor([0, 0, 0, 0, 0])
            >>> mask = torch.tensor(
            ...     [[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]],
            ...     dtype=torch.bool,
            ... )
            >>> source = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
            >>> self.masked_scatter(mask, source)
            tensor([[0, 0, 0, 0, 1],
                    [2, 3, 0, 4, 5]])
        """
    def masked_scatter_(self, mask: Tensor, source: Tensor) -> Tensor:
        """
        masked_scatter_(mask, source)

        Copies elements from :attr:`source` into :attr:`self` tensor at positions where
        the :attr:`mask` is True. Elements from :attr:`source` are copied into :attr:`self`
        starting at position 0 of :attr:`source` and continuing in order one-by-one for each
        occurrence of :attr:`mask` being True.
        The shape of :attr:`mask` must be :ref:`broadcastable <broadcasting-semantics>`
        with the shape of the underlying tensor. The :attr:`source` should have at least
        as many elements as the number of ones in :attr:`mask`.

        Args:
            mask (BoolTensor): the boolean mask
            source (Tensor): the tensor to copy from

        .. note::

            The :attr:`mask` operates on the :attr:`self` tensor, not on the given
            :attr:`source` tensor.

        Example:

            >>> self = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
            >>> mask = torch.tensor(
            ...     [[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]],
            ...     dtype=torch.bool,
            ... )
            >>> source = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
            >>> self.masked_scatter_(mask, source)
            tensor([[0, 0, 0, 0, 1],
                    [2, 3, 0, 4, 5]])
        """
    def masked_select(self, mask: Tensor) -> Tensor:
        """
        masked_select(mask) -> Tensor

        See :func:`torch.masked_select`
        """
    def matmul(self, other: Tensor) -> Tensor:
        """
        matmul(tensor2) -> Tensor

        See :func:`torch.matmul`
        """
    def matrix_exp(self) -> Tensor:
        """
        matrix_exp() -> Tensor

        See :func:`torch.matrix_exp`
        """
    def matrix_power(self, n: _int) -> Tensor:
        """
        matrix_power(n) -> Tensor

        .. note:: :meth:`~Tensor.matrix_power` is deprecated, use :func:`torch.linalg.matrix_power` instead.

        Alias for :func:`torch.linalg.matrix_power`
        """
    @overload
    def max(self) -> Tensor:
        """
        max(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

        See :func:`torch.max`
        """
    @overload
    def max(self, other: Tensor) -> Tensor:
        """
        max(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

        See :func:`torch.max`
        """
    @overload
    def max(self, dim: _int, keepdim: _bool = ...) -> torch.return_types.max:
        """
        max(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

        See :func:`torch.max`
        """
    @overload
    def max(self, dim: str | EllipsisType | None, keepdim: _bool = ...) -> torch.return_types.max:
        """
        max(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

        See :func:`torch.max`
        """
    def maximum(self, other: Tensor) -> Tensor:
        """
        maximum(other) -> Tensor

        See :func:`torch.maximum`
        """
    @overload
    def mean(self, *, dtype: _dtype | None = ...) -> Tensor:
        """
        mean(dim=None, keepdim=False, *, dtype=None) -> Tensor

        See :func:`torch.mean`
        """
    @overload
    def mean(self, dim: _int | _size | None, keepdim: _bool = ..., *, dtype: _dtype | None = ...) -> Tensor:
        """
        mean(dim=None, keepdim=False, *, dtype=None) -> Tensor

        See :func:`torch.mean`
        """
    @overload
    def mean(
        self, dim: Sequence[str | EllipsisType | None], keepdim: _bool = ..., *, dtype: _dtype | None = ...
    ) -> Tensor:
        """
        mean(dim=None, keepdim=False, *, dtype=None) -> Tensor

        See :func:`torch.mean`
        """
    @overload
    def median(self) -> Tensor:
        """
        median(dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.median`
        """
    @overload
    def median(self, dim: _int, keepdim: _bool = ...) -> torch.return_types.median:
        """
        median(dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.median`
        """
    @overload
    def median(self, dim: str | EllipsisType | None, keepdim: _bool = ...) -> torch.return_types.median:
        """
        median(dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.median`
        """
    @overload
    def min(self) -> Tensor:
        """
        min(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

        See :func:`torch.min`
        """
    @overload
    def min(self, other: Tensor) -> Tensor:
        """
        min(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

        See :func:`torch.min`
        """
    @overload
    def min(self, dim: _int, keepdim: _bool = ...) -> torch.return_types.min:
        """
        min(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

        See :func:`torch.min`
        """
    @overload
    def min(self, dim: str | EllipsisType | None, keepdim: _bool = ...) -> torch.return_types.min:
        """
        min(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

        See :func:`torch.min`
        """
    def minimum(self, other: Tensor) -> Tensor:
        """
        minimum(other) -> Tensor

        See :func:`torch.minimum`
        """
    def mm(self, mat2: Tensor) -> Tensor:
        """
        mm(mat2) -> Tensor

        See :func:`torch.mm`
        """
    @overload
    def mode(self, dim: _int = ..., keepdim: _bool = ...) -> torch.return_types.mode:
        """
        mode(dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.mode`
        """
    @overload
    def mode(self, dim: str | EllipsisType | None, keepdim: _bool = ...) -> torch.return_types.mode:
        """
        mode(dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.mode`
        """
    @overload
    def moveaxis(self, source: _int, destination: _int) -> Tensor:
        """
        moveaxis(source, destination) -> Tensor

        See :func:`torch.moveaxis`
        """
    @overload
    def moveaxis(self, source: _size, destination: _size) -> Tensor:
        """
        moveaxis(source, destination) -> Tensor

        See :func:`torch.moveaxis`
        """
    @overload
    def movedim(self, source: _int, destination: _int) -> Tensor:
        """
        movedim(source, destination) -> Tensor

        See :func:`torch.movedim`
        """
    @overload
    def movedim(self, source: _size, destination: _size) -> Tensor:
        """
        movedim(source, destination) -> Tensor

        See :func:`torch.movedim`
        """
    def msort(self) -> Tensor:
        """
        msort() -> Tensor

        See :func:`torch.msort`
        """
    def mul(
        self, other: Tensor | Number | _complex | torch.SymInt | torch.SymFloat, *, out: Tensor | None = ...
    ) -> Tensor:
        """
        mul(value) -> Tensor

        See :func:`torch.mul`.
        """
    def mul_(self, other: Tensor | Number | _complex | torch.SymInt | torch.SymFloat) -> Tensor:
        """
        mul_(value) -> Tensor

        In-place version of :meth:`~Tensor.mul`.
        """
    def multinomial(
        self, num_samples: _int | SymInt, replacement: _bool = ..., *, generator: Generator | None = ...
    ) -> Tensor:
        """
        multinomial(num_samples, replacement=False, *, generator=None) -> Tensor

        See :func:`torch.multinomial`
        """
    @overload
    def multiply(self, other: Tensor) -> Tensor:
        """
        multiply(value) -> Tensor

        See :func:`torch.multiply`.
        """
    @overload
    def multiply(self, other: Number | _complex) -> Tensor:
        """
        multiply(value) -> Tensor

        See :func:`torch.multiply`.
        """
    @overload
    def multiply_(self, other: Tensor) -> Tensor:
        """
        multiply_(value) -> Tensor

        In-place version of :meth:`~Tensor.multiply`.
        """
    @overload
    def multiply_(self, other: Number | _complex) -> Tensor:
        """
        multiply_(value) -> Tensor

        In-place version of :meth:`~Tensor.multiply`.
        """
    def mv(self, vec: Tensor) -> Tensor:
        """
        mv(vec) -> Tensor

        See :func:`torch.mv`
        """
    def mvlgamma(self, p: _int) -> Tensor:
        """
        mvlgamma(p) -> Tensor

        See :func:`torch.mvlgamma`
        """
    def mvlgamma_(self, p: _int) -> Tensor:
        """
        mvlgamma_(p) -> Tensor

        In-place version of :meth:`~Tensor.mvlgamma`
        """
    def nan_to_num(self, nan: _float | None = ..., posinf: _float | None = ..., neginf: _float | None = ...) -> Tensor:
        """
        nan_to_num(nan=0.0, posinf=None, neginf=None) -> Tensor

        See :func:`torch.nan_to_num`.
        """
    def nan_to_num_(self, nan: _float | None = ..., posinf: _float | None = ..., neginf: _float | None = ...) -> Tensor:
        """
        nan_to_num_(nan=0.0, posinf=None, neginf=None) -> Tensor

        In-place version of :meth:`~Tensor.nan_to_num`.
        """
    def nanmean(self, dim: _int | _size | None = ..., keepdim: _bool = ..., *, dtype: _dtype | None = ...) -> Tensor:
        """
        nanmean(dim=None, keepdim=False, *, dtype=None) -> Tensor

        See :func:`torch.nanmean`
        """
    @overload
    def nanmedian(self) -> Tensor:
        """
        nanmedian(dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.nanmedian`
        """
    @overload
    def nanmedian(self, dim: _int, keepdim: _bool = ...) -> torch.return_types.nanmedian:
        """
        nanmedian(dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.nanmedian`
        """
    @overload
    def nanmedian(self, dim: str | EllipsisType | None, keepdim: _bool = ...) -> torch.return_types.nanmedian:
        """
        nanmedian(dim=None, keepdim=False) -> (Tensor, LongTensor)

        See :func:`torch.nanmedian`
        """
    @overload
    def nanquantile(
        self, q: Tensor, dim: _int | None = ..., keepdim: _bool = ..., *, interpolation: str = ...
    ) -> Tensor:
        """
        nanquantile(q, dim=None, keepdim=False, *, interpolation='linear') -> Tensor

        See :func:`torch.nanquantile`
        """
    @overload
    def nanquantile(
        self, q: _float, dim: _int | None = ..., keepdim: _bool = ..., *, interpolation: str = ...
    ) -> Tensor:
        """
        nanquantile(q, dim=None, keepdim=False, *, interpolation='linear') -> Tensor

        See :func:`torch.nanquantile`
        """
    def nansum(self, dim: _int | _size | None = ..., keepdim: _bool = ..., *, dtype: _dtype | None = ...) -> Tensor:
        """
        nansum(dim=None, keepdim=False, dtype=None) -> Tensor

        See :func:`torch.nansum`
        """
    @overload
    def narrow(self, dim: _int, start: Tensor, length: _int | SymInt) -> Tensor:
        """
        narrow(dimension, start, length) -> Tensor

        See :func:`torch.narrow`.
        """
    @overload
    def narrow(self, dim: _int, start: _int | SymInt, length: _int | SymInt) -> Tensor:
        """
        narrow(dimension, start, length) -> Tensor

        See :func:`torch.narrow`.
        """
    def narrow_copy(self, dim: _int, start: _int | SymInt, length: _int | SymInt) -> Tensor:
        """
        narrow_copy(dimension, start, length) -> Tensor

        See :func:`torch.narrow_copy`.
        """
    def ndimension(self) -> _int:
        """
        ndimension() -> int

        Alias for :meth:`~Tensor.dim()`
        """
    @overload
    def ne(self, other: Tensor) -> Tensor:
        """
        ne(other) -> Tensor

        See :func:`torch.ne`.
        """
    @overload
    def ne(self, other: Number | _complex) -> Tensor:
        """
        ne(other) -> Tensor

        See :func:`torch.ne`.
        """
    @overload
    def ne_(self, other: Tensor) -> Tensor:
        """
        ne_(other) -> Tensor

        In-place version of :meth:`~Tensor.ne`.
        """
    @overload
    def ne_(self, other: Number | _complex) -> Tensor:
        """
        ne_(other) -> Tensor

        In-place version of :meth:`~Tensor.ne`.
        """
    def neg(self) -> Tensor:
        """
        neg() -> Tensor

        See :func:`torch.neg`
        """
    def neg_(self) -> Tensor:
        """
        neg_() -> Tensor

        In-place version of :meth:`~Tensor.neg`
        """
    def negative(self) -> Tensor:
        """
        negative() -> Tensor

        See :func:`torch.negative`
        """
    def negative_(self) -> Tensor:
        """
        negative_() -> Tensor

        In-place version of :meth:`~Tensor.negative`
        """
    def nelement(self) -> _int:
        """
        nelement() -> int

        Alias for :meth:`~Tensor.numel`
        """
    @overload
    def new(self, *args: Any, device: DeviceLikeType | None = ...) -> Self: ...
    @overload
    def new(self, storage: Storage) -> Self: ...
    @overload
    def new(self, other: Tensor) -> Self: ...
    @overload
    def new(self, size: _size, *, device: DeviceLikeType | None = ...) -> Self: ...
    @overload
    def new_empty(
        self,
        size: Sequence[_int | SymInt],
        *,
        dtype: _dtype | None = ...,
        layout: _layout | None = ...,
        device: DeviceLikeType | None = ...,
        pin_memory: _bool | None = ...,
        requires_grad: _bool | None = ...,
    ) -> Tensor:
        """
        new_empty(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a Tensor of size :attr:`size` filled with uninitialized data.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
                shape of the output tensor.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.ones(())
            >>> tensor.new_empty((2, 3))
            tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
                    [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])
        """
    @overload
    def new_empty(
        self,
        *size: _int | SymInt,
        dtype: _dtype | None = ...,
        layout: _layout | None = ...,
        device: DeviceLikeType | None = ...,
        pin_memory: _bool | None = ...,
        requires_grad: _bool | None = ...,
    ) -> Tensor:
        """
        new_empty(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a Tensor of size :attr:`size` filled with uninitialized data.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
                shape of the output tensor.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.ones(())
            >>> tensor.new_empty((2, 3))
            tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
                    [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])
        """
    def new_empty_strided(
        self,
        size: Sequence[_int | SymInt],
        stride: Sequence[_int | SymInt],
        *,
        dtype: _dtype | None = ...,
        layout: _layout | None = ...,
        device: DeviceLikeType | None = ...,
        pin_memory: _bool | None = ...,
        requires_grad: _bool | None = ...,
    ) -> Tensor:
        """
        new_empty_strided(size, stride, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a Tensor of size :attr:`size` and strides :attr:`stride` filled with
        uninitialized data. By default, the returned Tensor has the same
        :class:`torch.dtype` and :class:`torch.device` as this tensor.

        Args:
            size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
                shape of the output tensor.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.ones(())
            >>> tensor.new_empty_strided((2, 3), (3, 1))
            tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
                    [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])
        """
    def new_full(
        self,
        size: Sequence[_int | SymInt],
        fill_value: Number | _complex,
        *,
        dtype: _dtype | None = ...,
        layout: _layout | None = ...,
        device: DeviceLikeType | None = ...,
        pin_memory: _bool | None = ...,
        requires_grad: _bool | None = ...,
    ) -> Tensor:
        """
        new_full(size, fill_value, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a Tensor of size :attr:`size` filled with :attr:`fill_value`.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            fill_value (scalar): the number to fill the output tensor with.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.ones((2,), dtype=torch.float64)
            >>> tensor.new_full((3, 4), 3.141592)
            tensor([[ 3.1416,  3.1416,  3.1416,  3.1416],
                    [ 3.1416,  3.1416,  3.1416,  3.1416],
                    [ 3.1416,  3.1416,  3.1416,  3.1416]], dtype=torch.float64)
        """
    @overload
    def new_ones(
        self,
        size: _size,
        dtype: _dtype | None = ...,
        device: DeviceLikeType | None = ...,
        requires_grad: _bool = ...,
        pin_memory: _bool = ...,
    ) -> Tensor:
        """
        new_ones(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a Tensor of size :attr:`size` filled with ``1``.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
                shape of the output tensor.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.tensor((), dtype=torch.int32)
            >>> tensor.new_ones((2, 3))
            tensor([[ 1,  1,  1],
                    [ 1,  1,  1]], dtype=torch.int32)
        """
    @overload
    def new_ones(
        self,
        size: Sequence[_int | SymInt],
        *,
        dtype: _dtype | None = ...,
        layout: _layout | None = ...,
        device: DeviceLikeType | None = ...,
        pin_memory: _bool | None = ...,
        requires_grad: _bool | None = ...,
    ) -> Tensor:
        """
        new_ones(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a Tensor of size :attr:`size` filled with ``1``.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
                shape of the output tensor.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.tensor((), dtype=torch.int32)
            >>> tensor.new_ones((2, 3))
            tensor([[ 1,  1,  1],
                    [ 1,  1,  1]], dtype=torch.int32)
        """
    @overload
    def new_ones(
        self,
        *size: _int | SymInt,
        dtype: _dtype | None = ...,
        layout: _layout | None = ...,
        device: DeviceLikeType | None = ...,
        pin_memory: _bool | None = ...,
        requires_grad: _bool | None = ...,
    ) -> Tensor:
        """
        new_ones(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a Tensor of size :attr:`size` filled with ``1``.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
                shape of the output tensor.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.tensor((), dtype=torch.int32)
            >>> tensor.new_ones((2, 3))
            tensor([[ 1,  1,  1],
                    [ 1,  1,  1]], dtype=torch.int32)
        """
    def new_tensor(
        self,
        data: Any,
        dtype: _dtype | None = ...,
        device: DeviceLikeType | None = ...,
        requires_grad: _bool = ...,
        pin_memory: _bool = ...,
    ) -> Tensor:
        """
        new_tensor(data, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a new Tensor with :attr:`data` as the tensor data.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        .. warning::

            :func:`new_tensor` always copies :attr:`data`. If you have a Tensor
            ``data`` and want to avoid a copy, use :func:`torch.Tensor.requires_grad_`
            or :func:`torch.Tensor.detach`.
            If you have a numpy array and want to avoid a copy, use
            :func:`torch.from_numpy`.

        .. warning::

            When data is a tensor `x`, :func:`new_tensor()` reads out 'the data' from whatever it is passed,
            and constructs a leaf variable. Therefore ``tensor.new_tensor(x)`` is equivalent to ``x.detach().clone()``
            and ``tensor.new_tensor(x, requires_grad=True)`` is equivalent to ``x.detach().clone().requires_grad_(True)``.
            The equivalents using ``detach()`` and ``clone()`` are recommended.

        Args:
            data (array_like): The returned Tensor copies :attr:`data`.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.ones((2,), dtype=torch.int8)
            >>> data = [[0, 1], [2, 3]]
            >>> tensor.new_tensor(data)
            tensor([[ 0,  1],
                    [ 2,  3]], dtype=torch.int8)
        """
    @overload
    def new_zeros(
        self,
        size: Sequence[_int | SymInt],
        *,
        dtype: _dtype | None = ...,
        layout: _layout | None = ...,
        device: DeviceLikeType | None = ...,
        pin_memory: _bool | None = ...,
        requires_grad: _bool | None = ...,
    ) -> Tensor:
        """
        new_zeros(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a Tensor of size :attr:`size` filled with ``0``.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
                shape of the output tensor.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.tensor((), dtype=torch.float64)
            >>> tensor.new_zeros((2, 3))
            tensor([[ 0.,  0.,  0.],
                    [ 0.,  0.,  0.]], dtype=torch.float64)
        """
    @overload
    def new_zeros(
        self,
        *size: _int | SymInt,
        dtype: _dtype | None = ...,
        layout: _layout | None = ...,
        device: DeviceLikeType | None = ...,
        pin_memory: _bool | None = ...,
        requires_grad: _bool | None = ...,
    ) -> Tensor:
        """
        new_zeros(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False) -> Tensor


        Returns a Tensor of size :attr:`size` filled with ``0``.
        By default, the returned Tensor has the same :class:`torch.dtype` and
        :class:`torch.device` as this tensor.

        Args:
            size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
                shape of the output tensor.

        Keyword args:
            dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
                Default: if None, same :class:`torch.dtype` as this tensor.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if None, same :class:`torch.device` as this tensor.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            pin_memory (bool, optional): If set, returned tensor would be allocated in
                the pinned memory. Works only for CPU tensors. Default: ``False``.

        Example::

            >>> tensor = torch.tensor((), dtype=torch.float64)
            >>> tensor.new_zeros((2, 3))
            tensor([[ 0.,  0.,  0.],
                    [ 0.,  0.,  0.]], dtype=torch.float64)
        """
    def nextafter(self, other: Tensor) -> Tensor:
        """
        nextafter(other) -> Tensor
        See :func:`torch.nextafter`
        """
    def nextafter_(self, other: Tensor) -> Tensor:
        """
        nextafter_(other) -> Tensor
        In-place version of :meth:`~Tensor.nextafter`
        """
    @overload
    def nonzero(self, *, as_tuple: Literal[False] = ...) -> Tensor:
        """
        nonzero() -> LongTensor

        See :func:`torch.nonzero`
        """
    @overload
    def nonzero(self, *, as_tuple: Literal[True]) -> tuple[Tensor, ...]:
        """
        nonzero() -> LongTensor

        See :func:`torch.nonzero`
        """
    def nonzero_static(self, *, size: _int | SymInt, fill_value: _int = ...) -> Tensor:
        """
        nonzero_static(input, *, size, fill_value=-1) -> Tensor

        Returns a 2-D tensor where each row is the index for a non-zero value.
        The returned Tensor has the same `torch.dtype` as `torch.nonzero()`.

        Args:
            input (Tensor): the input tensor to count non-zero elements.

        Keyword args:
            size (int): the size of non-zero elements expected to be included in the out
                tensor. Pad the out tensor with `fill_value` if the `size` is larger
                than total number of non-zero elements, truncate out tensor if `size`
                is smaller. The size must be a non-negative integer.
            fill_value (int, optional): the value to fill the output tensor with when `size` is larger
                than the total number of non-zero elements. Default is `-1` to represent
                invalid index.

        Example:

            # Example 1: Padding
            >>> input_tensor = torch.tensor([[1, 0], [3, 2]])
            >>> static_size = 4
            >>> t = torch.nonzero_static(input_tensor, size=static_size)
            tensor([[  0,   0],
                    [  1,   0],
                    [  1,   1],
                    [  -1, -1]], dtype=torch.int64)

            # Example 2: Truncating
            >>> input_tensor = torch.tensor([[1, 0], [3, 2]])
            >>> static_size = 2
            >>> t = torch.nonzero_static(input_tensor, size=static_size)
            tensor([[  0,   0],
                    [  1,   0]], dtype=torch.int64)

            # Example 3: 0 size
            >>> input_tensor = torch.tensor([10])
            >>> static_size = 0
            >>> t = torch.nonzero_static(input_tensor, size=static_size)
            tensor([], size=(0, 1), dtype=torch.int64)

            # Example 4: 0 rank input
            >>> input_tensor = torch.tensor(10)
            >>> static_size = 2
            >>> t = torch.nonzero_static(input_tensor, size=static_size)
            tensor([], size=(2, 0), dtype=torch.int64)
        """
    def normal_(self, mean: _float = ..., std: _float = ..., *, generator: Generator | None = ...) -> Tensor:
        """
        normal_(mean=0, std=1, *, generator=None) -> Tensor

        Fills :attr:`self` tensor with elements samples from the normal distribution
        parameterized by :attr:`mean` and :attr:`std`.
        """
    @overload
    def not_equal(self, other: Tensor) -> Tensor:
        """
        not_equal(other) -> Tensor

        See :func:`torch.not_equal`.
        """
    @overload
    def not_equal(self, other: Number | _complex) -> Tensor:
        """
        not_equal(other) -> Tensor

        See :func:`torch.not_equal`.
        """
    @overload
    def not_equal_(self, other: Tensor) -> Tensor:
        """
        not_equal_(other) -> Tensor

        In-place version of :meth:`~Tensor.not_equal`.
        """
    @overload
    def not_equal_(self, other: Number | _complex) -> Tensor:
        """
        not_equal_(other) -> Tensor

        In-place version of :meth:`~Tensor.not_equal`.
        """
    def numel(self) -> _int:
        """
        numel() -> int

        See :func:`torch.numel`
        """
    def numpy(self, *, force: _bool = ...) -> np.ndarray:
        """
        numpy(*, force=False) -> numpy.ndarray

        Returns the tensor as a NumPy :class:`ndarray`.

        If :attr:`force` is ``False`` (the default), the conversion
        is performed only if the tensor is on the CPU, does not require grad,
        does not have its conjugate bit set, and is a dtype and layout that
        NumPy supports. The returned ndarray and the tensor will share their
        storage, so changes to the tensor will be reflected in the ndarray
        and vice versa.

        If :attr:`force` is ``True`` this is equivalent to
        calling ``t.detach().cpu().resolve_conj().resolve_neg().numpy()``.
        If the tensor isn't on the CPU or the conjugate or negative bit is set,
        the tensor won't share its storage with the returned ndarray.
        Setting :attr:`force` to ``True`` can be a useful shorthand.

        Args:
            force (bool): if ``True``, the ndarray may be a copy of the tensor
                       instead of always sharing memory, defaults to ``False``.
        """
    def orgqr(self, input2: Tensor) -> Tensor:
        """
        orgqr(input2) -> Tensor

        See :func:`torch.orgqr`
        """
    def ormqr(self, input2: Tensor, input3: Tensor, left: _bool = ..., transpose: _bool = ...) -> Tensor:
        """
        ormqr(input2, input3, left=True, transpose=False) -> Tensor

        See :func:`torch.ormqr`
        """
    def outer(self, vec2: Tensor) -> Tensor:
        """
        outer(vec2) -> Tensor

        See :func:`torch.outer`.
        """
    @overload
    def permute(self, dims: _size) -> Tensor:
        """
        permute(*dims) -> Tensor

        See :func:`torch.permute`
        """
    @overload
    def permute(self, *dims: _int) -> Tensor:
        """
        permute(*dims) -> Tensor

        See :func:`torch.permute`
        """
    def pin_memory(self, device: DeviceLikeType | None = ...) -> Tensor:
        """
        pin_memory() -> Tensor

        Copies the tensor to pinned memory, if it's not already pinned.
        By default, the device pinned memory on will be the current :ref:`accelerator<accelerators>`.
        """
    def pinverse(self, rcond: _float = ...) -> Tensor:
        """
        pinverse() -> Tensor

        See :func:`torch.pinverse`
        """
    def polygamma(self, n: _int) -> Tensor:
        """
        polygamma(n) -> Tensor

        See :func:`torch.polygamma`
        """
    def polygamma_(self, n: _int) -> Tensor:
        """
        polygamma_(n) -> Tensor

        In-place version of :meth:`~Tensor.polygamma`
        """
    def positive(self) -> Tensor:
        """
        positive() -> Tensor

        See :func:`torch.positive`
        """
    @overload
    def pow(self, exponent: Tensor) -> Tensor:
        """
        pow(exponent) -> Tensor

        See :func:`torch.pow`
        """
    @overload
    def pow(self, exponent: Number | _complex) -> Tensor:
        """
        pow(exponent) -> Tensor

        See :func:`torch.pow`
        """
    @overload
    def pow_(self, exponent: Tensor) -> Tensor:
        """
        pow_(exponent) -> Tensor

        In-place version of :meth:`~Tensor.pow`
        """
    @overload
    def pow_(self, exponent: Number | _complex) -> Tensor:
        """
        pow_(exponent) -> Tensor

        In-place version of :meth:`~Tensor.pow`
        """
    def prelu(self, weight: Tensor) -> Tensor: ...
    @overload
    def prod(self, *, dtype: _dtype | None = ...) -> Tensor:
        """
        prod(dim=None, keepdim=False, dtype=None) -> Tensor

        See :func:`torch.prod`
        """
    @overload
    def prod(self, dim: _int, keepdim: _bool = ..., *, dtype: _dtype | None = ...) -> Tensor:
        """
        prod(dim=None, keepdim=False, dtype=None) -> Tensor

        See :func:`torch.prod`
        """
    @overload
    def prod(self, dim: str | EllipsisType | None, keepdim: _bool = ..., *, dtype: _dtype | None = ...) -> Tensor:
        """
        prod(dim=None, keepdim=False, dtype=None) -> Tensor

        See :func:`torch.prod`
        """
    def put(self, index: Tensor, source: Tensor, accumulate: _bool = ...) -> Tensor:
        """
        put(input, index, source, accumulate=False) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.put_`.
        `input` corresponds to `self` in :meth:`torch.Tensor.put_`.
        """
    def put_(self, index: Tensor, source: Tensor, accumulate: _bool = ...) -> Tensor:
        """
        put_(index, source, accumulate=False) -> Tensor

        Copies the elements from :attr:`source` into the positions specified by
        :attr:`index`. For the purpose of indexing, the :attr:`self` tensor is treated as if
        it were a 1-D tensor.

        :attr:`index` and :attr:`source` need to have the same number of elements, but not necessarily
        the same shape.

        If :attr:`accumulate` is ``True``, the elements in :attr:`source` are added to
        :attr:`self`. If accumulate is ``False``, the behavior is undefined if :attr:`index`
        contain duplicate elements.

        Args:
            index (LongTensor): the indices into self
            source (Tensor): the tensor containing values to copy from
            accumulate (bool, optional): whether to accumulate into self. Default: ``False``

        Example::

            >>> src = torch.tensor([[4, 3, 5],
            ...                     [6, 7, 8]])
            >>> src.put_(torch.tensor([1, 3]), torch.tensor([9, 10]))
            tensor([[  4,   9,   5],
                    [ 10,   7,   8]])
        """
    def q_per_channel_axis(self) -> _int:
        """
        q_per_channel_axis() -> int

        Given a Tensor quantized by linear (affine) per-channel quantization,
        returns the index of dimension on which per-channel quantization is applied.
        """
    def q_per_channel_scales(self) -> Tensor:
        """
        q_per_channel_scales() -> Tensor

        Given a Tensor quantized by linear (affine) per-channel quantization,
        returns a Tensor of scales of the underlying quantizer. It has the number of
        elements that matches the corresponding dimensions (from q_per_channel_axis) of
        the tensor.
        """
    def q_per_channel_zero_points(self) -> Tensor:
        """
        q_per_channel_zero_points() -> Tensor

        Given a Tensor quantized by linear (affine) per-channel quantization,
        returns a tensor of zero_points of the underlying quantizer. It has the number of
        elements that matches the corresponding dimensions (from q_per_channel_axis) of
        the tensor.
        """
    def q_scale(self) -> _float:
        """
        q_scale() -> float

        Given a Tensor quantized by linear(affine) quantization,
        returns the scale of the underlying quantizer().
        """
    def q_zero_point(self) -> _int:
        """
        q_zero_point() -> int

        Given a Tensor quantized by linear(affine) quantization,
        returns the zero_point of the underlying quantizer().
        """
    def qr(self, some: _bool = ...) -> torch.return_types.qr:
        """
        qr(some=True) -> (Tensor, Tensor)

        See :func:`torch.qr`
        """
    def qscheme(self) -> _qscheme:
        """
        qscheme() -> torch.qscheme

        Returns the quantization scheme of a given QTensor.
        """
    @overload
    def quantile(self, q: Tensor, dim: _int | None = ..., keepdim: _bool = ..., *, interpolation: str = ...) -> Tensor:
        """
        quantile(q, dim=None, keepdim=False, *, interpolation='linear') -> Tensor

        See :func:`torch.quantile`
        """
    @overload
    def quantile(self, q: _float, dim: _int | None = ..., keepdim: _bool = ..., *, interpolation: str = ...) -> Tensor:
        """
        quantile(q, dim=None, keepdim=False, *, interpolation='linear') -> Tensor

        See :func:`torch.quantile`
        """
    def rad2deg(self) -> Tensor:
        """
        rad2deg() -> Tensor

        See :func:`torch.rad2deg`
        """
    def rad2deg_(self) -> Tensor:
        """
        rad2deg_() -> Tensor

        In-place version of :meth:`~Tensor.rad2deg`
        """
    @overload
    def random_(self, *, generator: Generator | None = ...) -> Tensor:
        """
        random_(from=0, to=None, *, generator=None) -> Tensor

        Fills :attr:`self` tensor with numbers sampled from the discrete uniform
        distribution over ``[from, to - 1]``. If not specified, the values are usually
        only bounded by :attr:`self` tensor's data type. However, for floating point
        types, if unspecified, range will be ``[0, 2^mantissa]`` to ensure that every
        value is representable. For example, `torch.tensor(1, dtype=torch.double).random_()`
        will be uniform in ``[0, 2^53]``.
        """
    @overload
    def random_(self, from_: _int, to: _int | None, *, generator: Generator | None = ...) -> Tensor:
        """
        random_(from=0, to=None, *, generator=None) -> Tensor

        Fills :attr:`self` tensor with numbers sampled from the discrete uniform
        distribution over ``[from, to - 1]``. If not specified, the values are usually
        only bounded by :attr:`self` tensor's data type. However, for floating point
        types, if unspecified, range will be ``[0, 2^mantissa]`` to ensure that every
        value is representable. For example, `torch.tensor(1, dtype=torch.double).random_()`
        will be uniform in ``[0, 2^53]``.
        """
    @overload
    def random_(self, to: _int, *, generator: Generator | None = ...) -> Tensor:
        """
        random_(from=0, to=None, *, generator=None) -> Tensor

        Fills :attr:`self` tensor with numbers sampled from the discrete uniform
        distribution over ``[from, to - 1]``. If not specified, the values are usually
        only bounded by :attr:`self` tensor's data type. However, for floating point
        types, if unspecified, range will be ``[0, 2^mantissa]`` to ensure that every
        value is representable. For example, `torch.tensor(1, dtype=torch.double).random_()`
        will be uniform in ``[0, 2^53]``.
        """
    def ravel(self) -> Tensor:
        """
        ravel() -> Tensor

        see :func:`torch.ravel`
        """
    def reciprocal(self) -> Tensor:
        """
        reciprocal() -> Tensor

        See :func:`torch.reciprocal`
        """
    def reciprocal_(self) -> Tensor:
        """
        reciprocal_() -> Tensor

        In-place version of :meth:`~Tensor.reciprocal`
        """
    def record_stream(self, s: Stream) -> None:
        """
        record_stream(stream)

        Marks the tensor as having been used by this stream.  When the tensor
        is deallocated, ensure the tensor memory is not reused for another tensor
        until all work queued on :attr:`stream` at the time of deallocation is
        complete.

        .. note::

            The caching allocator is aware of only the stream where a tensor was
            allocated. Due to the awareness, it already correctly manages the life
            cycle of tensors on only one stream. But if a tensor is used on a stream
            different from the stream of origin, the allocator might reuse the memory
            unexpectedly. Calling this method lets the allocator know which streams
            have used the tensor.

        .. warning::

            This method is most suitable for use cases where you are providing a
            function that created a tensor on a side stream, and want users to be able
            to make use of the tensor without having to think carefully about stream
            safety when making use of them.  These safety guarantees come at some
            performance and predictability cost (analogous to the tradeoff between GC
            and manual memory management), so if you are in a situation where
            you manage the full lifetime of your tensors, you may consider instead
            manually managing CUDA events so that calling this method is not necessary.
            In particular, when you call this method, on later allocations the
            allocator will poll the recorded stream to see if all operations have
            completed yet; you can potentially race with side stream computation and
            non-deterministically reuse or fail to reuse memory for an allocation.

            You can safely use tensors allocated on side streams without
            :meth:`~Tensor.record_stream`; you must manually ensure that
            any non-creation stream uses of a tensor are synced back to the creation
            stream before you deallocate the tensor.  As the CUDA caching allocator
            guarantees that the memory will only be reused with the same creation stream,
            this is sufficient to ensure that writes to future reallocations of the
            memory will be delayed until non-creation stream uses are done.
            (Counterintuitively, you may observe that on the CPU side we have already
            reallocated the tensor, even though CUDA kernels on the old tensor are
            still in progress.  This is fine, because CUDA operations on the new
            tensor will appropriately wait for the old operations to complete, as they
            are all on the same stream.)

            Concretely, this looks like this::

                with torch.cuda.stream(s0):
                    x = torch.zeros(N)

                s1.wait_stream(s0)
                with torch.cuda.stream(s1):
                    y = some_comm_op(x)

                ... some compute on s0 ...

                # synchronize creation stream s0 to side stream s1
                # before deallocating x
                s0.wait_stream(s1)
                del x

            Note that some discretion is required when deciding when to perform
            ``s0.wait_stream(s1)``.  In particular, if we were to wait immediately
            after ``some_comm_op``, there wouldn't be any point in having the side
            stream; it would be equivalent to have run ``some_comm_op`` on ``s0``.
            Instead, the synchronization must be placed at some appropriate, later
            point in time where you expect the side stream ``s1`` to have finished
            work.  This location is typically identified via profiling, e.g., using
            Chrome traces produced
            :meth:`torch.autograd.profiler.profile.export_chrome_trace`.  If you
            place the wait too early, work on s0 will block until ``s1`` has finished,
            preventing further overlapping of communication and computation.  If you
            place the wait too late, you will use more memory than is strictly
            necessary (as you are keeping ``x`` live for longer.)  For a concrete
            example of how this guidance can be applied in practice, see this post:
            `FSDP and CUDACachingAllocator
            <https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486>`_.
        """
    def refine_names(self, names: Sequence[str | EllipsisType | None]) -> Tensor: ...
    def relu(self) -> Tensor: ...
    def relu_(self) -> Tensor: ...
    @overload
    def remainder(self, other: Tensor) -> Tensor:
        """
        remainder(divisor) -> Tensor

        See :func:`torch.remainder`
        """
    @overload
    def remainder(self, other: Number | _complex) -> Tensor:
        """
        remainder(divisor) -> Tensor

        See :func:`torch.remainder`
        """
    @overload
    def remainder_(self, other: Tensor) -> Tensor:
        """
        remainder_(divisor) -> Tensor

        In-place version of :meth:`~Tensor.remainder`
        """
    @overload
    def remainder_(self, other: Number | _complex) -> Tensor:
        """
        remainder_(divisor) -> Tensor

        In-place version of :meth:`~Tensor.remainder`
        """
    def rename(self, names: Sequence[str | EllipsisType | None] | None) -> Tensor: ...
    def rename_(self, names: Sequence[str | EllipsisType | None] | None) -> Tensor: ...
    def renorm(self, p: Number | _complex, dim: _int, maxnorm: Number | _complex) -> Tensor:
        """
        renorm(p, dim, maxnorm) -> Tensor

        See :func:`torch.renorm`
        """
    def renorm_(self, p: Number | _complex, dim: _int, maxnorm: Number | _complex) -> Tensor:
        """
        renorm_(p, dim, maxnorm) -> Tensor

        In-place version of :meth:`~Tensor.renorm`
        """
    @overload
    def repeat(self, repeats: Sequence[_int | SymInt]) -> Tensor:
        """
        repeat(*repeats) -> Tensor

        Repeats this tensor along the specified dimensions.

        Unlike :meth:`~Tensor.expand`, this function copies the tensor's data.

        .. warning::

            :meth:`~Tensor.repeat` behaves differently from
            `numpy.repeat <https://numpy.org/doc/stable/reference/generated/numpy.repeat.html>`_,
            but is more similar to
            `numpy.tile <https://numpy.org/doc/stable/reference/generated/numpy.tile.html>`_.
            For the operator similar to `numpy.repeat`, see :func:`torch.repeat_interleave`.

        Args:
            repeat (torch.Size, int..., tuple of int or list of int): The number of times to repeat this tensor along each dimension

        Example::

            >>> x = torch.tensor([1, 2, 3])
            >>> x.repeat(4, 2)
            tensor([[ 1,  2,  3,  1,  2,  3],
                    [ 1,  2,  3,  1,  2,  3],
                    [ 1,  2,  3,  1,  2,  3],
                    [ 1,  2,  3,  1,  2,  3]])
            >>> x.repeat(4, 2, 1).size()
            torch.Size([4, 2, 3])
        """
    @overload
    def repeat(self, *repeats: _int | SymInt) -> Tensor:
        """
        repeat(*repeats) -> Tensor

        Repeats this tensor along the specified dimensions.

        Unlike :meth:`~Tensor.expand`, this function copies the tensor's data.

        .. warning::

            :meth:`~Tensor.repeat` behaves differently from
            `numpy.repeat <https://numpy.org/doc/stable/reference/generated/numpy.repeat.html>`_,
            but is more similar to
            `numpy.tile <https://numpy.org/doc/stable/reference/generated/numpy.tile.html>`_.
            For the operator similar to `numpy.repeat`, see :func:`torch.repeat_interleave`.

        Args:
            repeat (torch.Size, int..., tuple of int or list of int): The number of times to repeat this tensor along each dimension

        Example::

            >>> x = torch.tensor([1, 2, 3])
            >>> x.repeat(4, 2)
            tensor([[ 1,  2,  3,  1,  2,  3],
                    [ 1,  2,  3,  1,  2,  3],
                    [ 1,  2,  3,  1,  2,  3],
                    [ 1,  2,  3,  1,  2,  3]])
            >>> x.repeat(4, 2, 1).size()
            torch.Size([4, 2, 3])
        """
    @overload
    def repeat_interleave(
        self, repeats: Tensor, dim: _int | None = ..., *, output_size: _int | SymInt | None = ...
    ) -> Tensor:
        """
        repeat_interleave(repeats, dim=None, *, output_size=None) -> Tensor

        See :func:`torch.repeat_interleave`.
        """
    @overload
    def repeat_interleave(
        self, repeats: _int | SymInt, dim: _int | None = ..., *, output_size: _int | SymInt | None = ...
    ) -> Tensor:
        """
        repeat_interleave(repeats, dim=None, *, output_size=None) -> Tensor

        See :func:`torch.repeat_interleave`.
        """
    def requires_grad_(self, mode: _bool = ...) -> Tensor:
        """
        requires_grad_(requires_grad=True) -> Tensor

        Change if autograd should record operations on this tensor: sets this tensor's
        :attr:`requires_grad` attribute in-place. Returns this tensor.

        :func:`requires_grad_`'s main use case is to tell autograd to begin recording
        operations on a Tensor ``tensor``. If ``tensor`` has ``requires_grad=False``
        (because it was obtained through a DataLoader, or required preprocessing or
        initialization), ``tensor.requires_grad_()`` makes it so that autograd will
        begin to record operations on ``tensor``.

        Args:
            requires_grad (bool): If autograd should record operations on this tensor.
                Default: ``True``.

        Example::

            >>> # Let's say we want to preprocess some saved weights and use
            >>> # the result as new weights.
            >>> saved_weights = [0.1, 0.2, 0.3, 0.25]
            >>> loaded_weights = torch.tensor(saved_weights)
            >>> weights = preprocess(loaded_weights)  # some function
            >>> weights
            tensor([-0.5503,  0.4926, -2.1158, -0.8303])

            >>> # Now, start to record operations done to weights
            >>> weights.requires_grad_()
            >>> out = weights.pow(2).sum()
            >>> out.backward()
            >>> weights.grad
            tensor([-1.1007,  0.9853, -4.2316, -1.6606])
        """
    @overload
    def reshape(self, shape: Sequence[_int | SymInt]) -> Tensor:
        """
        reshape(*shape) -> Tensor

        Returns a tensor with the same data and number of elements as :attr:`self`
        but with the specified shape. This method returns a view if :attr:`shape` is
        compatible with the current shape. See :meth:`torch.Tensor.view` on when it is
        possible to return a view.

        See :func:`torch.reshape`

        Args:
            shape (tuple of ints or int...): the desired shape
        """
    @overload
    def reshape(self, *shape: _int | SymInt) -> Tensor:
        """
        reshape(*shape) -> Tensor

        Returns a tensor with the same data and number of elements as :attr:`self`
        but with the specified shape. This method returns a view if :attr:`shape` is
        compatible with the current shape. See :meth:`torch.Tensor.view` on when it is
        possible to return a view.

        See :func:`torch.reshape`

        Args:
            shape (tuple of ints or int...): the desired shape
        """
    def reshape_as(self, other: Tensor) -> Tensor:
        """
        reshape_as(other) -> Tensor

        Returns this tensor as the same shape as :attr:`other`.
        ``self.reshape_as(other)`` is equivalent to ``self.reshape(other.sizes())``.
        This method returns a view if ``other.sizes()`` is compatible with the current
        shape. See :meth:`torch.Tensor.view` on when it is possible to return a view.

        Please see :meth:`reshape` for more information about ``reshape``.

        Args:
            other (:class:`torch.Tensor`): The result tensor has the same shape
                as :attr:`other`.
        """
    @overload
    def resize_(self, size: Sequence[_int | SymInt], *, memory_format: memory_format | None = ...) -> Tensor:
        """
        resize_(*sizes, memory_format=torch.contiguous_format) -> Tensor

        Resizes :attr:`self` tensor to the specified size. If the number of elements is
        larger than the current storage size, then the underlying storage is resized
        to fit the new number of elements. If the number of elements is smaller, the
        underlying storage is not changed. Existing elements are preserved but any new
        memory is uninitialized.

        .. warning::

            This is a low-level method. The storage is reinterpreted as C-contiguous,
            ignoring the current strides (unless the target size equals the current
            size, in which case the tensor is left unchanged). For most purposes, you
            will instead want to use :meth:`~Tensor.view()`, which checks for
            contiguity, or :meth:`~Tensor.reshape()`, which copies data if needed. To
            change the size in-place with custom strides, see :meth:`~Tensor.set_()`.

        .. note::

            If :func:`torch.use_deterministic_algorithms()` and
            :attr:`torch.utils.deterministic.fill_uninitialized_memory` are both set to
            ``True``, new elements are initialized to prevent nondeterministic behavior
            from using the result as an input to an operation. Floating point and
            complex values are set to NaN, and integer values are set to the maximum
            value.

        Args:
            sizes (torch.Size or int...): the desired size
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                Tensor. Default: ``torch.contiguous_format``. Note that memory format of
                :attr:`self` is going to be unaffected if ``self.size()`` matches ``sizes``.

        Example::

            >>> x = torch.tensor([[1, 2], [3, 4], [5, 6]])
            >>> x.resize_(2, 2)
            tensor([[ 1,  2],
                    [ 3,  4]])
        """
    @overload
    def resize_(self, *size: _int | SymInt, memory_format: memory_format | None = ...) -> Tensor:
        """
        resize_(*sizes, memory_format=torch.contiguous_format) -> Tensor

        Resizes :attr:`self` tensor to the specified size. If the number of elements is
        larger than the current storage size, then the underlying storage is resized
        to fit the new number of elements. If the number of elements is smaller, the
        underlying storage is not changed. Existing elements are preserved but any new
        memory is uninitialized.

        .. warning::

            This is a low-level method. The storage is reinterpreted as C-contiguous,
            ignoring the current strides (unless the target size equals the current
            size, in which case the tensor is left unchanged). For most purposes, you
            will instead want to use :meth:`~Tensor.view()`, which checks for
            contiguity, or :meth:`~Tensor.reshape()`, which copies data if needed. To
            change the size in-place with custom strides, see :meth:`~Tensor.set_()`.

        .. note::

            If :func:`torch.use_deterministic_algorithms()` and
            :attr:`torch.utils.deterministic.fill_uninitialized_memory` are both set to
            ``True``, new elements are initialized to prevent nondeterministic behavior
            from using the result as an input to an operation. Floating point and
            complex values are set to NaN, and integer values are set to the maximum
            value.

        Args:
            sizes (torch.Size or int...): the desired size
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                Tensor. Default: ``torch.contiguous_format``. Note that memory format of
                :attr:`self` is going to be unaffected if ``self.size()`` matches ``sizes``.

        Example::

            >>> x = torch.tensor([[1, 2], [3, 4], [5, 6]])
            >>> x.resize_(2, 2)
            tensor([[ 1,  2],
                    [ 3,  4]])
        """
    def resize_as_(self, the_template: Tensor, *, memory_format: memory_format | None = ...) -> Tensor:
        """
        resize_as_(tensor, memory_format=torch.contiguous_format) -> Tensor

        Resizes the :attr:`self` tensor to be the same size as the specified
        :attr:`tensor`. This is equivalent to ``self.resize_(tensor.size())``.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                Tensor. Default: ``torch.contiguous_format``. Note that memory format of
                :attr:`self` is going to be unaffected if ``self.size()`` matches ``tensor.size()``.
        """
    def resize_as_sparse_(self, the_template: Tensor) -> Tensor: ...
    def resolve_conj(self) -> Tensor:
        """
        resolve_conj() -> Tensor

        See :func:`torch.resolve_conj`
        """
    def resolve_neg(self) -> Tensor:
        """
        resolve_neg() -> Tensor

        See :func:`torch.resolve_neg`
        """
    def retain_grad(self) -> None:
        """
        retain_grad() -> None

        Enables this Tensor to have their :attr:`grad` populated during
        :func:`backward`. This is a no-op for leaf tensors.
        """
    def roll(self, shifts: _int | SymInt | Sequence[_int | SymInt], dims: _int | _size = ...) -> Tensor:
        """
        roll(shifts, dims) -> Tensor

        See :func:`torch.roll`
        """
    def rot90(self, k: _int = ..., dims: _size = ...) -> Tensor:
        """
        rot90(k, dims) -> Tensor

        See :func:`torch.rot90`
        """
    @overload
    def round(self) -> Tensor:
        """
        round(decimals=0) -> Tensor

        See :func:`torch.round`
        """
    @overload
    def round(self, *, decimals: _int) -> Tensor:
        """
        round(decimals=0) -> Tensor

        See :func:`torch.round`
        """
    @overload
    def round_(self) -> Tensor:
        """
        round_(decimals=0) -> Tensor

        In-place version of :meth:`~Tensor.round`
        """
    @overload
    def round_(self, *, decimals: _int) -> Tensor:
        """
        round_(decimals=0) -> Tensor

        In-place version of :meth:`~Tensor.round`
        """
    def row_indices(self) -> Tensor: ...
    def rsqrt(self) -> Tensor:
        """
        rsqrt() -> Tensor

        See :func:`torch.rsqrt`
        """
    def rsqrt_(self) -> Tensor:
        """
        rsqrt_() -> Tensor

        In-place version of :meth:`~Tensor.rsqrt`
        """
    @overload
    def scatter(self, dim: _int, index: Tensor, src: Tensor) -> Tensor:
        """
        scatter(dim, index, src) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_`
        """
    @overload
    def scatter(self, dim: _int, index: Tensor, src: Tensor, *, reduce: str) -> Tensor:
        """
        scatter(dim, index, src) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_`
        """
    @overload
    def scatter(self, dim: _int, index: Tensor, value: Number | _complex, *, reduce: str) -> Tensor:
        """
        scatter(dim, index, src) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_`
        """
    @overload
    def scatter(self, dim: str | EllipsisType | None, index: Tensor, src: Tensor) -> Tensor:
        """
        scatter(dim, index, src) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_`
        """
    @overload
    def scatter(self, dim: _int, index: Tensor, value: Number | _complex) -> Tensor:
        """
        scatter(dim, index, src) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_`
        """
    @overload
    def scatter(self, dim: str | EllipsisType | None, index: Tensor, value: Number | _complex) -> Tensor:
        """
        scatter(dim, index, src) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_`
        """
    @overload
    def scatter_(self, dim: _int, index: Tensor, src: Tensor) -> Tensor:
        """
        scatter_(dim, index, src, *, reduce=None) -> Tensor

        Writes all values from the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index` tensor. For each value in :attr:`src`, its output
        index is specified by its index in :attr:`src` for ``dimension != dim`` and by
        the corresponding value in :attr:`index` for ``dimension = dim``.

        For a 3-D tensor, :attr:`self` is updated as::

            self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

        This is the reverse operation of the manner described in :meth:`~Tensor.gather`.

        It is also required that
        ``index.size(d) <= src.size(d)`` for all dimensions ``d``, and that
        ``index.size(d) <= self.size(d)`` for all dimensions ``d != dim``.
        Note that ``input`` and ``index`` do not broadcast against each other for NPUs,
        so when running on NPUs, :attr:`input` and :attr:`index` must have the same number of dimensions.
        Standard broadcasting occurs in all other cases.

        Moreover, as for :meth:`~Tensor.gather`, the values of :attr:`index` must be
        between ``0`` and ``self.size(dim) - 1`` inclusive.

        .. warning::

            When indices are not unique, the behavior is non-deterministic (one of the
            values from ``src`` will be picked arbitrarily) and the gradient will be
            incorrect (it will be propagated to all locations in the source that
            correspond to the same index)!

        .. note::

            The backward pass is implemented only for ``src.shape == index.shape``.

        Additionally accepts an optional :attr:`reduce` argument that allows
        specification of an optional reduction operation, which is applied to all
        values in the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index`. For each value in :attr:`src`, the reduction
        operation is applied to an index in :attr:`self` which is specified by
        its index in :attr:`src` for ``dimension != dim`` and by the corresponding
        value in :attr:`index` for ``dimension = dim``.

        Given a 3-D tensor and reduction using the multiplication operation, :attr:`self`
        is updated as::

            self[index[i][j][k]][j][k] *= src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] *= src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] *= src[i][j][k]  # if dim == 2

        Reducing with the addition operation is the same as using
        :meth:`~torch.Tensor.scatter_add_`.

        .. warning::
            The reduce argument with Tensor ``src`` is deprecated and will be removed in
            a future PyTorch release. Please use :meth:`~torch.Tensor.scatter_reduce_`
            instead for more reduction options.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter, can be either empty
                or of the same dimensionality as ``src``. When empty, the operation
                returns ``self`` unchanged.
            src (Tensor): the source element(s) to scatter.

        Keyword args:
            reduce (str, optional): reduction operation to apply, can be either
                ``'add'`` or ``'multiply'``.

        Example::

            >>> src = torch.arange(1, 11).reshape((2, 5))
            >>> src
            tensor([[ 1,  2,  3,  4,  5],
                    [ 6,  7,  8,  9, 10]])
            >>> index = torch.tensor([[0, 1, 2, 0]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
            tensor([[1, 0, 0, 4, 0],
                    [0, 2, 0, 0, 0],
                    [0, 0, 3, 0, 0]])
            >>> index = torch.tensor([[0, 1, 2], [0, 1, 4]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
            tensor([[1, 2, 3, 0, 0],
                    [6, 7, 0, 0, 8],
                    [0, 0, 0, 0, 0]])

            >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
            ...            1.23, reduce='multiply')
            tensor([[2.0000, 2.0000, 2.4600, 2.0000],
                    [2.0000, 2.0000, 2.0000, 2.4600]])
            >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
            ...            1.23, reduce='add')
            tensor([[2.0000, 2.0000, 3.2300, 2.0000],
                    [2.0000, 2.0000, 2.0000, 3.2300]])

        .. function:: scatter_(dim, index, value, *, reduce=None) -> Tensor:
           :noindex:

        Writes the value from :attr:`value` into :attr:`self` at the indices
        specified in the :attr:`index` tensor.  This operation is equivalent to the previous version,
        with the :attr:`src` tensor filled entirely with :attr:`value`.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter, can be either empty
                or of the same dimensionality as ``src``. When empty, the operation
                returns ``self`` unchanged.
            value (Scalar): the value to scatter.

        Keyword args:
            reduce (str, optional): reduction operation to apply, can be either
                ``'add'`` or ``'multiply'``.

        Example::

            >>> index = torch.tensor([[0, 1]])
            >>> value = 2
            >>> torch.zeros(3, 5).scatter_(0, index, value)
            tensor([[2., 0., 0., 0., 0.],
                    [0., 2., 0., 0., 0.],
                    [0., 0., 0., 0., 0.]])
        """
    @overload
    def scatter_(self, dim: _int, index: Tensor, src: Tensor, *, reduce: str) -> Tensor:
        """
        scatter_(dim, index, src, *, reduce=None) -> Tensor

        Writes all values from the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index` tensor. For each value in :attr:`src`, its output
        index is specified by its index in :attr:`src` for ``dimension != dim`` and by
        the corresponding value in :attr:`index` for ``dimension = dim``.

        For a 3-D tensor, :attr:`self` is updated as::

            self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

        This is the reverse operation of the manner described in :meth:`~Tensor.gather`.

        It is also required that
        ``index.size(d) <= src.size(d)`` for all dimensions ``d``, and that
        ``index.size(d) <= self.size(d)`` for all dimensions ``d != dim``.
        Note that ``input`` and ``index`` do not broadcast against each other for NPUs,
        so when running on NPUs, :attr:`input` and :attr:`index` must have the same number of dimensions.
        Standard broadcasting occurs in all other cases.

        Moreover, as for :meth:`~Tensor.gather`, the values of :attr:`index` must be
        between ``0`` and ``self.size(dim) - 1`` inclusive.

        .. warning::

            When indices are not unique, the behavior is non-deterministic (one of the
            values from ``src`` will be picked arbitrarily) and the gradient will be
            incorrect (it will be propagated to all locations in the source that
            correspond to the same index)!

        .. note::

            The backward pass is implemented only for ``src.shape == index.shape``.

        Additionally accepts an optional :attr:`reduce` argument that allows
        specification of an optional reduction operation, which is applied to all
        values in the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index`. For each value in :attr:`src`, the reduction
        operation is applied to an index in :attr:`self` which is specified by
        its index in :attr:`src` for ``dimension != dim`` and by the corresponding
        value in :attr:`index` for ``dimension = dim``.

        Given a 3-D tensor and reduction using the multiplication operation, :attr:`self`
        is updated as::

            self[index[i][j][k]][j][k] *= src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] *= src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] *= src[i][j][k]  # if dim == 2

        Reducing with the addition operation is the same as using
        :meth:`~torch.Tensor.scatter_add_`.

        .. warning::
            The reduce argument with Tensor ``src`` is deprecated and will be removed in
            a future PyTorch release. Please use :meth:`~torch.Tensor.scatter_reduce_`
            instead for more reduction options.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter, can be either empty
                or of the same dimensionality as ``src``. When empty, the operation
                returns ``self`` unchanged.
            src (Tensor): the source element(s) to scatter.

        Keyword args:
            reduce (str, optional): reduction operation to apply, can be either
                ``'add'`` or ``'multiply'``.

        Example::

            >>> src = torch.arange(1, 11).reshape((2, 5))
            >>> src
            tensor([[ 1,  2,  3,  4,  5],
                    [ 6,  7,  8,  9, 10]])
            >>> index = torch.tensor([[0, 1, 2, 0]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
            tensor([[1, 0, 0, 4, 0],
                    [0, 2, 0, 0, 0],
                    [0, 0, 3, 0, 0]])
            >>> index = torch.tensor([[0, 1, 2], [0, 1, 4]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
            tensor([[1, 2, 3, 0, 0],
                    [6, 7, 0, 0, 8],
                    [0, 0, 0, 0, 0]])

            >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
            ...            1.23, reduce='multiply')
            tensor([[2.0000, 2.0000, 2.4600, 2.0000],
                    [2.0000, 2.0000, 2.0000, 2.4600]])
            >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
            ...            1.23, reduce='add')
            tensor([[2.0000, 2.0000, 3.2300, 2.0000],
                    [2.0000, 2.0000, 2.0000, 3.2300]])

        .. function:: scatter_(dim, index, value, *, reduce=None) -> Tensor:
           :noindex:

        Writes the value from :attr:`value` into :attr:`self` at the indices
        specified in the :attr:`index` tensor.  This operation is equivalent to the previous version,
        with the :attr:`src` tensor filled entirely with :attr:`value`.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter, can be either empty
                or of the same dimensionality as ``src``. When empty, the operation
                returns ``self`` unchanged.
            value (Scalar): the value to scatter.

        Keyword args:
            reduce (str, optional): reduction operation to apply, can be either
                ``'add'`` or ``'multiply'``.

        Example::

            >>> index = torch.tensor([[0, 1]])
            >>> value = 2
            >>> torch.zeros(3, 5).scatter_(0, index, value)
            tensor([[2., 0., 0., 0., 0.],
                    [0., 2., 0., 0., 0.],
                    [0., 0., 0., 0., 0.]])
        """
    @overload
    def scatter_(self, dim: _int, index: Tensor, value: Number | _complex, *, reduce: str) -> Tensor:
        """
        scatter_(dim, index, src, *, reduce=None) -> Tensor

        Writes all values from the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index` tensor. For each value in :attr:`src`, its output
        index is specified by its index in :attr:`src` for ``dimension != dim`` and by
        the corresponding value in :attr:`index` for ``dimension = dim``.

        For a 3-D tensor, :attr:`self` is updated as::

            self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

        This is the reverse operation of the manner described in :meth:`~Tensor.gather`.

        It is also required that
        ``index.size(d) <= src.size(d)`` for all dimensions ``d``, and that
        ``index.size(d) <= self.size(d)`` for all dimensions ``d != dim``.
        Note that ``input`` and ``index`` do not broadcast against each other for NPUs,
        so when running on NPUs, :attr:`input` and :attr:`index` must have the same number of dimensions.
        Standard broadcasting occurs in all other cases.

        Moreover, as for :meth:`~Tensor.gather`, the values of :attr:`index` must be
        between ``0`` and ``self.size(dim) - 1`` inclusive.

        .. warning::

            When indices are not unique, the behavior is non-deterministic (one of the
            values from ``src`` will be picked arbitrarily) and the gradient will be
            incorrect (it will be propagated to all locations in the source that
            correspond to the same index)!

        .. note::

            The backward pass is implemented only for ``src.shape == index.shape``.

        Additionally accepts an optional :attr:`reduce` argument that allows
        specification of an optional reduction operation, which is applied to all
        values in the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index`. For each value in :attr:`src`, the reduction
        operation is applied to an index in :attr:`self` which is specified by
        its index in :attr:`src` for ``dimension != dim`` and by the corresponding
        value in :attr:`index` for ``dimension = dim``.

        Given a 3-D tensor and reduction using the multiplication operation, :attr:`self`
        is updated as::

            self[index[i][j][k]][j][k] *= src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] *= src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] *= src[i][j][k]  # if dim == 2

        Reducing with the addition operation is the same as using
        :meth:`~torch.Tensor.scatter_add_`.

        .. warning::
            The reduce argument with Tensor ``src`` is deprecated and will be removed in
            a future PyTorch release. Please use :meth:`~torch.Tensor.scatter_reduce_`
            instead for more reduction options.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter, can be either empty
                or of the same dimensionality as ``src``. When empty, the operation
                returns ``self`` unchanged.
            src (Tensor): the source element(s) to scatter.

        Keyword args:
            reduce (str, optional): reduction operation to apply, can be either
                ``'add'`` or ``'multiply'``.

        Example::

            >>> src = torch.arange(1, 11).reshape((2, 5))
            >>> src
            tensor([[ 1,  2,  3,  4,  5],
                    [ 6,  7,  8,  9, 10]])
            >>> index = torch.tensor([[0, 1, 2, 0]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
            tensor([[1, 0, 0, 4, 0],
                    [0, 2, 0, 0, 0],
                    [0, 0, 3, 0, 0]])
            >>> index = torch.tensor([[0, 1, 2], [0, 1, 4]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
            tensor([[1, 2, 3, 0, 0],
                    [6, 7, 0, 0, 8],
                    [0, 0, 0, 0, 0]])

            >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
            ...            1.23, reduce='multiply')
            tensor([[2.0000, 2.0000, 2.4600, 2.0000],
                    [2.0000, 2.0000, 2.0000, 2.4600]])
            >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
            ...            1.23, reduce='add')
            tensor([[2.0000, 2.0000, 3.2300, 2.0000],
                    [2.0000, 2.0000, 2.0000, 3.2300]])

        .. function:: scatter_(dim, index, value, *, reduce=None) -> Tensor:
           :noindex:

        Writes the value from :attr:`value` into :attr:`self` at the indices
        specified in the :attr:`index` tensor.  This operation is equivalent to the previous version,
        with the :attr:`src` tensor filled entirely with :attr:`value`.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter, can be either empty
                or of the same dimensionality as ``src``. When empty, the operation
                returns ``self`` unchanged.
            value (Scalar): the value to scatter.

        Keyword args:
            reduce (str, optional): reduction operation to apply, can be either
                ``'add'`` or ``'multiply'``.

        Example::

            >>> index = torch.tensor([[0, 1]])
            >>> value = 2
            >>> torch.zeros(3, 5).scatter_(0, index, value)
            tensor([[2., 0., 0., 0., 0.],
                    [0., 2., 0., 0., 0.],
                    [0., 0., 0., 0., 0.]])
        """
    @overload
    def scatter_(self, dim: _int, index: Tensor, value: Number | _complex) -> Tensor:
        """
        scatter_(dim, index, src, *, reduce=None) -> Tensor

        Writes all values from the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index` tensor. For each value in :attr:`src`, its output
        index is specified by its index in :attr:`src` for ``dimension != dim`` and by
        the corresponding value in :attr:`index` for ``dimension = dim``.

        For a 3-D tensor, :attr:`self` is updated as::

            self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

        This is the reverse operation of the manner described in :meth:`~Tensor.gather`.

        It is also required that
        ``index.size(d) <= src.size(d)`` for all dimensions ``d``, and that
        ``index.size(d) <= self.size(d)`` for all dimensions ``d != dim``.
        Note that ``input`` and ``index`` do not broadcast against each other for NPUs,
        so when running on NPUs, :attr:`input` and :attr:`index` must have the same number of dimensions.
        Standard broadcasting occurs in all other cases.

        Moreover, as for :meth:`~Tensor.gather`, the values of :attr:`index` must be
        between ``0`` and ``self.size(dim) - 1`` inclusive.

        .. warning::

            When indices are not unique, the behavior is non-deterministic (one of the
            values from ``src`` will be picked arbitrarily) and the gradient will be
            incorrect (it will be propagated to all locations in the source that
            correspond to the same index)!

        .. note::

            The backward pass is implemented only for ``src.shape == index.shape``.

        Additionally accepts an optional :attr:`reduce` argument that allows
        specification of an optional reduction operation, which is applied to all
        values in the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index`. For each value in :attr:`src`, the reduction
        operation is applied to an index in :attr:`self` which is specified by
        its index in :attr:`src` for ``dimension != dim`` and by the corresponding
        value in :attr:`index` for ``dimension = dim``.

        Given a 3-D tensor and reduction using the multiplication operation, :attr:`self`
        is updated as::

            self[index[i][j][k]][j][k] *= src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] *= src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] *= src[i][j][k]  # if dim == 2

        Reducing with the addition operation is the same as using
        :meth:`~torch.Tensor.scatter_add_`.

        .. warning::
            The reduce argument with Tensor ``src`` is deprecated and will be removed in
            a future PyTorch release. Please use :meth:`~torch.Tensor.scatter_reduce_`
            instead for more reduction options.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter, can be either empty
                or of the same dimensionality as ``src``. When empty, the operation
                returns ``self`` unchanged.
            src (Tensor): the source element(s) to scatter.

        Keyword args:
            reduce (str, optional): reduction operation to apply, can be either
                ``'add'`` or ``'multiply'``.

        Example::

            >>> src = torch.arange(1, 11).reshape((2, 5))
            >>> src
            tensor([[ 1,  2,  3,  4,  5],
                    [ 6,  7,  8,  9, 10]])
            >>> index = torch.tensor([[0, 1, 2, 0]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
            tensor([[1, 0, 0, 4, 0],
                    [0, 2, 0, 0, 0],
                    [0, 0, 3, 0, 0]])
            >>> index = torch.tensor([[0, 1, 2], [0, 1, 4]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
            tensor([[1, 2, 3, 0, 0],
                    [6, 7, 0, 0, 8],
                    [0, 0, 0, 0, 0]])

            >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
            ...            1.23, reduce='multiply')
            tensor([[2.0000, 2.0000, 2.4600, 2.0000],
                    [2.0000, 2.0000, 2.0000, 2.4600]])
            >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
            ...            1.23, reduce='add')
            tensor([[2.0000, 2.0000, 3.2300, 2.0000],
                    [2.0000, 2.0000, 2.0000, 3.2300]])

        .. function:: scatter_(dim, index, value, *, reduce=None) -> Tensor:
           :noindex:

        Writes the value from :attr:`value` into :attr:`self` at the indices
        specified in the :attr:`index` tensor.  This operation is equivalent to the previous version,
        with the :attr:`src` tensor filled entirely with :attr:`value`.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter, can be either empty
                or of the same dimensionality as ``src``. When empty, the operation
                returns ``self`` unchanged.
            value (Scalar): the value to scatter.

        Keyword args:
            reduce (str, optional): reduction operation to apply, can be either
                ``'add'`` or ``'multiply'``.

        Example::

            >>> index = torch.tensor([[0, 1]])
            >>> value = 2
            >>> torch.zeros(3, 5).scatter_(0, index, value)
            tensor([[2., 0., 0., 0., 0.],
                    [0., 2., 0., 0., 0.],
                    [0., 0., 0., 0., 0.]])
        """
    @overload
    def scatter_add(self, dim: _int, index: Tensor, src: Tensor) -> Tensor:
        """
        scatter_add(dim, index, src) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_add_`
        """
    @overload
    def scatter_add(self, dim: str | EllipsisType | None, index: Tensor, src: Tensor) -> Tensor:
        """
        scatter_add(dim, index, src) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_add_`
        """
    def scatter_add_(self, dim: _int, index: Tensor, src: Tensor) -> Tensor:
        """
        scatter_add_(dim, index, src) -> Tensor

        Adds all values from the tensor :attr:`src` into :attr:`self` at the indices
        specified in the :attr:`index` tensor in a similar fashion as
        :meth:`~torch.Tensor.scatter_`. For each value in :attr:`src`, it is added to
        an index in :attr:`self` which is specified by its index in :attr:`src`
        for ``dimension != dim`` and by the corresponding value in :attr:`index` for
        ``dimension = dim``.

        For a 3-D tensor, :attr:`self` is updated as::

            self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2

        :attr:`self`, :attr:`index` and :attr:`src` should have same number of
        dimensions. It is also required that ``index.size(d) <= src.size(d)`` for all
        dimensions ``d``, and that ``index.size(d) <= self.size(d)`` for all dimensions
        ``d != dim``. Note that ``index`` and ``src`` do not broadcast.
        When :attr:`index` is empty, we always return the original tensor
        without further error checking.

        Note:
            This operation may behave nondeterministically when given tensors on a CUDA device. See :doc:`/notes/randomness` for more information.

        .. note::

            The backward pass is implemented only for ``src.shape == index.shape``.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter and add, can be
                either empty or of the same dimensionality as ``src``. When empty, the
                operation returns ``self`` unchanged.
            src (Tensor): the source elements to scatter and add

        Example::

            >>> src = torch.ones((2, 5))
            >>> index = torch.tensor([[0, 1, 2, 0, 0]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_add_(0, index, src)
            tensor([[1., 0., 0., 1., 1.],
                    [0., 1., 0., 0., 0.],
                    [0., 0., 1., 0., 0.]])
            >>> index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 2, 2]])
            >>> torch.zeros(3, 5, dtype=src.dtype).scatter_add_(0, index, src)
            tensor([[2., 0., 0., 1., 1.],
                    [0., 2., 0., 0., 0.],
                    [0., 0., 2., 1., 1.]])
        """
    def scatter_reduce(
        self, dim: _int, index: Tensor, src: Tensor, reduce: str, *, include_self: _bool = ...
    ) -> Tensor:
        """
        scatter_reduce(dim, index, src, reduce, *, include_self=True) -> Tensor

        Out-of-place version of :meth:`torch.Tensor.scatter_reduce_`
        """
    def scatter_reduce_(
        self, dim: _int, index: Tensor, src: Tensor, reduce: str, *, include_self: _bool = ...
    ) -> Tensor:
        """
        scatter_reduce_(dim, index, src, reduce, *, include_self=True) -> Tensor

        Reduces all values from the :attr:`src` tensor to the indices specified in
        the :attr:`index` tensor in the :attr:`self` tensor using the applied reduction
        defined via the :attr:`reduce` argument (:obj:`"sum"`, :obj:`"prod"`, :obj:`"mean"`,
        :obj:`"amax"`, :obj:`"amin"`). For each value in :attr:`src`, it is reduced to an
        index in :attr:`self` which is specified by its index in :attr:`src` for
        ``dimension != dim`` and by the corresponding value in :attr:`index` for
        ``dimension = dim``. If :obj:`include_self="True"`, the values in the :attr:`self`
        tensor are included in the reduction.

        :attr:`self`, :attr:`index` and :attr:`src` should all have
        the same number of dimensions. It is also required that
        ``index.size(d) <= src.size(d)`` for all dimensions ``d``, and that
        ``index.size(d) <= self.size(d)`` for all dimensions ``d != dim``.
        Note that ``index`` and ``src`` do not broadcast.

        For a 3-D tensor with :obj:`reduce="sum"` and :obj:`include_self=True` the
        output is given as::

            self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2

        Note:
            This operation may behave nondeterministically when given tensors on a CUDA device. See :doc:`/notes/randomness` for more information.

        .. note::

            The backward pass is implemented only for ``src.shape == index.shape``.

        .. warning::

            This function is in beta and may change in the near future.

        Args:
            dim (int): the axis along which to index
            index (LongTensor): the indices of elements to scatter and reduce.
            src (Tensor): the source elements to scatter and reduce
            reduce (str): the reduction operation to apply for non-unique indices
                (:obj:`"sum"`, :obj:`"prod"`, :obj:`"mean"`, :obj:`"amax"`, :obj:`"amin"`)
            include_self (bool): whether elements from the :attr:`self` tensor are
                included in the reduction

        Example::

            >>> src = torch.tensor([1., 2., 3., 4., 5., 6.])
            >>> index = torch.tensor([0, 1, 0, 1, 2, 1])
            >>> input = torch.tensor([1., 2., 3., 4.])
            >>> input.scatter_reduce(0, index, src, reduce="sum")
            tensor([5., 14., 8., 4.])
            >>> input.scatter_reduce(0, index, src, reduce="sum", include_self=False)
            tensor([4., 12., 5., 4.])
            >>> input2 = torch.tensor([5., 4., 3., 2.])
            >>> input2.scatter_reduce(0, index, src, reduce="amax")
            tensor([5., 6., 5., 2.])
            >>> input2.scatter_reduce(0, index, src, reduce="amax", include_self=False)
            tensor([3., 6., 5., 2.])
        """
    @overload
    def select(self, dim: _int, index: _int | SymInt) -> Tensor:
        """
        select(dim, index) -> Tensor

        See :func:`torch.select`
        """
    @overload
    def select(self, dim: str | EllipsisType | None, index: _int) -> Tensor:
        """
        select(dim, index) -> Tensor

        See :func:`torch.select`
        """
    def select_scatter(self, src: Tensor, dim: _int, index: _int | SymInt) -> Tensor:
        """
        select_scatter(src, dim, index) -> Tensor

        See :func:`torch.select_scatter`
        """
    @overload
    def set_(
        self,
        source: Storage | TypedStorage | UntypedStorage,
        storage_offset: IntLikeType,
        size: _symsize,
        stride: _symsize,
    ) -> Tensor:
        """
        set_(source=None, storage_offset=0, size=None, stride=None) -> Tensor

        Sets the underlying storage, size, and strides. If :attr:`source` is a tensor,
        :attr:`self` tensor will share the same storage and have the same size and
        strides as :attr:`source`. Changes to elements in one tensor will be reflected
        in the other.

        If :attr:`source` is a :class:`~torch.Storage`, the method sets the underlying
        storage, offset, size, and stride.

        Args:
            source (Tensor or Storage): the tensor or storage to use
            storage_offset (int, optional): the offset in the storage
            size (torch.Size, optional): the desired size. Defaults to the size of the source.
            stride (tuple, optional): the desired stride. Defaults to C-contiguous strides.
        """
    @overload
    def set_(self, source: Storage | TypedStorage | UntypedStorage) -> Tensor:
        """
        set_(source=None, storage_offset=0, size=None, stride=None) -> Tensor

        Sets the underlying storage, size, and strides. If :attr:`source` is a tensor,
        :attr:`self` tensor will share the same storage and have the same size and
        strides as :attr:`source`. Changes to elements in one tensor will be reflected
        in the other.

        If :attr:`source` is a :class:`~torch.Storage`, the method sets the underlying
        storage, offset, size, and stride.

        Args:
            source (Tensor or Storage): the tensor or storage to use
            storage_offset (int, optional): the offset in the storage
            size (torch.Size, optional): the desired size. Defaults to the size of the source.
            stride (tuple, optional): the desired stride. Defaults to C-contiguous strides.
        """
    def sgn(self) -> Tensor:
        """
        sgn() -> Tensor

        See :func:`torch.sgn`
        """
    def sgn_(self) -> Tensor:
        """
        sgn_() -> Tensor

        In-place version of :meth:`~Tensor.sgn`
        """
    def short(self) -> Tensor:
        """
        short(memory_format=torch.preserve_format) -> Tensor

        ``self.short()`` is equivalent to ``self.to(torch.int16)``. See :func:`to`.

        Args:
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
    def sigmoid(self) -> Tensor:
        """
        sigmoid() -> Tensor

        See :func:`torch.sigmoid`
        """
    def sigmoid_(self) -> Tensor:
        """
        sigmoid_() -> Tensor

        In-place version of :meth:`~Tensor.sigmoid`
        """
    def sign(self) -> Tensor:
        """
        sign() -> Tensor

        See :func:`torch.sign`
        """
    def sign_(self) -> Tensor:
        """
        sign_() -> Tensor

        In-place version of :meth:`~Tensor.sign`
        """
    def signbit(self) -> Tensor:
        """
        signbit() -> Tensor

        See :func:`torch.signbit`
        """
    def sin(self) -> Tensor:
        """
        sin() -> Tensor

        See :func:`torch.sin`
        """
    def sin_(self) -> Tensor:
        """
        sin_() -> Tensor

        In-place version of :meth:`~Tensor.sin`
        """
    def sinc(self) -> Tensor:
        """
        sinc() -> Tensor

        See :func:`torch.sinc`
        """
    def sinc_(self) -> Tensor:
        """
        sinc_() -> Tensor

        In-place version of :meth:`~Tensor.sinc`
        """
    def sinh(self) -> Tensor:
        """
        sinh() -> Tensor

        See :func:`torch.sinh`
        """
    def sinh_(self) -> Tensor:
        """
        sinh_() -> Tensor

        In-place version of :meth:`~Tensor.sinh`
        """
    @overload
    def size(self, dim: None = ...) -> Size:
        """
        size(dim=None) -> torch.Size or int

        Returns the size of the :attr:`self` tensor. If ``dim`` is not specified,
        the returned value is a :class:`torch.Size`, a subclass of :class:`tuple`.
        If ``dim`` is specified, returns an int holding the size of that dimension.

        Args:
          dim (int, optional): The dimension for which to retrieve the size.

        Example::

            >>> t = torch.empty(3, 4, 5)
            >>> t.size()
            torch.Size([3, 4, 5])
            >>> t.size(dim=1)
            4
        """
    @overload
    def size(self, dim: _int) -> _int:
        """
        size(dim=None) -> torch.Size or int

        Returns the size of the :attr:`self` tensor. If ``dim`` is not specified,
        the returned value is a :class:`torch.Size`, a subclass of :class:`tuple`.
        If ``dim`` is specified, returns an int holding the size of that dimension.

        Args:
          dim (int, optional): The dimension for which to retrieve the size.

        Example::

            >>> t = torch.empty(3, 4, 5)
            >>> t.size()
            torch.Size([3, 4, 5])
            >>> t.size(dim=1)
            4
        """
    def slice_inverse(
        self,
        src: Tensor,
        dim: _int = ...,
        start: _int | SymInt | None = ...,
        end: _int | SymInt | None = ...,
        step: _int | SymInt = ...,
    ) -> Tensor: ...
    def slice_scatter(
        self,
        src: Tensor,
        dim: _int = ...,
        start: _int | SymInt | None = ...,
        end: _int | SymInt | None = ...,
        step: _int | SymInt = ...,
    ) -> Tensor:
        """
        slice_scatter(src, dim=0, start=None, end=None, step=1) -> Tensor

        See :func:`torch.slice_scatter`
        """
    def slogdet(self) -> torch.return_types.slogdet:
        """
        slogdet() -> (Tensor, Tensor)

        See :func:`torch.slogdet`
        """
    def smm(self, mat2: Tensor) -> Tensor:
        """
        smm(mat) -> Tensor

        See :func:`torch.smm`
        """
    @overload
    def softmax(self, dim: _int, dtype: _dtype | None = ...) -> Tensor:
        """
        softmax(dim) -> Tensor

        Alias for :func:`torch.nn.functional.softmax`.
        """
    @overload
    def softmax(self, dim: str | EllipsisType | None, *, dtype: _dtype | None = ...) -> Tensor:
        """
        softmax(dim) -> Tensor

        Alias for :func:`torch.nn.functional.softmax`.
        """
    @overload
    def sort(self, *, stable: _bool | None, dim: _int = ..., descending: _bool = ...) -> torch.return_types.sort:
        """
        sort(dim=-1, descending=False) -> (Tensor, LongTensor)

        See :func:`torch.sort`
        """
    @overload
    def sort(self, dim: _int = ..., descending: _bool = ...) -> torch.return_types.sort:
        """
        sort(dim=-1, descending=False) -> (Tensor, LongTensor)

        See :func:`torch.sort`
        """
    @overload
    def sort(
        self, *, stable: _bool | None, dim: str | EllipsisType | None, descending: _bool = ...
    ) -> torch.return_types.sort:
        """
        sort(dim=-1, descending=False) -> (Tensor, LongTensor)

        See :func:`torch.sort`
        """
    @overload
    def sort(self, dim: str | EllipsisType | None, descending: _bool = ...) -> torch.return_types.sort:
        """
        sort(dim=-1, descending=False) -> (Tensor, LongTensor)

        See :func:`torch.sort`
        """
    def sparse_dim(self) -> _int:
        """
        sparse_dim() -> int

        Return the number of sparse dimensions in a :ref:`sparse tensor <sparse-docs>` :attr:`self`.

        .. note::
          Returns ``0`` if :attr:`self` is not a sparse tensor.

        See also :meth:`Tensor.dense_dim` and :ref:`hybrid tensors <sparse-hybrid-coo-docs>`.
        """
    def sparse_mask(self, mask: Tensor) -> Tensor:
        """
        sparse_mask(mask) -> Tensor

        Returns a new :ref:`sparse tensor <sparse-docs>` with values from a
        strided tensor :attr:`self` filtered by the indices of the sparse
        tensor :attr:`mask`. The values of :attr:`mask` sparse tensor are
        ignored. :attr:`self` and :attr:`mask` tensors must have the same
        shape.

        .. note::

          The returned sparse tensor might contain duplicate values if :attr:`mask`
          is not coalesced. It is therefore advisable to pass ``mask.coalesce()``
          if such behavior is not desired.

        .. note::

          The returned sparse tensor has the same indices as the sparse tensor
          :attr:`mask`, even when the corresponding values in :attr:`self` are
          zeros.

        Args:
            mask (Tensor): a sparse tensor whose indices are used as a filter

        Example::

            >>> nse = 5
            >>> dims = (5, 5, 2, 2)
            >>> I = torch.cat([torch.randint(0, dims[0], size=(nse,)),
            ...                torch.randint(0, dims[1], size=(nse,))], 0).reshape(2, nse)
            >>> V = torch.randn(nse, dims[2], dims[3])
            >>> S = torch.sparse_coo_tensor(I, V, dims).coalesce()
            >>> D = torch.randn(dims)
            >>> D.sparse_mask(S)
            tensor(indices=tensor([[0, 0, 0, 2],
                                   [0, 1, 4, 3]]),
                   values=tensor([[[ 1.6550,  0.2397],
                                   [-0.1611, -0.0779]],

                                  [[ 0.2326, -1.0558],
                                   [ 1.4711,  1.9678]],

                                  [[-0.5138, -0.0411],
                                   [ 1.9417,  0.5158]],

                                  [[ 0.0793,  0.0036],
                                   [-0.2569, -0.1055]]]),
                   size=(5, 5, 2, 2), nnz=4, layout=torch.sparse_coo)
        """
    def sparse_resize_(self, size: _size, sparse_dim: _int, dense_dim: _int) -> Tensor:
        """
        sparse_resize_(size, sparse_dim, dense_dim) -> Tensor

        Resizes :attr:`self` :ref:`sparse tensor <sparse-docs>` to the desired
        size and the number of sparse and dense dimensions.

        .. note::
          If the number of specified elements in :attr:`self` is zero, then
          :attr:`size`, :attr:`sparse_dim`, and :attr:`dense_dim` can be any
          size and positive integers such that ``len(size) == sparse_dim +
          dense_dim``.

          If :attr:`self` specifies one or more elements, however, then each
          dimension in :attr:`size` must not be smaller than the corresponding
          dimension of :attr:`self`, :attr:`sparse_dim` must equal the number
          of sparse dimensions in :attr:`self`, and :attr:`dense_dim` must
          equal the number of dense dimensions in :attr:`self`.

        .. warning::
          Throws an error if :attr:`self` is not a sparse tensor.

        Args:
            size (torch.Size): the desired size. If :attr:`self` is non-empty
              sparse tensor, the desired size cannot be smaller than the
              original size.
            sparse_dim (int): the number of sparse dimensions
            dense_dim (int): the number of dense dimensions
        """
    def sparse_resize_and_clear_(self, size: _size, sparse_dim: _int, dense_dim: _int) -> Tensor:
        """
        sparse_resize_and_clear_(size, sparse_dim, dense_dim) -> Tensor

        Removes all specified elements from a :ref:`sparse tensor
        <sparse-docs>` :attr:`self` and resizes :attr:`self` to the desired
        size and the number of sparse and dense dimensions.

        .. warning:
          Throws an error if :attr:`self` is not a sparse tensor.

        Args:
            size (torch.Size): the desired size.
            sparse_dim (int): the number of sparse dimensions
            dense_dim (int): the number of dense dimensions
        """
    @overload
    def split(self, split_size: _int, dim: _int = ...) -> Sequence[Tensor]: ...
    @overload
    def split(self, split_size: tuple[_int, ...], dim: _int = ...) -> Sequence[Tensor]: ...
    def split_with_sizes(self, split_sizes: Sequence[_int | SymInt], dim: _int = ...) -> tuple[Tensor, ...]: ...
    def sqrt(self) -> Tensor:
        """
        sqrt() -> Tensor

        See :func:`torch.sqrt`
        """
    def sqrt_(self) -> Tensor:
        """
        sqrt_() -> Tensor

        In-place version of :meth:`~Tensor.sqrt`
        """
    def square(self) -> Tensor:
        """
        square() -> Tensor

        See :func:`torch.square`
        """
    def square_(self) -> Tensor:
        """
        square_() -> Tensor

        In-place version of :meth:`~Tensor.square`
        """
    @overload
    def squeeze(self) -> Tensor:
        """
        squeeze(dim=None) -> Tensor

        See :func:`torch.squeeze`
        """
    @overload
    def squeeze(self, dim: _int) -> Tensor:
        """
        squeeze(dim=None) -> Tensor

        See :func:`torch.squeeze`
        """
    @overload
    def squeeze(self, dim: _size) -> Tensor:
        """
        squeeze(dim=None) -> Tensor

        See :func:`torch.squeeze`
        """
    @overload
    def squeeze(self, *dim: _int) -> Tensor:
        """
        squeeze(dim=None) -> Tensor

        See :func:`torch.squeeze`
        """
    @overload
    def squeeze(self, dim: str | EllipsisType | None) -> Tensor:
        """
        squeeze(dim=None) -> Tensor

        See :func:`torch.squeeze`
        """
    @overload
    def squeeze_(self) -> Tensor:
        """
        squeeze_(dim=None) -> Tensor

        In-place version of :meth:`~Tensor.squeeze`
        """
    @overload
    def squeeze_(self, dim: _int) -> Tensor:
        """
        squeeze_(dim=None) -> Tensor

        In-place version of :meth:`~Tensor.squeeze`
        """
    @overload
    def squeeze_(self, dim: _size) -> Tensor:
        """
        squeeze_(dim=None) -> Tensor

        In-place version of :meth:`~Tensor.squeeze`
        """
    @overload
    def squeeze_(self, *dim: _int) -> Tensor:
        """
        squeeze_(dim=None) -> Tensor

        In-place version of :meth:`~Tensor.squeeze`
        """
    @overload
    def squeeze_(self, dim: str | EllipsisType | None) -> Tensor:
        """
        squeeze_(dim=None) -> Tensor

        In-place version of :meth:`~Tensor.squeeze`
        """
    def sspaddmm(
        self, mat1: Tensor, mat2: Tensor, *, beta: Number | _complex = ..., alpha: Number | _complex = ...
    ) -> Tensor:
        """
        sspaddmm(mat1, mat2, *, beta=1, alpha=1) -> Tensor

        See :func:`torch.sspaddmm`
        """
    @overload
    def std(self, dim: _int | _size | None, unbiased: _bool = ..., keepdim: _bool = ...) -> Tensor:
        """
        std(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.std`
        """
    @overload
    def std(
        self, dim: _int | _size | None = ..., *, correction: Number | _complex | None = ..., keepdim: _bool = ...
    ) -> Tensor:
        """
        std(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.std`
        """
    @overload
    def std(self, unbiased: _bool = ...) -> Tensor:
        """
        std(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.std`
        """
    @overload
    def std(self, dim: Sequence[str | EllipsisType | None], unbiased: _bool = ..., keepdim: _bool = ...) -> Tensor:
        """
        std(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.std`
        """
    @overload
    def std(
        self,
        dim: Sequence[str | EllipsisType | None],
        *,
        correction: Number | _complex | None = ...,
        keepdim: _bool = ...,
    ) -> Tensor:
        """
        std(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.std`
        """
    def untyped_storage(self) -> UntypedStorage:
        """
        untyped_storage() -> torch.UntypedStorage

        Returns the underlying :class:`UntypedStorage`.
        """
    def storage_offset(self) -> _int | SymInt:
        """
        storage_offset() -> int

        Returns :attr:`self` tensor's offset in the underlying storage in terms of
        number of storage elements (not bytes).

        Example::

            >>> x = torch.tensor([1, 2, 3, 4, 5])
            >>> x.storage_offset()
            0
            >>> x[3:].storage_offset()
            3
        """
    def storage_type(self) -> Storage: ...
    @overload
    def stride(self, dim: None = ...) -> tuple[_int, ...]:
        """
        stride(dim) -> tuple or int

        Returns the stride of :attr:`self` tensor.

        Stride is the jump necessary to go from one element to the next one in the
        specified dimension :attr:`dim`. A tuple of all strides is returned when no
        argument is passed in. Otherwise, an integer value is returned as the stride in
        the particular dimension :attr:`dim`.

        Args:
            dim (int, optional): the desired dimension in which stride is required

        Example::

            >>> x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
            >>> x.stride()
            (5, 1)
            >>> x.stride(0)
            5
            >>> x.stride(-1)
            1
        """
    @overload
    def stride(self, dim: _int) -> _int:
        """
        stride(dim) -> tuple or int

        Returns the stride of :attr:`self` tensor.

        Stride is the jump necessary to go from one element to the next one in the
        specified dimension :attr:`dim`. A tuple of all strides is returned when no
        argument is passed in. Otherwise, an integer value is returned as the stride in
        the particular dimension :attr:`dim`.

        Args:
            dim (int, optional): the desired dimension in which stride is required

        Example::

            >>> x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
            >>> x.stride()
            (5, 1)
            >>> x.stride(0)
            5
            >>> x.stride(-1)
            1
        """
    def sub(
        self,
        other: Tensor | Number | _complex | torch.SymInt | torch.SymFloat,
        *,
        alpha: Number | _complex | None = ...,
        out: Tensor | None = ...,
    ) -> Tensor:
        """
        sub(other, *, alpha=1) -> Tensor

        See :func:`torch.sub`.
        """
    def sub_(
        self,
        other: Tensor | Number | _complex | torch.SymInt | torch.SymFloat,
        *,
        alpha: Number | _complex | None = ...,
    ) -> Tensor:
        """
        sub_(other, *, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.sub`
        """
    @overload
    def subtract(self, other: Tensor, *, alpha: Number | _complex = ...) -> Tensor:
        """
        subtract(other, *, alpha=1) -> Tensor

        See :func:`torch.subtract`.
        """
    @overload
    def subtract(self, other: Number | _complex, alpha: Number | _complex = ...) -> Tensor:
        """
        subtract(other, *, alpha=1) -> Tensor

        See :func:`torch.subtract`.
        """
    @overload
    def subtract_(self, other: Tensor, *, alpha: Number | _complex = ...) -> Tensor:
        """
        subtract_(other, *, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.subtract`.
        """
    @overload
    def subtract_(self, other: Number | _complex, alpha: Number | _complex = ...) -> Tensor:
        """
        subtract_(other, *, alpha=1) -> Tensor

        In-place version of :meth:`~Tensor.subtract`.
        """
    @overload
    def sum(self, *, dtype: _dtype | None = ...) -> Tensor:
        """
        sum(dim=None, keepdim=False, dtype=None) -> Tensor

        See :func:`torch.sum`
        """
    @overload
    def sum(self, dim: _int | _size | None, keepdim: _bool = ..., *, dtype: _dtype | None = ...) -> Tensor:
        """
        sum(dim=None, keepdim=False, dtype=None) -> Tensor

        See :func:`torch.sum`
        """
    @overload
    def sum(
        self, dim: Sequence[str | EllipsisType | None], keepdim: _bool = ..., *, dtype: _dtype | None = ...
    ) -> Tensor:
        """
        sum(dim=None, keepdim=False, dtype=None) -> Tensor

        See :func:`torch.sum`
        """
    @overload
    def sum_to_size(self, size: Sequence[_int | SymInt]) -> Tensor:
        """
        sum_to_size(*size) -> Tensor

        Sum ``this`` tensor to :attr:`size`.
        :attr:`size` must be broadcastable to ``this`` tensor size.

        Args:
            size (int...): a sequence of integers defining the shape of the output tensor.
        """
    @overload
    def sum_to_size(self, *size: _int | SymInt) -> Tensor:
        """
        sum_to_size(*size) -> Tensor

        Sum ``this`` tensor to :attr:`size`.
        :attr:`size` must be broadcastable to ``this`` tensor size.

        Args:
            size (int...): a sequence of integers defining the shape of the output tensor.
        """
    def svd(self, some: _bool = ..., compute_uv: _bool = ...) -> torch.return_types.svd:
        """
        svd(some=True, compute_uv=True) -> (Tensor, Tensor, Tensor)

        See :func:`torch.svd`
        """
    def swapaxes(self, axis0: _int, axis1: _int) -> Tensor:
        """
        swapaxes(axis0, axis1) -> Tensor

        See :func:`torch.swapaxes`
        """
    def swapaxes_(self, axis0: _int, axis1: _int) -> Tensor:
        """
        swapaxes_(axis0, axis1) -> Tensor

        In-place version of :meth:`~Tensor.swapaxes`
        """
    def swapdims(self, dim0: _int, dim1: _int) -> Tensor:
        """
        swapdims(dim0, dim1) -> Tensor

        See :func:`torch.swapdims`
        """
    def swapdims_(self, dim0: _int, dim1: _int) -> Tensor:
        """
        swapdims_(dim0, dim1) -> Tensor

        In-place version of :meth:`~Tensor.swapdims`
        """
    def t(self) -> Tensor:
        """
        t() -> Tensor

        See :func:`torch.t`
        """
    def t_(self) -> Tensor:
        """
        t_() -> Tensor

        In-place version of :meth:`~Tensor.t`
        """
    def take(self, index: Tensor) -> Tensor:
        """
        take(indices) -> Tensor

        See :func:`torch.take`
        """
    def take_along_dim(self, indices: Tensor, dim: _int | None = ...) -> Tensor:
        """
        take_along_dim(indices, dim) -> Tensor

        See :func:`torch.take_along_dim`
        """
    def tan(self) -> Tensor:
        """
        tan() -> Tensor

        See :func:`torch.tan`
        """
    def tan_(self) -> Tensor:
        """
        tan_() -> Tensor

        In-place version of :meth:`~Tensor.tan`
        """
    def tanh(self) -> Tensor:
        """
        tanh() -> Tensor

        See :func:`torch.tanh`
        """
    def tanh_(self) -> Tensor:
        """
        tanh_() -> Tensor

        In-place version of :meth:`~Tensor.tanh`
        """
    @overload
    def tensor_split(self, indices: Sequence[_int | SymInt], dim: _int = ...) -> tuple[Tensor, ...]:
        """
        tensor_split(indices_or_sections, dim=0) -> List of Tensors

        See :func:`torch.tensor_split`
        """
    @overload
    def tensor_split(self, tensor_indices_or_sections: Tensor, dim: _int = ...) -> tuple[Tensor, ...]:
        """
        tensor_split(indices_or_sections, dim=0) -> List of Tensors

        See :func:`torch.tensor_split`
        """
    @overload
    def tensor_split(self, sections: _int | SymInt, dim: _int = ...) -> tuple[Tensor, ...]:
        """
        tensor_split(indices_or_sections, dim=0) -> List of Tensors

        See :func:`torch.tensor_split`
        """
    @overload
    def tile(self, dims: Sequence[_int | SymInt]) -> Tensor:
        """
        tile(dims) -> Tensor

        See :func:`torch.tile`
        """
    @overload
    def tile(self, *dims: _int | SymInt) -> Tensor:
        """
        tile(dims) -> Tensor

        See :func:`torch.tile`
        """
    @overload
    def to(
        self,
        dtype: _dtype,
        non_blocking: _bool = ...,
        copy: _bool = ...,
        *,
        memory_format: torch.memory_format | None = ...,
    ) -> Tensor:
        """
        to(*args, **kwargs) -> Tensor

        Performs Tensor dtype and/or device conversion. A :class:`torch.dtype` and :class:`torch.device` are
        inferred from the arguments of ``self.to(*args, **kwargs)``.

        .. note::

            If the ``self`` Tensor already
            has the correct :class:`torch.dtype` and :class:`torch.device`, then ``self`` is returned.
            Otherwise, the returned tensor is a copy of ``self`` with the desired
            :class:`torch.dtype` and :class:`torch.device`.

        .. note::

            If ``self`` requires gradients (``requires_grad=True``) but the target
            ``dtype`` specified is an integer type, the returned tensor will implicitly
            set ``requires_grad=False``. This is because only tensors with
            floating-point or complex dtypes can require gradients.

        Here are the ways to call ``to``:

        .. method:: to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
           :noindex:

            Returns a Tensor with the specified :attr:`dtype`

            Args:
                memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.

        .. note::

            According to `C++ type conversion rules <https://en.cppreference.com/w/cpp/language/implicit_conversion.html>`_,
            converting floating point value to integer type will truncate the fractional part.
            If the truncated value cannot fit into the target type (e.g., casting ``torch.inf`` to ``torch.long``),
            the behavior is undefined and the result may vary across platforms.

        .. method:: to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
           :noindex:

            Returns a Tensor with the specified :attr:`device` and (optional)
            :attr:`dtype`. If :attr:`dtype` is ``None`` it is inferred to be ``self.dtype``.
            When :attr:`non_blocking` is set to ``True``, the function attempts to perform
            the conversion asynchronously with respect to the host, if possible. This
            asynchronous behavior applies to both pinned and pageable memory. However,
            caution is advised when using this feature. For more information, refer to the
            `tutorial on good usage of non_blocking and pin_memory <https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html>`__.
            When :attr:`copy` is set, a new Tensor is created even when the Tensor
            already matches the desired conversion.

            Args:
                memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.

        .. method:: to(other, non_blocking=False, copy=False) -> Tensor
           :noindex:

            Returns a Tensor with same :class:`torch.dtype` and :class:`torch.device` as
            the Tensor :attr:`other`.
            When :attr:`non_blocking` is set to ``True``, the function attempts to perform
            the conversion asynchronously with respect to the host, if possible. This
            asynchronous behavior applies to both pinned and pageable memory. However,
            caution is advised when using this feature. For more information, refer to the
            `tutorial on good usage of non_blocking and pin_memory <https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html>`__.
            When :attr:`copy` is set, a new Tensor is created even when the Tensor
            already matches the desired conversion.

        Example::

            >>> tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
            >>> tensor.to(torch.float64)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64)

            >>> cuda0 = torch.device('cuda:0')
            >>> tensor.to(cuda0)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], device='cuda:0')

            >>> tensor.to(cuda0, dtype=torch.float64)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')

            >>> other = torch.randn((), dtype=torch.float64, device=cuda0)
            >>> tensor.to(other, non_blocking=True)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
        """
    @overload
    def to(
        self,
        device: DeviceLikeType | None = ...,
        dtype: _dtype | None = ...,
        non_blocking: _bool = ...,
        copy: _bool = ...,
        *,
        memory_format: torch.memory_format | None = ...,
    ) -> Tensor:
        """
        to(*args, **kwargs) -> Tensor

        Performs Tensor dtype and/or device conversion. A :class:`torch.dtype` and :class:`torch.device` are
        inferred from the arguments of ``self.to(*args, **kwargs)``.

        .. note::

            If the ``self`` Tensor already
            has the correct :class:`torch.dtype` and :class:`torch.device`, then ``self`` is returned.
            Otherwise, the returned tensor is a copy of ``self`` with the desired
            :class:`torch.dtype` and :class:`torch.device`.

        .. note::

            If ``self`` requires gradients (``requires_grad=True``) but the target
            ``dtype`` specified is an integer type, the returned tensor will implicitly
            set ``requires_grad=False``. This is because only tensors with
            floating-point or complex dtypes can require gradients.

        Here are the ways to call ``to``:

        .. method:: to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
           :noindex:

            Returns a Tensor with the specified :attr:`dtype`

            Args:
                memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.

        .. note::

            According to `C++ type conversion rules <https://en.cppreference.com/w/cpp/language/implicit_conversion.html>`_,
            converting floating point value to integer type will truncate the fractional part.
            If the truncated value cannot fit into the target type (e.g., casting ``torch.inf`` to ``torch.long``),
            the behavior is undefined and the result may vary across platforms.

        .. method:: to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
           :noindex:

            Returns a Tensor with the specified :attr:`device` and (optional)
            :attr:`dtype`. If :attr:`dtype` is ``None`` it is inferred to be ``self.dtype``.
            When :attr:`non_blocking` is set to ``True``, the function attempts to perform
            the conversion asynchronously with respect to the host, if possible. This
            asynchronous behavior applies to both pinned and pageable memory. However,
            caution is advised when using this feature. For more information, refer to the
            `tutorial on good usage of non_blocking and pin_memory <https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html>`__.
            When :attr:`copy` is set, a new Tensor is created even when the Tensor
            already matches the desired conversion.

            Args:
                memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.

        .. method:: to(other, non_blocking=False, copy=False) -> Tensor
           :noindex:

            Returns a Tensor with same :class:`torch.dtype` and :class:`torch.device` as
            the Tensor :attr:`other`.
            When :attr:`non_blocking` is set to ``True``, the function attempts to perform
            the conversion asynchronously with respect to the host, if possible. This
            asynchronous behavior applies to both pinned and pageable memory. However,
            caution is advised when using this feature. For more information, refer to the
            `tutorial on good usage of non_blocking and pin_memory <https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html>`__.
            When :attr:`copy` is set, a new Tensor is created even when the Tensor
            already matches the desired conversion.

        Example::

            >>> tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
            >>> tensor.to(torch.float64)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64)

            >>> cuda0 = torch.device('cuda:0')
            >>> tensor.to(cuda0)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], device='cuda:0')

            >>> tensor.to(cuda0, dtype=torch.float64)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')

            >>> other = torch.randn((), dtype=torch.float64, device=cuda0)
            >>> tensor.to(other, non_blocking=True)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
        """
    @overload
    def to(
        self,
        other: Tensor,
        non_blocking: _bool = ...,
        copy: _bool = ...,
        *,
        memory_format: torch.memory_format | None = ...,
    ) -> Tensor:
        """
        to(*args, **kwargs) -> Tensor

        Performs Tensor dtype and/or device conversion. A :class:`torch.dtype` and :class:`torch.device` are
        inferred from the arguments of ``self.to(*args, **kwargs)``.

        .. note::

            If the ``self`` Tensor already
            has the correct :class:`torch.dtype` and :class:`torch.device`, then ``self`` is returned.
            Otherwise, the returned tensor is a copy of ``self`` with the desired
            :class:`torch.dtype` and :class:`torch.device`.

        .. note::

            If ``self`` requires gradients (``requires_grad=True``) but the target
            ``dtype`` specified is an integer type, the returned tensor will implicitly
            set ``requires_grad=False``. This is because only tensors with
            floating-point or complex dtypes can require gradients.

        Here are the ways to call ``to``:

        .. method:: to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
           :noindex:

            Returns a Tensor with the specified :attr:`dtype`

            Args:
                memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.

        .. note::

            According to `C++ type conversion rules <https://en.cppreference.com/w/cpp/language/implicit_conversion.html>`_,
            converting floating point value to integer type will truncate the fractional part.
            If the truncated value cannot fit into the target type (e.g., casting ``torch.inf`` to ``torch.long``),
            the behavior is undefined and the result may vary across platforms.

        .. method:: to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
           :noindex:

            Returns a Tensor with the specified :attr:`device` and (optional)
            :attr:`dtype`. If :attr:`dtype` is ``None`` it is inferred to be ``self.dtype``.
            When :attr:`non_blocking` is set to ``True``, the function attempts to perform
            the conversion asynchronously with respect to the host, if possible. This
            asynchronous behavior applies to both pinned and pageable memory. However,
            caution is advised when using this feature. For more information, refer to the
            `tutorial on good usage of non_blocking and pin_memory <https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html>`__.
            When :attr:`copy` is set, a new Tensor is created even when the Tensor
            already matches the desired conversion.

            Args:
                memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.

        .. method:: to(other, non_blocking=False, copy=False) -> Tensor
           :noindex:

            Returns a Tensor with same :class:`torch.dtype` and :class:`torch.device` as
            the Tensor :attr:`other`.
            When :attr:`non_blocking` is set to ``True``, the function attempts to perform
            the conversion asynchronously with respect to the host, if possible. This
            asynchronous behavior applies to both pinned and pageable memory. However,
            caution is advised when using this feature. For more information, refer to the
            `tutorial on good usage of non_blocking and pin_memory <https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html>`__.
            When :attr:`copy` is set, a new Tensor is created even when the Tensor
            already matches the desired conversion.

        Example::

            >>> tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
            >>> tensor.to(torch.float64)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64)

            >>> cuda0 = torch.device('cuda:0')
            >>> tensor.to(cuda0)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], device='cuda:0')

            >>> tensor.to(cuda0, dtype=torch.float64)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')

            >>> other = torch.randn((), dtype=torch.float64, device=cuda0)
            >>> tensor.to(other, non_blocking=True)
            tensor([[-0.5044,  0.0005],
                    [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
        """
    def to_dense(self, dtype: _dtype | None = ..., *, masked_grad: _bool | None = ...) -> Tensor:
        """
        to_dense(dtype=None, *, masked_grad=True) -> Tensor

        Creates a strided copy of :attr:`self` if :attr:`self` is not a strided tensor, otherwise returns :attr:`self`.

        Keyword args:
            {dtype}
            masked_grad (bool, optional): If set to ``True`` (default) and
              :attr:`self` has a sparse layout then the backward of
              :meth:`to_dense` returns ``grad.sparse_mask(self)``.

        Example::

            >>> s = torch.sparse_coo_tensor(
            ...        torch.tensor([[1, 1],
            ...                      [0, 2]]),
            ...        torch.tensor([9, 10]),
            ...        size=(3, 3))
            >>> s.to_dense()
            tensor([[ 0,  0,  0],
                    [ 9,  0, 10],
                    [ 0,  0,  0]])
        """
    def to_mkldnn(self, dtype: _dtype | None = ...) -> Tensor:
        """
        to_mkldnn() -> Tensor
        Returns a copy of the tensor in ``torch.mkldnn`` layout.
        """
    def to_padded_tensor(self, padding: _float, output_size: Sequence[_int | SymInt] | None = ...) -> Tensor:
        """
        to_padded_tensor(padding, output_size=None) -> Tensor
        See :func:`to_padded_tensor`
        """
    @overload
    def to_sparse(
        self, *, layout: _layout | None = ..., blocksize: _int | _size | None = ..., dense_dim: _int | None = ...
    ) -> Tensor:
        """
        to_sparse(sparseDims) -> Tensor

        Returns a sparse copy of the tensor.  PyTorch supports sparse tensors in
        :ref:`coordinate format <sparse-coo-docs>`.

        Args:
            sparseDims (int, optional): the number of sparse dimensions to include in the new sparse tensor

        Example::

            >>> d = torch.tensor([[0, 0, 0], [9, 0, 10], [0, 0, 0]])
            >>> d
            tensor([[ 0,  0,  0],
                    [ 9,  0, 10],
                    [ 0,  0,  0]])
            >>> d.to_sparse()
            tensor(indices=tensor([[1, 1],
                                   [0, 2]]),
                   values=tensor([ 9, 10]),
                   size=(3, 3), nnz=2, layout=torch.sparse_coo)
            >>> d.to_sparse(1)
            tensor(indices=tensor([[1]]),
                   values=tensor([[ 9,  0, 10]]),
                   size=(3, 3), nnz=1, layout=torch.sparse_coo)

        .. method:: to_sparse(*, layout=None, blocksize=None, dense_dim=None) -> Tensor
           :noindex:

        Returns a sparse tensor with the specified layout and blocksize.  If
        the :attr:`self` is strided, the number of dense dimensions could be
        specified, and a hybrid sparse tensor will be created, with
        `dense_dim` dense dimensions and `self.dim() - 2 - dense_dim` batch
        dimension.

        .. note:: If the :attr:`self` layout and blocksize parameters match
                  with the specified layout and blocksize, return
                  :attr:`self`. Otherwise, return a sparse tensor copy of
                  :attr:`self`.

        Args:

            layout (:class:`torch.layout`, optional): The desired sparse
              layout. One of ``torch.sparse_coo``, ``torch.sparse_csr``,
              ``torch.sparse_csc``, ``torch.sparse_bsr``, or
              ``torch.sparse_bsc``. Default: if ``None``,
              ``torch.sparse_coo``.

            blocksize (list, tuple, :class:`torch.Size`, optional): Block size
              of the resulting BSR or BSC tensor. For other layouts,
              specifying the block size that is not ``None`` will result in a
              RuntimeError exception.  A block size must be a tuple of length
              two such that its items evenly divide the two sparse dimensions.

            dense_dim (int, optional): Number of dense dimensions of the
              resulting CSR, CSC, BSR or BSC tensor.  This argument should be
              used only if :attr:`self` is a strided tensor, and must be a
              value between 0 and dimension of :attr:`self` tensor minus two.

        Example::

            >>> x = torch.tensor([[1, 0], [0, 0], [2, 3]])
            >>> x.to_sparse(layout=torch.sparse_coo)
            tensor(indices=tensor([[0, 2, 2],
                                   [0, 0, 1]]),
                   values=tensor([1, 2, 3]),
                   size=(3, 2), nnz=3, layout=torch.sparse_coo)
            >>> x.to_sparse(layout=torch.sparse_bsr, blocksize=(1, 2))
            tensor(crow_indices=tensor([0, 1, 1, 2]),
                   col_indices=tensor([0, 0]),
                   values=tensor([[[1, 0]],
                                  [[2, 3]]]), size=(3, 2), nnz=2, layout=torch.sparse_bsr)
            >>> x.to_sparse(layout=torch.sparse_bsr, blocksize=(2, 1))
            RuntimeError: Tensor size(-2) 3 needs to be divisible by blocksize[0] 2
            >>> x.to_sparse(layout=torch.sparse_csr, blocksize=(3, 1))
            RuntimeError: to_sparse for Strided to SparseCsr conversion does not use specified blocksize

            >>> x = torch.tensor([[[1], [0]], [[0], [0]], [[2], [3]]])
            >>> x.to_sparse(layout=torch.sparse_csr, dense_dim=1)
            tensor(crow_indices=tensor([0, 1, 1, 3]),
                   col_indices=tensor([0, 0, 1]),
                   values=tensor([[1],
                                  [2],
                                  [3]]), size=(3, 2, 1), nnz=3, layout=torch.sparse_csr)
        """
    @overload
    def to_sparse(self, sparse_dim: _int) -> Tensor:
        """
        to_sparse(sparseDims) -> Tensor

        Returns a sparse copy of the tensor.  PyTorch supports sparse tensors in
        :ref:`coordinate format <sparse-coo-docs>`.

        Args:
            sparseDims (int, optional): the number of sparse dimensions to include in the new sparse tensor

        Example::

            >>> d = torch.tensor([[0, 0, 0], [9, 0, 10], [0, 0, 0]])
            >>> d
            tensor([[ 0,  0,  0],
                    [ 9,  0, 10],
                    [ 0,  0,  0]])
            >>> d.to_sparse()
            tensor(indices=tensor([[1, 1],
                                   [0, 2]]),
                   values=tensor([ 9, 10]),
                   size=(3, 3), nnz=2, layout=torch.sparse_coo)
            >>> d.to_sparse(1)
            tensor(indices=tensor([[1]]),
                   values=tensor([[ 9,  0, 10]]),
                   size=(3, 3), nnz=1, layout=torch.sparse_coo)

        .. method:: to_sparse(*, layout=None, blocksize=None, dense_dim=None) -> Tensor
           :noindex:

        Returns a sparse tensor with the specified layout and blocksize.  If
        the :attr:`self` is strided, the number of dense dimensions could be
        specified, and a hybrid sparse tensor will be created, with
        `dense_dim` dense dimensions and `self.dim() - 2 - dense_dim` batch
        dimension.

        .. note:: If the :attr:`self` layout and blocksize parameters match
                  with the specified layout and blocksize, return
                  :attr:`self`. Otherwise, return a sparse tensor copy of
                  :attr:`self`.

        Args:

            layout (:class:`torch.layout`, optional): The desired sparse
              layout. One of ``torch.sparse_coo``, ``torch.sparse_csr``,
              ``torch.sparse_csc``, ``torch.sparse_bsr``, or
              ``torch.sparse_bsc``. Default: if ``None``,
              ``torch.sparse_coo``.

            blocksize (list, tuple, :class:`torch.Size`, optional): Block size
              of the resulting BSR or BSC tensor. For other layouts,
              specifying the block size that is not ``None`` will result in a
              RuntimeError exception.  A block size must be a tuple of length
              two such that its items evenly divide the two sparse dimensions.

            dense_dim (int, optional): Number of dense dimensions of the
              resulting CSR, CSC, BSR or BSC tensor.  This argument should be
              used only if :attr:`self` is a strided tensor, and must be a
              value between 0 and dimension of :attr:`self` tensor minus two.

        Example::

            >>> x = torch.tensor([[1, 0], [0, 0], [2, 3]])
            >>> x.to_sparse(layout=torch.sparse_coo)
            tensor(indices=tensor([[0, 2, 2],
                                   [0, 0, 1]]),
                   values=tensor([1, 2, 3]),
                   size=(3, 2), nnz=3, layout=torch.sparse_coo)
            >>> x.to_sparse(layout=torch.sparse_bsr, blocksize=(1, 2))
            tensor(crow_indices=tensor([0, 1, 1, 2]),
                   col_indices=tensor([0, 0]),
                   values=tensor([[[1, 0]],
                                  [[2, 3]]]), size=(3, 2), nnz=2, layout=torch.sparse_bsr)
            >>> x.to_sparse(layout=torch.sparse_bsr, blocksize=(2, 1))
            RuntimeError: Tensor size(-2) 3 needs to be divisible by blocksize[0] 2
            >>> x.to_sparse(layout=torch.sparse_csr, blocksize=(3, 1))
            RuntimeError: to_sparse for Strided to SparseCsr conversion does not use specified blocksize

            >>> x = torch.tensor([[[1], [0]], [[0], [0]], [[2], [3]]])
            >>> x.to_sparse(layout=torch.sparse_csr, dense_dim=1)
            tensor(crow_indices=tensor([0, 1, 1, 3]),
                   col_indices=tensor([0, 0, 1]),
                   values=tensor([[1],
                                  [2],
                                  [3]]), size=(3, 2, 1), nnz=3, layout=torch.sparse_csr)
        """
    def to_sparse_bsc(self, blocksize: _int | _size, dense_dim: _int | None = ...) -> Tensor:
        """
        to_sparse_bsc(blocksize, dense_dim) -> Tensor

        Convert a tensor to a block sparse column (BSC) storage format of
        given blocksize.  If the :attr:`self` is strided, then the number of
        dense dimensions could be specified, and a hybrid BSC tensor will be
        created, with `dense_dim` dense dimensions and `self.dim() - 2 -
        dense_dim` batch dimension.

        Args:

            blocksize (list, tuple, :class:`torch.Size`, optional): Block size
              of the resulting BSC tensor. A block size must be a tuple of
              length two such that its items evenly divide the two sparse
              dimensions.

            dense_dim (int, optional): Number of dense dimensions of the
              resulting BSC tensor.  This argument should be used only if
              :attr:`self` is a strided tensor, and must be a value between 0
              and dimension of :attr:`self` tensor minus two.

        Example::

            >>> dense = torch.randn(10, 10)
            >>> sparse = dense.to_sparse_csr()
            >>> sparse_bsc = sparse.to_sparse_bsc((5, 5))
            >>> sparse_bsc.row_indices()
            tensor([0, 1, 0, 1])

            >>> dense = torch.zeros(4, 3, 1)
            >>> dense[0:2, 0] = dense[0:2, 2] = dense[2:4, 1] = 1
            >>> dense.to_sparse_bsc((2, 1), 1)
            tensor(ccol_indices=tensor([0, 1, 2, 3]),
                   row_indices=tensor([0, 1, 0]),
                   values=tensor([[[[1.]],

                                   [[1.]]],


                                  [[[1.]],

                                   [[1.]]],


                                  [[[1.]],

                                   [[1.]]]]), size=(4, 3, 1), nnz=3,
                   layout=torch.sparse_bsc)
        """
    def to_sparse_bsr(self, blocksize: _int | _size, dense_dim: _int | None = ...) -> Tensor:
        """
        to_sparse_bsr(blocksize, dense_dim) -> Tensor

        Convert a tensor to a block sparse row (BSR) storage format of given
        blocksize.  If the :attr:`self` is strided, then the number of dense
        dimensions could be specified, and a hybrid BSR tensor will be
        created, with `dense_dim` dense dimensions and `self.dim() - 2 -
        dense_dim` batch dimension.

        Args:

            blocksize (list, tuple, :class:`torch.Size`, optional): Block size
              of the resulting BSR tensor. A block size must be a tuple of
              length two such that its items evenly divide the two sparse
              dimensions.

            dense_dim (int, optional): Number of dense dimensions of the
              resulting BSR tensor.  This argument should be used only if
              :attr:`self` is a strided tensor, and must be a value between 0
              and dimension of :attr:`self` tensor minus two.

        Example::

            >>> dense = torch.randn(10, 10)
            >>> sparse = dense.to_sparse_csr()
            >>> sparse_bsr = sparse.to_sparse_bsr((5, 5))
            >>> sparse_bsr.col_indices()
            tensor([0, 1, 0, 1])

            >>> dense = torch.zeros(4, 3, 1)
            >>> dense[0:2, 0] = dense[0:2, 2] = dense[2:4, 1] = 1
            >>> dense.to_sparse_bsr((2, 1), 1)
            tensor(crow_indices=tensor([0, 2, 3]),
                   col_indices=tensor([0, 2, 1]),
                   values=tensor([[[[1.]],

                                   [[1.]]],


                                  [[[1.]],

                                   [[1.]]],


                                  [[[1.]],

                                   [[1.]]]]), size=(4, 3, 1), nnz=3,
                   layout=torch.sparse_bsr)
        """
    def to_sparse_csc(self, dense_dim: _int | None = ...) -> Tensor:
        """
        to_sparse_csc() -> Tensor

        Convert a tensor to compressed column storage (CSC) format.  Except
        for strided tensors, only works with 2D tensors.  If the :attr:`self`
        is strided, then the number of dense dimensions could be specified,
        and a hybrid CSC tensor will be created, with `dense_dim` dense
        dimensions and `self.dim() - 2 - dense_dim` batch dimension.

        Args:

            dense_dim (int, optional): Number of dense dimensions of the
              resulting CSC tensor.  This argument should be used only if
              :attr:`self` is a strided tensor, and must be a value between 0
              and dimension of :attr:`self` tensor minus two.

        Example::

            >>> dense = torch.randn(5, 5)
            >>> sparse = dense.to_sparse_csc()
            >>> sparse._nnz()
            25

            >>> dense = torch.zeros(3, 3, 1, 1)
            >>> dense[0, 0] = dense[1, 2] = dense[2, 1] = 1
            >>> dense.to_sparse_csc(dense_dim=2)
            tensor(ccol_indices=tensor([0, 1, 2, 3]),
                   row_indices=tensor([0, 2, 1]),
                   values=tensor([[[1.]],

                                  [[1.]],

                                  [[1.]]]), size=(3, 3, 1, 1), nnz=3,
                   layout=torch.sparse_csc)
        """
    def to_sparse_csr(self, dense_dim: _int | None = ...) -> Tensor:
        """
        to_sparse_csr(dense_dim=None) -> Tensor

        Convert a tensor to compressed row storage format (CSR).  Except for
        strided tensors, only works with 2D tensors.  If the :attr:`self` is
        strided, then the number of dense dimensions could be specified, and a
        hybrid CSR tensor will be created, with `dense_dim` dense dimensions
        and `self.dim() - 2 - dense_dim` batch dimension.

        Args:

            dense_dim (int, optional): Number of dense dimensions of the
              resulting CSR tensor.  This argument should be used only if
              :attr:`self` is a strided tensor, and must be a value between 0
              and dimension of :attr:`self` tensor minus two.

        Example::

            >>> dense = torch.randn(5, 5)
            >>> sparse = dense.to_sparse_csr()
            >>> sparse._nnz()
            25

            >>> dense = torch.zeros(3, 3, 1, 1)
            >>> dense[0, 0] = dense[1, 2] = dense[2, 1] = 1
            >>> dense.to_sparse_csr(dense_dim=2)
            tensor(crow_indices=tensor([0, 1, 2, 3]),
                   col_indices=tensor([0, 2, 1]),
                   values=tensor([[[1.]],

                                  [[1.]],

                                  [[1.]]]), size=(3, 3, 1, 1), nnz=3,
                   layout=torch.sparse_csr)
        """
    def tolist(self) -> list[Number]:
        """
        tolist() -> list or number

        Returns the tensor as a (nested) list. For scalars, a standard
        Python number is returned, just like with :meth:`~Tensor.item`.
        Tensors are automatically moved to the CPU first if necessary.

        This operation is not differentiable.

        Examples::

            >>> a = torch.randn(2, 2)
            >>> a.tolist()
            [[0.012766935862600803, 0.5415473580360413],
             [-0.08909505605697632, 0.7729271650314331]]
            >>> a[0,0].tolist()
            0.012766935862600803
        """
    def topk(
        self, k: _int | SymInt, dim: _int = ..., largest: _bool = ..., sorted: _bool = ...
    ) -> torch.return_types.topk:
        """
        topk(k, dim=None, largest=True, sorted=True) -> (Tensor, LongTensor)

        See :func:`torch.topk`
        """
    def trace(self) -> Tensor:
        """
        trace() -> Tensor

        See :func:`torch.trace`
        """
    @overload
    def transpose(self, dim0: _int, dim1: _int) -> Tensor:
        """
        transpose(dim0, dim1) -> Tensor

        See :func:`torch.transpose`
        """
    @overload
    def transpose(self, dim0: str | EllipsisType | None, dim1: str | EllipsisType | None) -> Tensor:
        """
        transpose(dim0, dim1) -> Tensor

        See :func:`torch.transpose`
        """
    def transpose_(self, dim0: _int, dim1: _int) -> Tensor:
        """
        transpose_(dim0, dim1) -> Tensor

        In-place version of :meth:`~Tensor.transpose`
        """
    def triangular_solve(
        self, A: Tensor, upper: _bool = ..., transpose: _bool = ..., unitriangular: _bool = ...
    ) -> torch.return_types.triangular_solve:
        """
        triangular_solve(A, upper=True, transpose=False, unitriangular=False) -> (Tensor, Tensor)

        See :func:`torch.triangular_solve`
        """
    def tril(self, diagonal: _int = ...) -> Tensor:
        """
        tril(diagonal=0) -> Tensor

        See :func:`torch.tril`
        """
    def tril_(self, diagonal: _int = ...) -> Tensor:
        """
        tril_(diagonal=0) -> Tensor

        In-place version of :meth:`~Tensor.tril`
        """
    def triu(self, diagonal: _int = ...) -> Tensor:
        """
        triu(diagonal=0) -> Tensor

        See :func:`torch.triu`
        """
    def triu_(self, diagonal: _int = ...) -> Tensor:
        """
        triu_(diagonal=0) -> Tensor

        In-place version of :meth:`~Tensor.triu`
        """
    def true_divide(
        self, other: Tensor | Number | torch.SymInt | torch.SymFloat, *, out: Tensor | None = ...
    ) -> Tensor:
        """
        true_divide(value) -> Tensor

        See :func:`torch.true_divide`
        """
    def true_divide_(self, other: Tensor | Number | torch.SymInt | torch.SymFloat) -> Tensor:
        """
        true_divide_(value) -> Tensor

        In-place version of :meth:`~Tensor.true_divide_`
        """
    def trunc(self) -> Tensor:
        """
        trunc() -> Tensor

        See :func:`torch.trunc`
        """
    def trunc_(self) -> Tensor:
        """
        trunc_() -> Tensor

        In-place version of :meth:`~Tensor.trunc`
        """
    @overload
    def type(self, dtype: None = ..., non_blocking: _bool = ...) -> str:
        """
        type(dtype=None, non_blocking=False, **kwargs) -> str or Tensor
        Returns the type if `dtype` is not provided, else casts this object to
        the specified type.

        If this is already of the correct type, no copy is performed and the
        original object is returned.

        Args:
            dtype (dtype or string): The desired type
            non_blocking (bool): If ``True``, and the source is in pinned memory
                and destination is on the GPU or vice versa, the copy is performed
                asynchronously with respect to the host. Otherwise, the argument
                has no effect.
            **kwargs: For compatibility, may contain the key ``async`` in place of
                the ``non_blocking`` argument. The ``async`` arg is deprecated.
        """
    @overload
    def type(self, dtype: str | _dtype, non_blocking: _bool = ...) -> Tensor:
        """
        type(dtype=None, non_blocking=False, **kwargs) -> str or Tensor
        Returns the type if `dtype` is not provided, else casts this object to
        the specified type.

        If this is already of the correct type, no copy is performed and the
        original object is returned.

        Args:
            dtype (dtype or string): The desired type
            non_blocking (bool): If ``True``, and the source is in pinned memory
                and destination is on the GPU or vice versa, the copy is performed
                asynchronously with respect to the host. Otherwise, the argument
                has no effect.
            **kwargs: For compatibility, may contain the key ``async`` in place of
                the ``non_blocking`` argument. The ``async`` arg is deprecated.
        """
    def type_as(self, other: Tensor) -> Tensor:
        """
        type_as(tensor) -> Tensor

        Returns this tensor cast to the type of the given tensor.

        This is a no-op if the tensor is already of the correct type. This is
        equivalent to ``self.type(tensor.type())``

        Args:
            tensor (Tensor): the tensor which has the desired type
        """
    @overload
    def unbind(self, dim: _int = ...) -> tuple[Tensor, ...]:
        """
        unbind(dim=0) -> seq

        See :func:`torch.unbind`
        """
    @overload
    def unbind(self, dim: str | EllipsisType | None) -> tuple[Tensor, ...]:
        """
        unbind(dim=0) -> seq

        See :func:`torch.unbind`
        """
    @overload
    def unflatten(
        self, dim: str | EllipsisType | None, sizes: Sequence[_int | SymInt], names: Sequence[str | EllipsisType | None]
    ) -> Tensor: ...
    @overload
    def unflatten(self, dim: _int, sizes: Sequence[_int | SymInt]) -> Tensor: ...
    def unfold(self, dimension: _int, size: _int, step: _int) -> Tensor:
        """
        unfold(dimension, size, step) -> Tensor

        Returns a view of the original tensor which contains all slices of size :attr:`size` from
        :attr:`self` tensor in the dimension :attr:`dimension`.

        Step between two slices is given by :attr:`step`.

        If `sizedim` is the size of dimension :attr:`dimension` for :attr:`self`, the size of
        dimension :attr:`dimension` in the returned tensor will be
        `(sizedim - size) / step + 1`.

        An additional dimension of size :attr:`size` is appended in the returned tensor.

        Args:
            dimension (int): dimension in which unfolding happens
            size (int): the size of each slice that is unfolded
            step (int): the step between each slice

        Example::

            >>> x = torch.arange(1., 8)
            >>> x
            tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
            >>> x.unfold(0, 2, 1)
            tensor([[ 1.,  2.],
                    [ 2.,  3.],
                    [ 3.,  4.],
                    [ 4.,  5.],
                    [ 5.,  6.],
                    [ 6.,  7.]])
            >>> x.unfold(0, 2, 2)
            tensor([[ 1.,  2.],
                    [ 3.,  4.],
                    [ 5.,  6.]])
        """
    def uniform_(self, from_: _float = ..., to: _float = ..., *, generator: Generator | None = ...) -> Tensor:
        r"""
        uniform_(from=0, to=1, *, generator=None) -> Tensor

        Fills :attr:`self` tensor with numbers sampled from the continuous uniform
        distribution:

        .. math::
            f(x) = \dfrac{1}{\text{to} - \text{from}}
        """
    def unsafe_chunk(self, chunks: _int, dim: _int = ...) -> tuple[Tensor, ...]:
        """
        unsafe_chunk(chunks, dim=0) -> List of Tensors

        See :func:`torch.unsafe_chunk`
        """
    def unsafe_split(self, split_size: _int | SymInt, dim: _int = ...) -> tuple[Tensor, ...]:
        """
        unsafe_split(split_size, dim=0) -> List of Tensors

        See :func:`torch.unsafe_split`
        """
    def unsafe_split_with_sizes(self, split_sizes: Sequence[_int | SymInt], dim: _int = ...) -> tuple[Tensor, ...]: ...
    def unsqueeze(self, dim: _int) -> Tensor:
        """
        unsqueeze(dim) -> Tensor

        See :func:`torch.unsqueeze`
        """
    def unsqueeze_(self, dim: _int) -> Tensor:
        """
        unsqueeze_(dim) -> Tensor

        In-place version of :meth:`~Tensor.unsqueeze`
        """
    def values(self) -> Tensor:
        """
        values() -> Tensor

        Return the values tensor of a :ref:`sparse COO tensor <sparse-coo-docs>`.

        .. warning::
          Throws an error if :attr:`self` is not a sparse COO tensor.

        See also :meth:`Tensor.indices`.

        .. note::
          This method can only be called on a coalesced sparse tensor. See
          :meth:`Tensor.coalesce` for details.
        """
    @overload
    def var(self, dim: _int | _size | None, unbiased: _bool = ..., keepdim: _bool = ...) -> Tensor:
        """
        var(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.var`
        """
    @overload
    def var(
        self, dim: _int | _size | None = ..., *, correction: Number | _complex | None = ..., keepdim: _bool = ...
    ) -> Tensor:
        """
        var(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.var`
        """
    @overload
    def var(self, unbiased: _bool = ...) -> Tensor:
        """
        var(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.var`
        """
    @overload
    def var(self, dim: Sequence[str | EllipsisType | None], unbiased: _bool = ..., keepdim: _bool = ...) -> Tensor:
        """
        var(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.var`
        """
    @overload
    def var(
        self,
        dim: Sequence[str | EllipsisType | None],
        *,
        correction: Number | _complex | None = ...,
        keepdim: _bool = ...,
    ) -> Tensor:
        """
        var(dim=None, *, correction=1, keepdim=False) -> Tensor

        See :func:`torch.var`
        """
    def vdot(self, other: Tensor) -> Tensor:
        """
        vdot(other) -> Tensor

        See :func:`torch.vdot`
        """
    @overload
    def view(self, dtype: _dtype) -> Tensor:
        r"""
        view(*shape) -> Tensor

        Returns a new tensor with the same data as the :attr:`self` tensor but of a
        different :attr:`shape`.

        The returned tensor shares the same data and must have the same number
        of elements, but may have a different size. For a tensor to be viewed, the new
        view size must be compatible with its original size and stride, i.e., each new
        view dimension must either be a subspace of an original dimension, or only span
        across original dimensions :math:`d, d+1, \dots, d+k` that satisfy the following
        contiguity-like condition that :math:`\forall i = d, \dots, d+k-1`,

        .. math::

          \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]

        Otherwise, it will not be possible to view :attr:`self` tensor as :attr:`shape`
        without copying it (e.g., via :meth:`contiguous`). When it is unclear whether a
        :meth:`view` can be performed, it is advisable to use :meth:`reshape`, which
        returns a view if the shapes are compatible, and copies (equivalent to calling
        :meth:`contiguous`) otherwise.

        Args:
            shape (torch.Size or int...): the desired size

        Example::

            >>> x = torch.randn(4, 4)
            >>> x.size()
            torch.Size([4, 4])
            >>> y = x.view(16)
            >>> y.size()
            torch.Size([16])
            >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
            >>> z.size()
            torch.Size([2, 8])

            >>> a = torch.randn(1, 2, 3, 4)
            >>> a.size()
            torch.Size([1, 2, 3, 4])
            >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
            >>> b.size()
            torch.Size([1, 3, 2, 4])
            >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
            >>> c.size()
            torch.Size([1, 3, 2, 4])
            >>> torch.equal(b, c)
            False


        .. method:: view(dtype) -> Tensor
           :noindex:

        Returns a new tensor with the same data as the :attr:`self` tensor but of a
        different :attr:`dtype`.

        If the element size of :attr:`dtype` is different than that of ``self.dtype``,
        then the size of the last dimension of the output will be scaled
        proportionally.  For instance, if :attr:`dtype` element size is twice that of
        ``self.dtype``, then each pair of elements in the last dimension of
        :attr:`self` will be combined, and the size of the last dimension of the output
        will be half that of :attr:`self`. If :attr:`dtype` element size is half that
        of ``self.dtype``, then each element in the last dimension of :attr:`self` will
        be split in two, and the size of the last dimension of the output will be
        double that of :attr:`self`. For this to be possible, the following conditions
        must be true:

            * ``self.dim()`` must be greater than 0.
            * ``self.stride(-1)`` must be 1.

        Additionally, if the element size of :attr:`dtype` is greater than that of
        ``self.dtype``, the following conditions must be true as well:

            * ``self.size(-1)`` must be divisible by the ratio between the element
              sizes of the dtypes.
            * ``self.storage_offset()`` must be divisible by the ratio between the
              element sizes of the dtypes.
            * The strides of all dimensions, except the last dimension, must be
              divisible by the ratio between the element sizes of the dtypes.

        If any of the above conditions are not met, an error is thrown.

        .. warning::

            This overload is not supported by TorchScript, and using it in a Torchscript
            program will cause undefined behavior.


        Args:
            dtype (:class:`torch.dtype`): the desired dtype

        Example::

            >>> x = torch.randn(4, 4)
            >>> x
            tensor([[ 0.9482, -0.0310,  1.4999, -0.5316],
                    [-0.1520,  0.7472,  0.5617, -0.8649],
                    [-2.4724, -0.0334, -0.2976, -0.8499],
                    [-0.2109,  1.9913, -0.9607, -0.6123]])
            >>> x.dtype
            torch.float32

            >>> y = x.view(torch.int32)
            >>> y
            tensor([[ 1064483442, -1124191867,  1069546515, -1089989247],
                    [-1105482831,  1061112040,  1057999968, -1084397505],
                    [-1071760287, -1123489973, -1097310419, -1084649136],
                    [-1101533110,  1073668768, -1082790149, -1088634448]],
                dtype=torch.int32)
            >>> y[0, 0] = 1000000000
            >>> x
            tensor([[ 0.0047, -0.0310,  1.4999, -0.5316],
                    [-0.1520,  0.7472,  0.5617, -0.8649],
                    [-2.4724, -0.0334, -0.2976, -0.8499],
                    [-0.2109,  1.9913, -0.9607, -0.6123]])

            >>> x.view(torch.cfloat)
            tensor([[ 0.0047-0.0310j,  1.4999-0.5316j],
                    [-0.1520+0.7472j,  0.5617-0.8649j],
                    [-2.4724-0.0334j, -0.2976-0.8499j],
                    [-0.2109+1.9913j, -0.9607-0.6123j]])
            >>> x.view(torch.cfloat).size()
            torch.Size([4, 2])

            >>> x.view(torch.uint8)
            tensor([[  0, 202, 154,  59, 182, 243, 253, 188, 185, 252, 191,  63, 240,  22,
                       8, 191],
                    [227, 165,  27, 190, 128,  72,  63,  63, 146, 203,  15,  63,  22, 106,
                      93, 191],
                    [205,  59,  30, 192, 112, 206,   8, 189,   7,  95, 152, 190,  12, 147,
                      89, 191],
                    [ 43, 246,  87, 190, 235, 226, 254,  63, 111, 240, 117, 191, 177, 191,
                      28, 191]], dtype=torch.uint8)
            >>> x.view(torch.uint8).size()
            torch.Size([4, 16])
        """
    @overload
    def view(self, size: Sequence[_int | SymInt]) -> Tensor:
        r"""
        view(*shape) -> Tensor

        Returns a new tensor with the same data as the :attr:`self` tensor but of a
        different :attr:`shape`.

        The returned tensor shares the same data and must have the same number
        of elements, but may have a different size. For a tensor to be viewed, the new
        view size must be compatible with its original size and stride, i.e., each new
        view dimension must either be a subspace of an original dimension, or only span
        across original dimensions :math:`d, d+1, \dots, d+k` that satisfy the following
        contiguity-like condition that :math:`\forall i = d, \dots, d+k-1`,

        .. math::

          \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]

        Otherwise, it will not be possible to view :attr:`self` tensor as :attr:`shape`
        without copying it (e.g., via :meth:`contiguous`). When it is unclear whether a
        :meth:`view` can be performed, it is advisable to use :meth:`reshape`, which
        returns a view if the shapes are compatible, and copies (equivalent to calling
        :meth:`contiguous`) otherwise.

        Args:
            shape (torch.Size or int...): the desired size

        Example::

            >>> x = torch.randn(4, 4)
            >>> x.size()
            torch.Size([4, 4])
            >>> y = x.view(16)
            >>> y.size()
            torch.Size([16])
            >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
            >>> z.size()
            torch.Size([2, 8])

            >>> a = torch.randn(1, 2, 3, 4)
            >>> a.size()
            torch.Size([1, 2, 3, 4])
            >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
            >>> b.size()
            torch.Size([1, 3, 2, 4])
            >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
            >>> c.size()
            torch.Size([1, 3, 2, 4])
            >>> torch.equal(b, c)
            False


        .. method:: view(dtype) -> Tensor
           :noindex:

        Returns a new tensor with the same data as the :attr:`self` tensor but of a
        different :attr:`dtype`.

        If the element size of :attr:`dtype` is different than that of ``self.dtype``,
        then the size of the last dimension of the output will be scaled
        proportionally.  For instance, if :attr:`dtype` element size is twice that of
        ``self.dtype``, then each pair of elements in the last dimension of
        :attr:`self` will be combined, and the size of the last dimension of the output
        will be half that of :attr:`self`. If :attr:`dtype` element size is half that
        of ``self.dtype``, then each element in the last dimension of :attr:`self` will
        be split in two, and the size of the last dimension of the output will be
        double that of :attr:`self`. For this to be possible, the following conditions
        must be true:

            * ``self.dim()`` must be greater than 0.
            * ``self.stride(-1)`` must be 1.

        Additionally, if the element size of :attr:`dtype` is greater than that of
        ``self.dtype``, the following conditions must be true as well:

            * ``self.size(-1)`` must be divisible by the ratio between the element
              sizes of the dtypes.
            * ``self.storage_offset()`` must be divisible by the ratio between the
              element sizes of the dtypes.
            * The strides of all dimensions, except the last dimension, must be
              divisible by the ratio between the element sizes of the dtypes.

        If any of the above conditions are not met, an error is thrown.

        .. warning::

            This overload is not supported by TorchScript, and using it in a Torchscript
            program will cause undefined behavior.


        Args:
            dtype (:class:`torch.dtype`): the desired dtype

        Example::

            >>> x = torch.randn(4, 4)
            >>> x
            tensor([[ 0.9482, -0.0310,  1.4999, -0.5316],
                    [-0.1520,  0.7472,  0.5617, -0.8649],
                    [-2.4724, -0.0334, -0.2976, -0.8499],
                    [-0.2109,  1.9913, -0.9607, -0.6123]])
            >>> x.dtype
            torch.float32

            >>> y = x.view(torch.int32)
            >>> y
            tensor([[ 1064483442, -1124191867,  1069546515, -1089989247],
                    [-1105482831,  1061112040,  1057999968, -1084397505],
                    [-1071760287, -1123489973, -1097310419, -1084649136],
                    [-1101533110,  1073668768, -1082790149, -1088634448]],
                dtype=torch.int32)
            >>> y[0, 0] = 1000000000
            >>> x
            tensor([[ 0.0047, -0.0310,  1.4999, -0.5316],
                    [-0.1520,  0.7472,  0.5617, -0.8649],
                    [-2.4724, -0.0334, -0.2976, -0.8499],
                    [-0.2109,  1.9913, -0.9607, -0.6123]])

            >>> x.view(torch.cfloat)
            tensor([[ 0.0047-0.0310j,  1.4999-0.5316j],
                    [-0.1520+0.7472j,  0.5617-0.8649j],
                    [-2.4724-0.0334j, -0.2976-0.8499j],
                    [-0.2109+1.9913j, -0.9607-0.6123j]])
            >>> x.view(torch.cfloat).size()
            torch.Size([4, 2])

            >>> x.view(torch.uint8)
            tensor([[  0, 202, 154,  59, 182, 243, 253, 188, 185, 252, 191,  63, 240,  22,
                       8, 191],
                    [227, 165,  27, 190, 128,  72,  63,  63, 146, 203,  15,  63,  22, 106,
                      93, 191],
                    [205,  59,  30, 192, 112, 206,   8, 189,   7,  95, 152, 190,  12, 147,
                      89, 191],
                    [ 43, 246,  87, 190, 235, 226, 254,  63, 111, 240, 117, 191, 177, 191,
                      28, 191]], dtype=torch.uint8)
            >>> x.view(torch.uint8).size()
            torch.Size([4, 16])
        """
    @overload
    def view(self, *size: _int | SymInt) -> Tensor:
        r"""
        view(*shape) -> Tensor

        Returns a new tensor with the same data as the :attr:`self` tensor but of a
        different :attr:`shape`.

        The returned tensor shares the same data and must have the same number
        of elements, but may have a different size. For a tensor to be viewed, the new
        view size must be compatible with its original size and stride, i.e., each new
        view dimension must either be a subspace of an original dimension, or only span
        across original dimensions :math:`d, d+1, \dots, d+k` that satisfy the following
        contiguity-like condition that :math:`\forall i = d, \dots, d+k-1`,

        .. math::

          \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]

        Otherwise, it will not be possible to view :attr:`self` tensor as :attr:`shape`
        without copying it (e.g., via :meth:`contiguous`). When it is unclear whether a
        :meth:`view` can be performed, it is advisable to use :meth:`reshape`, which
        returns a view if the shapes are compatible, and copies (equivalent to calling
        :meth:`contiguous`) otherwise.

        Args:
            shape (torch.Size or int...): the desired size

        Example::

            >>> x = torch.randn(4, 4)
            >>> x.size()
            torch.Size([4, 4])
            >>> y = x.view(16)
            >>> y.size()
            torch.Size([16])
            >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
            >>> z.size()
            torch.Size([2, 8])

            >>> a = torch.randn(1, 2, 3, 4)
            >>> a.size()
            torch.Size([1, 2, 3, 4])
            >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
            >>> b.size()
            torch.Size([1, 3, 2, 4])
            >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
            >>> c.size()
            torch.Size([1, 3, 2, 4])
            >>> torch.equal(b, c)
            False


        .. method:: view(dtype) -> Tensor
           :noindex:

        Returns a new tensor with the same data as the :attr:`self` tensor but of a
        different :attr:`dtype`.

        If the element size of :attr:`dtype` is different than that of ``self.dtype``,
        then the size of the last dimension of the output will be scaled
        proportionally.  For instance, if :attr:`dtype` element size is twice that of
        ``self.dtype``, then each pair of elements in the last dimension of
        :attr:`self` will be combined, and the size of the last dimension of the output
        will be half that of :attr:`self`. If :attr:`dtype` element size is half that
        of ``self.dtype``, then each element in the last dimension of :attr:`self` will
        be split in two, and the size of the last dimension of the output will be
        double that of :attr:`self`. For this to be possible, the following conditions
        must be true:

            * ``self.dim()`` must be greater than 0.
            * ``self.stride(-1)`` must be 1.

        Additionally, if the element size of :attr:`dtype` is greater than that of
        ``self.dtype``, the following conditions must be true as well:

            * ``self.size(-1)`` must be divisible by the ratio between the element
              sizes of the dtypes.
            * ``self.storage_offset()`` must be divisible by the ratio between the
              element sizes of the dtypes.
            * The strides of all dimensions, except the last dimension, must be
              divisible by the ratio between the element sizes of the dtypes.

        If any of the above conditions are not met, an error is thrown.

        .. warning::

            This overload is not supported by TorchScript, and using it in a Torchscript
            program will cause undefined behavior.


        Args:
            dtype (:class:`torch.dtype`): the desired dtype

        Example::

            >>> x = torch.randn(4, 4)
            >>> x
            tensor([[ 0.9482, -0.0310,  1.4999, -0.5316],
                    [-0.1520,  0.7472,  0.5617, -0.8649],
                    [-2.4724, -0.0334, -0.2976, -0.8499],
                    [-0.2109,  1.9913, -0.9607, -0.6123]])
            >>> x.dtype
            torch.float32

            >>> y = x.view(torch.int32)
            >>> y
            tensor([[ 1064483442, -1124191867,  1069546515, -1089989247],
                    [-1105482831,  1061112040,  1057999968, -1084397505],
                    [-1071760287, -1123489973, -1097310419, -1084649136],
                    [-1101533110,  1073668768, -1082790149, -1088634448]],
                dtype=torch.int32)
            >>> y[0, 0] = 1000000000
            >>> x
            tensor([[ 0.0047, -0.0310,  1.4999, -0.5316],
                    [-0.1520,  0.7472,  0.5617, -0.8649],
                    [-2.4724, -0.0334, -0.2976, -0.8499],
                    [-0.2109,  1.9913, -0.9607, -0.6123]])

            >>> x.view(torch.cfloat)
            tensor([[ 0.0047-0.0310j,  1.4999-0.5316j],
                    [-0.1520+0.7472j,  0.5617-0.8649j],
                    [-2.4724-0.0334j, -0.2976-0.8499j],
                    [-0.2109+1.9913j, -0.9607-0.6123j]])
            >>> x.view(torch.cfloat).size()
            torch.Size([4, 2])

            >>> x.view(torch.uint8)
            tensor([[  0, 202, 154,  59, 182, 243, 253, 188, 185, 252, 191,  63, 240,  22,
                       8, 191],
                    [227, 165,  27, 190, 128,  72,  63,  63, 146, 203,  15,  63,  22, 106,
                      93, 191],
                    [205,  59,  30, 192, 112, 206,   8, 189,   7,  95, 152, 190,  12, 147,
                      89, 191],
                    [ 43, 246,  87, 190, 235, 226, 254,  63, 111, 240, 117, 191, 177, 191,
                      28, 191]], dtype=torch.uint8)
            >>> x.view(torch.uint8).size()
            torch.Size([4, 16])
        """
    def view_as(self, other: Tensor) -> Tensor:
        """
        view_as(other) -> Tensor

        View this tensor as the same size as :attr:`other`.
        ``self.view_as(other)`` is equivalent to ``self.view(other.size())``.

        Please see :meth:`~Tensor.view` for more information about ``view``.

        Args:
            other (:class:`torch.Tensor`): The result tensor has the same size
                as :attr:`other`.
        """
    @overload
    def vsplit(self, sections: _int) -> tuple[Tensor, ...]:
        """
        vsplit(split_size_or_sections) -> List of Tensors

        See :func:`torch.vsplit`
        """
    @overload
    def vsplit(self, indices: _size) -> tuple[Tensor, ...]:
        """
        vsplit(split_size_or_sections) -> List of Tensors

        See :func:`torch.vsplit`
        """
    @overload
    def vsplit(self, *indices: _int) -> tuple[Tensor, ...]:
        """
        vsplit(split_size_or_sections) -> List of Tensors

        See :func:`torch.vsplit`
        """
    @overload
    def where(self, condition: Tensor, other: Tensor) -> Tensor:
        """
        where(condition, y) -> Tensor

        ``self.where(condition, y)`` is equivalent to ``torch.where(condition, self, y)``.
        See :func:`torch.where`
        """
    @overload
    def where(self, condition: Tensor, other: Number | _complex) -> Tensor:
        """
        where(condition, y) -> Tensor

        ``self.where(condition, y)`` is equivalent to ``torch.where(condition, self, y)``.
        See :func:`torch.where`
        """
    @overload
    def xlogy(self, other: Tensor) -> Tensor:
        """
        xlogy(other) -> Tensor

        See :func:`torch.xlogy`
        """
    @overload
    def xlogy(self, other: Number | _complex) -> Tensor:
        """
        xlogy(other) -> Tensor

        See :func:`torch.xlogy`
        """
    @overload
    def xlogy_(self, other: Tensor) -> Tensor:
        """
        xlogy_(other) -> Tensor

        In-place version of :meth:`~Tensor.xlogy`
        """
    @overload
    def xlogy_(self, other: Number | _complex) -> Tensor:
        """
        xlogy_(other) -> Tensor

        In-place version of :meth:`~Tensor.xlogy`
        """
    def xpu(
        self,
        device: _device | _int | str | None = ...,
        non_blocking: _bool = ...,
        memory_format: torch.memory_format = ...,
    ) -> Tensor:
        """
        xpu(device=None, non_blocking=False, memory_format=torch.preserve_format) -> Tensor

        Returns a copy of this object in XPU memory.

        If this object is already in XPU memory and on the correct device,
        then no copy is performed and the original object is returned.

        Args:
            device (:class:`torch.device`, optional): The destination XPU device.
                Defaults to the current XPU device.
            non_blocking (bool, optional): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host.
                Otherwise, the argument has no effect. Default: ``False``.
            memory_format (:class:`torch.memory_format`, optional): the desired memory format of
                returned Tensor. Default: ``torch.preserve_format``.
        """
    def zero_(self) -> Tensor:
        """
        zero_() -> Tensor

        Fills :attr:`self` tensor with zeros.
        """

_TensorBase = TensorBase

class _cuda_CUDAAllocator_AllocatorState: ...
class _cuda_CUDAAllocator: ...

class _CudaDeviceProperties:
    name: str
    major: _int
    minor: _int
    multi_processor_count: _int
    total_memory: _int
    is_integrated: _int
    is_multi_gpu_board: _int
    max_threads_per_multi_processor: _int
    gcnArchName: str
    warp_size: _int
    uuid: str
    L2_cache_size: _int

class _SDPAParams:
    query: Tensor
    key: Tensor
    value: Tensor
    attn_mask: Tensor | None
    dropout: _float
    is_causal: _bool
    enable_gqa: _bool
    def __init__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None,
        dropout: _float,
        is_causal: _bool,
        enable_gqa: _bool,
    ) -> None:
        """__init__(self: torch._C._SDPAParams, arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor | None, arg4: typing.SupportsFloat, arg5: bool, arg6: bool) -> None"""

class _SDPBackend(Enum):
    """
    An enum-like class that contains the different backends for scaled dot product attention.

    ... warning:: This class is in beta and subject to change.

    This backend class is designed to be used with the sdpa_kernel context manager.See :func: torch.nn.attention.sdpa_kernel for more details.

    Members:

      ERROR

      MATH

      FLASH_ATTENTION

      EFFICIENT_ATTENTION

      CUDNN_ATTENTION

      OVERRIDEABLE
    """

    ERROR = ...
    MATH = ...
    FLASH_ATTENTION = ...
    EFFICIENT_ATTENTION = ...
    CUDNN_ATTENTION = ...
    OVERRIDEABLE = ...

class _CudaStreamBase(Stream):
    stream_id: _int
    device_index: _int
    device_type: _int
    device: _device
    cuda_stream: _int
    priority: _int
    def __new__(
        cls, priority: _int = ..., stream_id: _int = ..., device_index: _int = ..., stream_ptr: _int = ...
    ) -> Self: ...
    def query(self) -> _bool: ...
    def synchronize(self) -> None: ...
    def priority_range(self) -> tuple[_int, _int]: ...

class _CudaEventBase:
    device: _device
    cuda_event: _int
    def __new__(
        cls, enable_timing: _bool = ..., blocking: _bool = ..., interprocess: _bool = ..., external: _bool = ...
    ) -> Self: ...
    @classmethod
    def from_ipc_handle(cls, device: _device, ipc_handle: bytes) -> _CudaEventBase: ...
    def record(self, stream: _CudaStreamBase) -> None: ...
    def wait(self, stream: _CudaStreamBase) -> None: ...
    def query(self) -> _bool: ...
    def elapsed_time(self, other: _CudaEventBase) -> _float: ...
    def synchronize(self) -> None: ...
    def ipc_handle(self) -> bytes: ...

class _CUDAGraph:
    def __new__(cls, keep_graph: _bool = ...) -> Self: ...
    def capture_begin(self, pool: _POOL_HANDLE | None = ..., capture_error_mode: str = ...) -> None: ...
    def capture_end(self) -> None: ...
    def instantiate(self) -> None: ...
    def register_generator_state(self, Generator) -> None: ...
    def replay(self) -> None: ...
    def reset(self) -> None: ...
    def pool(self) -> _POOL_HANDLE: ...
    def enable_debug_mode(self) -> None: ...
    def debug_dump(self, debug_path: str) -> None: ...
    def raw_cuda_graph(self) -> _int: ...
    def raw_cuda_graph_exec(self) -> _int: ...

class _MemPool:
    def __init__(
        self, allocator: _cuda_CUDAAllocator | None = ..., is_user_created: _bool = ..., use_on_oom: _bool = ...
    ) -> None: ...
    @property
    def id(self) -> tuple[_int, _int]: ...
    @property
    def allocator(self) -> _cuda_CUDAAllocator | None: ...
    def use_count(self) -> _int: ...

class _XpuDeviceProperties:
    name: str
    platform_name: str
    vendor: str
    device_id: _int
    driver_version: str
    version: str
    max_compute_units: _int
    gpu_eu_count: _int
    max_work_group_size: _int
    max_num_sub_groups: _int
    sub_group_sizes: list[_int]
    has_fp16: _bool
    has_fp64: _bool
    has_atomic64: _bool
    has_bfloat16_conversions: _bool
    has_subgroup_matrix_multiply_accumulate: _bool
    has_subgroup_matrix_multiply_accumulate_tensor_float32: _bool
    has_subgroup_2d_block_io: _bool
    total_memory: _int
    gpu_subslice_count: _int
    architecture: _int
    type: str
    uuid: Any

class _XpuStreamBase(Stream):
    stream_id: _int
    device_index: _int
    device_type: _int
    device: _device
    sycl_queue: _int
    priority: _int
    def __new__(
        cls, priority: _int = ..., stream_id: _int = ..., device_index: _int = ..., device_type: _int = ...
    ) -> Self: ...
    def query(self) -> _bool: ...
    def synchronize(self) -> None: ...
    @staticmethod
    def priority_range() -> tuple: ...

class _XpuEventBase:
    device: _device
    sycl_event: _int
    def __new__(cls, enable_timing: _bool = ...) -> Self: ...
    def record(self, stream: _XpuEventBase) -> None: ...
    def wait(self, stream: _XpuStreamBase) -> None: ...
    def query(self) -> _bool: ...
    def elapsed_time(self, other: _XpuEventBase) -> _float: ...
    def synchronize(self) -> None: ...

class TracingState:
    def push_scope(self, scope_name: str) -> None:
        """push_scope(self: torch._C.TracingState, arg0: str) -> None"""
    def pop_scope(self) -> None:
        """pop_scope(self: torch._C.TracingState) -> None"""
    def current_scope(self) -> str:
        """current_scope(self: torch._C.TracingState) -> str"""
    def set_graph(self, graph: Graph) -> None:
        """set_graph(self: torch._C.TracingState, arg0: torch._C.Graph) -> None"""
    def graph(self) -> Graph:
        """graph(self: torch._C.TracingState) -> torch._C.Graph"""

class IValue: ...

type Stack = list[IValue]

class JitType:
    annotation_str: str
    def isSubtypeOf(self, other: JitType) -> _bool: ...
    def with_dtype(self, dtype: _dtype) -> JitType: ...
    def with_sizes(self, sizes: list[_int | None]) -> JitType: ...
    def kind(self) -> str: ...
    def scalarType(self) -> str | None: ...
    def getElementType(self) -> JitType: ...
    def dtype(self) -> _dtype | None: ...

class InferredType:
    def __init__(self, arg: JitType | str) -> None:
        """
        __init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: torch._C.InferredType, arg0: torch._C.Type) -> None

        2. __init__(self: torch._C.InferredType, arg0: str) -> None
        """
    def type(self) -> JitType:
        """type(self: torch._C.InferredType) -> torch._C.Type"""
    def success(self) -> _bool:
        """success(self: torch._C.InferredType) -> bool"""
    def reason(self) -> str:
        """reason(self: torch._C.InferredType) -> str"""

class Type(JitType):
    def str(self) -> _str:
        """str(self: torch._C.Type) -> str"""
    def containedTypes(self) -> list[JitType]:
        """containedTypes(self: torch._C.Type) -> list[torch._C.Type]"""
    def dim(self) -> _int | None:
        """dim(self: torch._C.Type) -> object"""
    def undefined(self) -> _bool | None:
        """undefined(self: torch._C.Type) -> object"""
    def sizes(self) -> list[_int] | None:
        """sizes(self: torch._C.Type) -> object"""
    def symbol_sizes(self) -> list[_int] | None: ...
    def varyingSizes(self) -> list[_int | None] | None:
        """varyingSizes(self: torch._C.Type) -> object"""
    def strides(self) -> list[_int] | None:
        """strides(self: torch._C.Type) -> object"""
    def contiguous(self) -> Self:
        """contiguous(self: torch._C.Type) -> torch._C.Type"""
    def device(self) -> _device | None:
        """device(self: torch._C.Type) -> object"""
    def is_interface_type(self) -> _bool:
        """is_interface_type(self: torch._C.Type) -> bool"""
    def requires_grad(self) -> _bool:
        """requires_grad(self: torch._C.Type) -> bool"""
    @property
    def annotation_string(self) -> _str: ...

class AnyType(JitType):
    @staticmethod
    def get() -> AnyType:
        """get() -> torch._C.AnyType"""

class NoneType(JitType):
    @staticmethod
    def get() -> NoneType:
        """get() -> torch._C.NoneType"""

class BoolType(JitType):
    @staticmethod
    def get() -> BoolType:
        """get() -> torch._C.BoolType"""

class FloatType(JitType):
    @staticmethod
    def get() -> FloatType:
        """get() -> torch._C.FloatType"""

class ComplexType(JitType):
    @staticmethod
    def get() -> ComplexType:
        """get() -> torch._C.ComplexType"""

class IntType(JitType):
    @staticmethod
    def get() -> IntType:
        """get() -> torch._C.IntType"""

class SymIntType(JitType):
    @staticmethod
    def get() -> SymIntType:
        """get() -> torch._C.SymIntType"""

class SymBoolType(JitType):
    @staticmethod
    def get() -> SymBoolType:
        """get() -> torch._C.SymBoolType"""

class NumberType(JitType):
    @staticmethod
    def get() -> NumberType:
        """get() -> torch._C.NumberType"""

class StringType(JitType):
    @staticmethod
    def get() -> StringType:
        """get() -> torch._C.StringType"""

class DeviceObjType(JitType):
    @staticmethod
    def get() -> DeviceObjType:
        """get() -> torch._C.DeviceObjType"""

class _GeneratorType(JitType):
    @staticmethod
    def get() -> _GeneratorType:
        """get() -> torch._C._GeneratorType"""

class StreamObjType(JitType):
    @staticmethod
    def get() -> StreamObjType:
        """get() -> torch._C.StreamObjType"""

class ListType(JitType):
    def __init__(self, a: JitType) -> None:
        """__init__(self: torch._C.ListType, arg0: torch._C.Type) -> None"""
    def getElementType(self) -> JitType:
        """getElementType(self: torch._C.ListType) -> torch._C.Type"""
    @staticmethod
    def ofInts() -> ListType:
        """ofInts() -> torch._C.ListType"""
    @staticmethod
    def ofTensors() -> ListType:
        """ofTensors() -> torch._C.ListType"""
    @staticmethod
    def ofFloats() -> ListType:
        """ofFloats() -> torch._C.ListType"""
    @staticmethod
    def ofComplexDoubles() -> ListType:
        """ofComplexDoubles() -> torch._C.ListType"""
    @staticmethod
    def ofBools() -> ListType:
        """ofBools() -> torch._C.ListType"""
    @staticmethod
    def ofStrings() -> ListType:
        """ofStrings() -> torch._C.ListType"""

class DictType(JitType):
    def __init__(self, key: JitType, value: JitType) -> None:
        """__init__(self: torch._C.DictType, arg0: torch._C.Type, arg1: torch._C.Type) -> None"""
    def getKeyType(self) -> JitType:
        """getKeyType(self: torch._C.DictType) -> torch._C.Type"""
    def getValueType(self) -> JitType:
        """getValueType(self: torch._C.DictType) -> torch._C.Type"""

class TupleType(JitType):
    def __init__(self, a: list[JitType | None]) -> None:
        """__init__(self: torch._C.TupleType, arg0: collections.abc.Sequence[torch._C.Type]) -> None"""
    def elements(self) -> list[JitType]:
        """elements(self: torch._C.TupleType) -> list[torch._C.Type]"""

class UnionType(JitType):
    def __init__(self, a: list[JitType]) -> None:
        """__init__(self: torch._C.UnionType, arg0: collections.abc.Sequence[torch._C.Type]) -> None"""

class ClassType(JitType):
    def __init__(self, qualified_name: str) -> None:
        """__init__(self: torch._C.ClassType, arg0: str) -> None"""
    def qualified_name(self) -> str:
        """qualified_name(self: torch._C.ClassType) -> str"""

class InterfaceType(JitType):
    def __init__(self, qualified_name: str) -> None:
        """__init__(self: torch._C.InterfaceType, arg0: str) -> None"""
    def getMethod(self, name: str) -> FunctionSchema | None:
        """getMethod(self: torch._C.InterfaceType, arg0: str) -> torch._C.FunctionSchema"""
    def getMethodNames(self) -> list[str]:
        """getMethodNames(self: torch._C.InterfaceType) -> list[str]"""

JitTypeT = TypeVar("JitTypeT", bound=JitType)

class OptionalType[JitTypeT: JitType](JitType):
    def __init__(self, a: JitTypeT) -> None:
        """__init__(self: torch._C.OptionalType, arg0: torch._C.Type) -> None"""
    def getElementType(self) -> JitTypeT:
        """getElementType(self: torch._C.OptionalType) -> torch._C.Type"""
    @staticmethod
    def ofTensor() -> OptionalType:
        """ofTensor() -> torch._C.Type"""

class FutureType(JitType):
    def __init__(self, a: JitType) -> None:
        """__init__(self: torch._C.FutureType, arg0: torch._C.Type) -> None"""
    def getElementType(self) -> JitType:
        """getElementType(self: torch._C.FutureType) -> torch._C.Type"""

class AwaitType(JitType):
    def __init__(self, a: JitType) -> None:
        """__init__(self: torch._C.AwaitType, arg0: torch._C.Type) -> None"""
    def getElementType(self) -> JitType:
        """getElementType(self: torch._C.AwaitType) -> torch._C.Type"""

class RRefType(JitType):
    def __init__(self, a: JitType) -> None:
        """__init__(self: torch._C.RRefType, arg0: torch._C.Type) -> None"""

class EnumType(JitType):
    def __init__(self, qualified_name: str, value_type: JitType, enum_names_values: list[Any]) -> None:
        """__init__(self: torch._C.EnumType, arg0: str, arg1: torch._C.Type, arg2: collections.abc.Sequence[object]) -> None"""

class TensorType(JitType):
    @classmethod
    def get(cls) -> TensorType:
        """get() -> torch._C.TensorType"""
    @classmethod
    def getInferred(cls) -> TensorType:
        """getInferred() -> torch._C.TensorType"""
    def with_sizes(self, other: list[_int | None] | None) -> TensorType:
        """with_sizes(self: torch._C.Type, arg0: collections.abc.Sequence[typing.SupportsInt | None] | None) -> object"""
    def sizes(self) -> list[_int] | None:
        """sizes(self: torch._C.Type) -> object"""
    def varyingSizes(self) -> list[_int | None] | None:
        """varyingSizes(self: torch._C.Type) -> object"""
    def strides(self) -> list[_int] | None:
        """strides(self: torch._C.Type) -> object"""
    def device(self) -> _device | None:
        """device(self: torch._C.Type) -> object"""
    def dim(self) -> _int:
        """dim(self: torch._C.Type) -> object"""
    def dtype(self) -> _dtype | None:
        """dtype(self: torch._C.Type) -> object"""
    @staticmethod
    def create_from_tensor(t: Tensor) -> TensorType:
        """create_from_tensor(arg0: torch.Tensor) -> torch._C.TensorType"""

class SourceRange: ...
class TreeView: ...

class Ident(TreeView):
    @property
    def name(self) -> str: ...

class ClassDef(TreeView): ...

class Def(TreeView):
    def name(self) -> Ident: ...

class Decl(TreeView): ...

class AcceleratorError(RuntimeError):
    """Exception raised while executing on device"""

class OutOfMemoryError(RuntimeError):
    """Exception raised when device is out of memory"""

class _DistError(RuntimeError):
    """Exception raised when an error occurs in the distributed library"""

class _DistBackendError(RuntimeError):
    """Exception raised when a backend error occurs in distributed"""

class _DistStoreError(RuntimeError):
    """Exception raised when an error occurs in the distributed store"""

class _DistNetworkError(RuntimeError):
    """Exception raised when a network error occurs in distributed"""

class _DistQueueEmptyError(_DistStoreError):
    """Exception raised when an error occurs in the distributed store"""

class CapturedTraceback: ...

def gather_traceback(python: _bool, script: _bool, cpp: _bool) -> CapturedTraceback: ...
def symbolize_tracebacks(tracebacks: list[CapturedTraceback]) -> list[dict[str, Any]]: ...

class _NodeBase:
    _erased: _bool
    _prev: FxNode
    _next: FxNode
    def __init__(self, graph: Any, name: str, op: str, target: Any, return_type: Any) -> None: ...

class _NodeIter(Iterator[FxNode]):
    def __init__(self, root: FxNode, reversed: _bool) -> None: ...
    def __iter__(self) -> Self:
        """Implement iter(self)."""
    def __next__(self) -> FxNode:
        """Implement next(self)."""

class _StaticCudaLauncher: ...
