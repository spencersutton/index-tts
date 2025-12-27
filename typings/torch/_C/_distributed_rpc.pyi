"""distributed rpc bindings"""

from typing import Any, TypeVar, overload

import torch
from torch._C._distributed_c10d import Store

_DEFAULT_INIT_METHOD: str
_DEFAULT_NUM_WORKER_THREADS: int
_UNSET_RPC_TIMEOUT: float
_DEFAULT_RPC_TIMEOUT_SEC: float
_T = TypeVar("_T")

class RpcBackendOptions:
    """
    An abstract structure encapsulating the options passed into the RPC
    backend. An instance of this class can be passed in to
    :meth:`~torch.distributed.rpc.init_rpc` in order to initialize RPC
    with specific configurations, such as the RPC timeout and
    ``init_method`` to be used.
    """

    rpc_timeout: float
    init_method: str
    def __init__(self, rpc_timeout: float = ..., init_method: str = ...) -> None:
        """
        __init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: torch._C._distributed_rpc.RpcBackendOptions) -> None

        2. __init__(self: torch._C._distributed_rpc.RpcBackendOptions, rpc_timeout: typing.SupportsFloat = 60.0, init_method: str = 'env://') -> None
        """

class WorkerInfo:
    """
    A structure that encapsulates information of a worker in the system.
    Contains the name and ID of the worker. This class is not meant to
    be constructed directly, rather, an instance can be retrieved
    through :meth:`~torch.distributed.rpc.get_worker_info` and the
    result can be passed in to functions such as
    :meth:`~torch.distributed.rpc.rpc_sync`, :meth:`~torch.distributed.rpc.rpc_async`,
    :meth:`~torch.distributed.rpc.remote` to avoid copying a string on
    every invocation.
    """
    def __init__(self, name: str, worker_id: int) -> None:
        """__init__(self: torch._C._distributed_rpc.WorkerInfo, name: str, id: typing.SupportsInt) -> None"""
    @property
    def name(self) -> str:
        """The name of the worker."""
    @property
    def id(self) -> int:
        """Globally unique id to identify the worker."""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: torch._C._distributed_rpc.WorkerInfo, arg0: torch._C._distributed_rpc.WorkerInfo) -> bool"""

class RpcAgent:
    def join(self, shutdown: bool = ..., timeout: float = ...):
        """join(self: torch._C._distributed_rpc.RpcAgent, shutdown: bool = False, timeout: typing.SupportsFloat = 0) -> None"""
    def sync(self):
        """sync(self: torch._C._distributed_rpc.RpcAgent) -> None"""
    def shutdown(self):
        """shutdown(self: torch._C._distributed_rpc.RpcAgent) -> None"""
    @overload
    def get_worker_info(self) -> WorkerInfo:
        """
        get_worker_info(*args, **kwargs)
        Overloaded function.

        1. get_worker_info(self: torch._C._distributed_rpc.RpcAgent) -> torch._C._distributed_rpc.WorkerInfo

        2. get_worker_info(self: torch._C._distributed_rpc.RpcAgent, arg0: str) -> torch._C._distributed_rpc.WorkerInfo
        """
    @overload
    def get_worker_info(self, workerName: str) -> WorkerInfo:
        """
        get_worker_info(*args, **kwargs)
        Overloaded function.

        1. get_worker_info(self: torch._C._distributed_rpc.RpcAgent) -> torch._C._distributed_rpc.WorkerInfo

        2. get_worker_info(self: torch._C._distributed_rpc.RpcAgent, arg0: str) -> torch._C._distributed_rpc.WorkerInfo
        """
    def get_worker_infos(self) -> list[WorkerInfo]:
        """get_worker_infos(self: torch._C._distributed_rpc.RpcAgent) -> list[torch._C._distributed_rpc.WorkerInfo]"""
    def get_debug_info(self) -> dict[str, str]:
        """get_debug_info(self: torch._C._distributed_rpc.RpcAgent) -> dict[str, str]"""
    def get_metrics(self) -> dict[str, str]:
        """get_metrics(self: torch._C._distributed_rpc.RpcAgent) -> dict[str, str]"""

class PyRRef[T]:
    """
    A class encapsulating a reference to a value of some type on a remote
    worker. This handle will keep the referenced remote value alive on the
    worker. A ``UserRRef`` will be deleted when 1) no references to it in
    both the application code and in the local RRef context, or 2) the
    application has called a graceful shutdown. Invoking methods on a
    deleted RRef leads to undefined behaviors. RRef implementation only
    offers best-effort error detection, and applications should not use
    ``UserRRefs`` after ``rpc.shutdown()``.

    .. warning::
        RRefs can only be serialized and deserialized by the RPC module.
        Serializing and deserializing RRefs without RPC (e.g., Python
        pickle, torch :meth:`~torch.save` / :meth:`~torch.load`,
        JIT :meth:`~torch.jit.save` / :meth:`~torch.jit.load`, etc.) will
        lead to errors.

    Args:
        value (object): The value to be wrapped by this RRef.
        type_hint (Type, optional): Python type that should be passed to
            ``TorchScript`` compiler as type hint for ``value``.

    Example::
        Following examples skip RPC initialization and shutdown code
        for simplicity. Refer to RPC docs for those details.

        1. Create an RRef using rpc.remote

        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rref = rpc.remote("worker1", torch.add, args=(torch.ones(2), 3))
        >>> # get a copy of value from the RRef
        >>> x = rref.to_here()

        2. Create an RRef from a local object

        >>> import torch
        >>> from torch.distributed.rpc import RRef
        >>> x = torch.zeros(2, 2)
        >>> rref = RRef(x)

        3. Share an RRef with other workers

        >>> # On both worker0 and worker1:
        >>> def f(rref):
        >>>   return rref.to_here() + 1

        >>> # On worker0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> from torch.distributed.rpc import RRef
        >>> rref = RRef(torch.zeros(2, 2))
        >>> # the following RPC shares the rref with worker1, reference
        >>> # count is automatically updated.
        >>> rpc.rpc_sync("worker1", f, args=(rref,))
    """
    def __init__(self, value: _T, type_hint: Any = ...) -> None:
        """__init__(self: torch._C._distributed_rpc.PyRRef, value: object, type_hint: object = None) -> None"""
    def is_owner(self) -> bool:
        """
        is_owner(self: torch._C._distributed_rpc.PyRRef) -> bool


        Returns whether or not the current node is the owner of this
        ``RRef``.
        """
    def confirmed_by_owner(self) -> bool:
        """
        confirmed_by_owner(self: torch._C._distributed_rpc.PyRRef) -> bool


        Returns whether this ``RRef`` has been confirmed by the owner.
        ``OwnerRRef`` always returns true, while ``UserRRef`` only
        returns true when the owner knowns about this ``UserRRef``.
        """
    def owner(self) -> WorkerInfo:
        """
        owner(self: torch._C._distributed_rpc.PyRRef) -> torch._C._distributed_rpc.WorkerInfo


        Returns worker information of the node that owns this ``RRef``.
        """
    def owner_name(self) -> str:
        """
        owner_name(self: torch._C._distributed_rpc.PyRRef) -> str


        Returns worker name of the node that owns this ``RRef``.
        """
    def to_here(self, timeout: float = ...) -> _T:
        """
        to_here(self: torch._C._distributed_rpc.PyRRef, timeout: typing.SupportsFloat = -1.0) -> object


        Blocking call that copies the value of the RRef from the owner
        to the local node and returns it. If the current node is the
        owner, returns a reference to the local value.

        Args:
            timeout (float, optional): Timeout for ``to_here``. If
                the call does not complete within this timeframe, an
                exception indicating so will be raised. If this
                argument is not provided, the default RPC timeout
                (60s) will be used.
        """
    def local_value(self) -> Any:
        """
        local_value(self: torch._C._distributed_rpc.PyRRef) -> object


        If the current node is the owner, returns a reference to the
        local value. Otherwise, throws an exception.
        """
    def rpc_sync(self, timeout: float = ...) -> Any:
        """
        rpc_sync(self: torch._C._distributed_rpc.PyRRef, timeout: typing.SupportsFloat = -1.0) -> object


        Create a helper proxy to easily launch an ``rpc_sync`` using
        the owner of the RRef as the destination to run functions on
        the object referenced by this RRef. More specifically,
        ``rref.rpc_sync().func_name(*args, **kwargs)`` is the same as
        the following:

        >>> def run(rref, func_name, args, kwargs):
        >>>   return getattr(rref.local_value(), func_name)(*args, **kwargs)
        >>>
        >>> rpc.rpc_sync(rref.owner(), run, args=(rref, func_name, args, kwargs))

        Args:
            timeout (float, optional): Timeout for ``rref.rpc_sync()``.
                If the call does not complete within this timeframe, an
                exception indicating so will be raised. If this argument
                is not provided, the default RPC timeout will be used.

        Example::
            >>> from torch.distributed import rpc
            >>> rref = rpc.remote("worker1", torch.add, args=(torch.zeros(2, 2), 1))
            >>> rref.rpc_sync().size()  # returns torch.Size([2, 2])
            >>> rref.rpc_sync().view(1, 4)  # returns tensor([[1., 1., 1., 1.]])
        """
    def rpc_async(self, timeout: float = ...) -> Any:
        """
        rpc_async(self: torch._C._distributed_rpc.PyRRef, timeout: typing.SupportsFloat = -1.0) -> object


        Create a helper proxy to easily launch an ``rpc_async`` using
        the owner of the RRef as the destination to run functions on
        the object referenced by this RRef. More specifically,
        ``rref.rpc_async().func_name(*args, **kwargs)`` is the same as
        the following:

        >>> def run(rref, func_name, args, kwargs):
        >>>   return getattr(rref.local_value(), func_name)(*args, **kwargs)
        >>>
        >>> rpc.rpc_async(rref.owner(), run, args=(rref, func_name, args, kwargs))

        Args:
            timeout (float, optional): Timeout for ``rref.rpc_async()``.
                If the call does not complete within this timeframe, an
                exception indicating so will be raised. If this argument
                is not provided, the default RPC timeout will be used.

        Example::
            >>> from torch.distributed import rpc
            >>> rref = rpc.remote("worker1", torch.add, args=(torch.zeros(2, 2), 1))
            >>> rref.rpc_async().size().wait()  # returns torch.Size([2, 2])
            >>> rref.rpc_async().view(1, 4).wait()  # returns tensor([[1., 1., 1., 1.]])
        """
    def remote(self, timeout: float = ...) -> Any:
        """
        remote(self: torch._C._distributed_rpc.PyRRef, timeout: typing.SupportsFloat = -1.0) -> object


        Create a helper proxy to easily launch a ``remote`` using
        the owner of the RRef as the destination to run functions on
        the object referenced by this RRef. More specifically,
        ``rref.remote().func_name(*args, **kwargs)`` is the same as
        the following:

        >>> def run(rref, func_name, args, kwargs):
        >>>   return getattr(rref.local_value(), func_name)(*args, **kwargs)
        >>>
        >>> rpc.remote(rref.owner(), run, args=(rref, func_name, args, kwargs))

        Args:
            timeout (float, optional): Timeout for ``rref.remote()``. If
                the creation of this :class:`~torch.distributed.rpc.RRef`
                is not successfully completed within the timeout, then the
                next time there is an attempt to use the RRef
                (such as ``to_here``), a timeout will be raised. If not
                provided, the default RPC timeout will be used. Please see
                ``rpc.remote()`` for specific timeout semantics for
                :class:`~torch.distributed.rpc.RRef`.

        Example::
            >>> from torch.distributed import rpc
            >>> rref = rpc.remote("worker1", torch.add, args=(torch.zeros(2, 2), 1))
            >>> rref.remote().size().to_here()  # returns torch.Size([2, 2])
            >>> rref.remote().view(1, 4).to_here()  # returns tensor([[1., 1., 1., 1.]])
        """

class _TensorPipeRpcBackendOptionsBase(RpcBackendOptions):
    num_worker_threads: int
    device_maps: dict[str, dict[torch.device, torch.device]]
    devices: list[torch.device]
    def __init__(
        self,
        num_worker_threads: int,
        _transports: list | None,
        _channels: list | None,
        rpc_timeout: float = ...,
        init_method: str = ...,
        device_maps: dict[str, dict[torch.device, torch.device]] = ...,
        devices: list[torch.device] = ...,
    ) -> None:
        """__init__(self: torch._C._distributed_rpc._TensorPipeRpcBackendOptionsBase, num_worker_threads: typing.SupportsInt = 16, _transports: collections.abc.Sequence[str] | None = None, _channels: collections.abc.Sequence[str] | None = None, rpc_timeout: typing.SupportsFloat = 60.0, init_method: str = 'env://', device_maps: collections.abc.Mapping[str, collections.abc.Mapping[torch.device, torch.device]] = {}, devices: collections.abc.Sequence[torch.device] = []) -> None"""

class TensorPipeAgent(RpcAgent):
    def __init__(
        self,
        store: Store,
        name: str,
        worker_id: int,
        world_size: int | None,
        opts: _TensorPipeRpcBackendOptionsBase,
        reverse_device_maps: dict[str, dict[torch.device, torch.device]],
        devices: list[torch.device],
    ) -> None:
        """__init__(self: torch._C._distributed_rpc.TensorPipeAgent, store: torch.distributed.distributed_c10d.Store, name: str, rank: typing.SupportsInt, world_size: typing.SupportsInt | None, rpc_backend_options: torch._C._distributed_rpc._TensorPipeRpcBackendOptionsBase, reverse_device_maps: collections.abc.Mapping[str, collections.abc.Mapping[torch.device, torch.device]], devices: collections.abc.Sequence[torch.device]) -> None"""
    def join(self, shutdown: bool = ..., timeout: float = ...):
        """join(self: torch._C._distributed_rpc.TensorPipeAgent, shutdown: bool = False, timeout: typing.SupportsFloat = 0) -> None"""
    def shutdown(self):
        """shutdown(self: torch._C._distributed_rpc.TensorPipeAgent) -> None"""
    @overload
    def get_worker_info(self) -> WorkerInfo:
        """
        get_worker_info(*args, **kwargs)
        Overloaded function.

        1. get_worker_info(self: torch._C._distributed_rpc.TensorPipeAgent) -> torch._C._distributed_rpc.WorkerInfo

        2. get_worker_info(self: torch._C._distributed_rpc.TensorPipeAgent, arg0: str) -> torch._C._distributed_rpc.WorkerInfo

        3. get_worker_info(self: torch._C._distributed_rpc.TensorPipeAgent, arg0: typing.SupportsInt) -> torch._C._distributed_rpc.WorkerInfo
        """
    @overload
    def get_worker_info(self, workerName: str) -> WorkerInfo:
        """
        get_worker_info(*args, **kwargs)
        Overloaded function.

        1. get_worker_info(self: torch._C._distributed_rpc.TensorPipeAgent) -> torch._C._distributed_rpc.WorkerInfo

        2. get_worker_info(self: torch._C._distributed_rpc.TensorPipeAgent, arg0: str) -> torch._C._distributed_rpc.WorkerInfo

        3. get_worker_info(self: torch._C._distributed_rpc.TensorPipeAgent, arg0: typing.SupportsInt) -> torch._C._distributed_rpc.WorkerInfo
        """
    @overload
    def get_worker_info(self, id: int) -> WorkerInfo:
        """
        get_worker_info(*args, **kwargs)
        Overloaded function.

        1. get_worker_info(self: torch._C._distributed_rpc.TensorPipeAgent) -> torch._C._distributed_rpc.WorkerInfo

        2. get_worker_info(self: torch._C._distributed_rpc.TensorPipeAgent, arg0: str) -> torch._C._distributed_rpc.WorkerInfo

        3. get_worker_info(self: torch._C._distributed_rpc.TensorPipeAgent, arg0: typing.SupportsInt) -> torch._C._distributed_rpc.WorkerInfo
        """
    def get_worker_infos(self) -> list[WorkerInfo]:
        """get_worker_infos(self: torch._C._distributed_rpc.TensorPipeAgent) -> list[torch._C._distributed_rpc.WorkerInfo]"""
    @property
    def is_static_group(self) -> bool: ...
    @property
    def store(self) -> Store: ...

def get_rpc_timeout() -> float:
    """
    get_rpc_timeout() -> float


    Retrieve the default timeout for all RPCs that was set during RPC initialization.
    The returned value will be in seconds.
    Returns:
      ``float`` indicating the RPC timeout in seconds.
    """

def enable_gil_profiling(flag: bool):
    """
    enable_gil_profiling(arg0: bool) -> None


    Set whether GIL wait times should be enabled or not. This incurs a slight
    overhead cost. Default is disabled for performance reasons.

    Args:
        flag (bool): True to set GIL profiling, False to disable.

    """

class RemoteProfilerManager:
    @staticmethod
    def set_current_profiling_key(key: str):
        """set_current_profiling_key(self: str) -> None"""
