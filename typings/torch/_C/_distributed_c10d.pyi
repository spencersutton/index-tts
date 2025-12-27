"""distributed c10d bindings"""

from datetime import timedelta
from enum import Enum
from typing import Any, overload

import torch
from torch import Tensor
from torch._C import ScriptObject
from torch._C._autograd import DeviceType
from torch.futures import Future

_DEFAULT_FIRST_BUCKET_BYTES: int
_DEFAULT_NO_TIMEOUT: timedelta
_DEFAULT_PG_TIMEOUT: timedelta
_DEFAULT_PG_NCCL_TIMEOUT: timedelta

class BuiltinCommHookType(Enum):
    """
    An enum-like class for built-in communication hooks: ``ALLREDUCE`` and ``FP16_COMPRESS``.

    Members:

      ALLREDUCE

      FP16_COMPRESS
    """

    ALLREDUCE = ...
    FP16_COMPRESS = ...

class GradBucket:
    """
    This class mainly passes a flattened gradient tensor
    (returned by :meth:`~torch.distributed.GradBucket.buffer`)
    to DDP communication hook.
    This tensor can be further decomposed into a list of per-parameter tensors within this bucket
    (returned by :meth:`~torch.distributed.GradBucket.get_per_parameter_tensors`)
    to apply layer-wise operations.
    """
    def index(self) -> int:
        """
        index(self: torch._C._distributed_c10d.GradBucket) -> int


        .. warning::
            Since the buckets are rebuilt after the first iteration, should not rely on the indices at the beginning of training.

        Returns:
            The index of a bucket that stores gradients of a few contiguous layers.
            All the gradients are bucketized.
        """
    def buffer(self) -> Tensor:
        """
        buffer(self: torch._C._distributed_c10d.GradBucket) -> torch.Tensor


        Returns:
            A flattened 1D ``torch.Tensor`` buffer,
            which can be further decomposed into a list of per-parameter tensors within this bucket.
        """
    def gradients(self) -> list[Tensor]:
        """
        gradients(self: torch._C._distributed_c10d.GradBucket) -> list[torch.Tensor]


        Returns:
            A list of ``torch.Tensor``. Each tensor in the list corresponds to a gradient.
        """
    def is_last(self) -> bool:
        """
        is_last(self: torch._C._distributed_c10d.GradBucket) -> bool


        Returns:
            Whether this bucket is the last bucket to allreduce in an iteration.
            This also means that this bucket corresponds to the first few layers in the forward pass.
        """
    def set_buffer(self, tensor: Tensor) -> None:
        """
        set_buffer(self: torch._C._distributed_c10d.GradBucket, buffer: torch.Tensor) -> None


        Replaces the tensor in the bucket with the input tensor buffer.
        """
    def parameters(self) -> list[Tensor]:
        """
        parameters(self: torch._C._distributed_c10d.GradBucket) -> list[torch.Tensor]


        Returns:
            A list of ``torch.Tensor``. Each tensor in the list corresponds to a model
            parameter.
        """

class Reducer:
    def __init__(
        self,
        params: list[Tensor],
        bucket_indices: list[list[int]],
        per_bucket_size_limits: list[int],
        process_group: ProcessGroup,
        expect_sparse_gradients: list[bool] = ...,
        bucket_bytes_cap: int = ...,
        find_unused_parameters: bool = ...,
        gradient_as_bucket_view: bool = ...,
        param_to_name_mapping: dict[int, str] = ...,
        first_bucket_types_cap: int = ...,
        skip_all_reduce_unused_params: bool = ...,
        use_python_reducer: bool = ...,
    ) -> None:
        """__init__(self: torch._C._distributed_c10d.Reducer, params: collections.abc.Sequence[torch.Tensor], bucket_indices: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]], per_bucket_size_limits: collections.abc.Sequence[typing.SupportsInt], process_group: c10d::ProcessGroup, expect_sparse_gradients: collections.abc.Sequence[bool] = [], bucket_bytes_cap: typing.SupportsInt = 26214400, find_unused_parameters: bool = False, gradient_as_bucket_view: bool = False, param_to_name_mapping: collections.abc.Mapping[typing.SupportsInt, str] = {}, first_bucket_bytes_cap: typing.SupportsInt = 1048576, skip_all_reduce_unused_params: bool = False, use_python_reducer: bool = False) -> None"""
    def prepare_for_forward(self) -> None:
        """prepare_for_forward(self: torch._C._distributed_c10d.Reducer) -> None"""
    def prepare_for_backward(self, output: list[Tensor]) -> None:
        """
        prepare_for_backward(*args, **kwargs)
        Overloaded function.

        1. prepare_for_backward(self: torch._C._distributed_c10d.Reducer, arg0: collections.abc.Sequence[torch.Tensor]) -> None

        2. prepare_for_backward(self: torch._C._distributed_c10d.Reducer, arg0: torch.Tensor) -> None
        """
    def get_backward_stats(self) -> list[int]:
        """get_backward_stats(self: torch._C._distributed_c10d.Reducer) -> list[int]"""
    def set_logger(self, logger: Logger) -> None:
        """set_logger(self: torch._C._distributed_c10d.Reducer, arg0: c10d::Logger) -> None"""

class DDPLoggingData:
    strs_map: dict[str, str]
    ints_map: dict[str, int]

class Logger:
    def __init__(self, reducer: Reducer) -> None:
        """__init__(self: torch._C._distributed_c10d.Logger, reducer: torch._C._distributed_c10d.Reducer) -> None"""
    def set_construction_data_and_log(
        self,
        module_name: str,
        device_ids: list[int],
        output_device: int,
        broadcast_buffers: bool,
        has_sync_bn: bool,
        static_graph: bool,
    ):
        """set_construction_data_and_log(self: torch._C._distributed_c10d.Logger, module_name: str, device_ids: collections.abc.Sequence[typing.SupportsInt], output_device: typing.SupportsInt, broadcast_buffers: bool, has_sync_bn: bool, static_graph: bool) -> None"""
    def set_runtime_stats_and_log(self) -> None:
        """set_runtime_stats_and_log(self: torch._C._distributed_c10d.Logger) -> None"""
    def set_error_and_log(self, error: str) -> None:
        """set_error_and_log(self: torch._C._distributed_c10d.Logger, arg0: str) -> None"""

class _WorkerServer:
    def __init__(self, socket_path: str) -> None:
        """__init__(self: torch._C._distributed_c10d._WorkerServer, host_or_file: str, port: typing.SupportsInt = -1) -> None"""
    def shutdown(self) -> None:
        """shutdown(self: torch._C._distributed_c10d._WorkerServer) -> None"""

def get_debug_level():
    """
    get_debug_level() -> torch._C._distributed_c10d.DebugLevel

    Gets the debug level of the torch.distributed package.
    """

def set_debug_level():
    """
    set_debug_level(arg0: torch._C._distributed_c10d.DebugLevel) -> None

    Sets the debug level of the torch.distributed package.
    """

def set_debug_level_from_env():
    """
    set_debug_level_from_env() -> None

    Sets the debug level of the torch.distributed package from the
              ``TORCH_DISTRIBUTED_DEBUG`` environment variable.
    """

class DebugLevel(Enum):
    """
          An enum whose values correspond to different debug levels of the
          torch.distributed package. Currently supporting OFF, INFO, and DETAIL,
          which can be set via the TORCH_DISTRIBUTED_DEBUG environment variable
          or via ``set_debug_level()`` function.


    Members:

      OFF

      INFO

      DETAIL
    """

    OFF = ...
    INFO = ...
    DETAIL = ...

class ReduceOp:
    """
    An enum-like class for available reduction operations: ``SUM``, ``PRODUCT``,
    ``MIN``, ``MAX``, ``BAND``, ``BOR``, ``BXOR``, and ``PREMUL_SUM``.

    ``BAND``, ``BOR``, and ``BXOR`` reductions are not available when
    using the ``NCCL`` backend.

    ``AVG`` divides values by the world size before summing across ranks.
    ``AVG`` is only available with the ``NCCL`` backend,
    and only for NCCL versions 2.10 or later.

    ``PREMUL_SUM`` multiplies inputs by a given scalar locally before reduction.
    ``PREMUL_SUM`` is only available with the ``NCCL`` backend,
    and only available for NCCL versions 2.11 or later. Users are supposed to
    use ``torch.distributed._make_nccl_premul_sum``.

    Additionally, ``MAX``, ``MIN`` and ``PRODUCT`` are not supported for complex tensors.

    The values of this class can be accessed as attributes, e.g., ``ReduceOp.SUM``.
    They are used in specifying strategies for reduction collectives, e.g.,
    :func:`reduce`.

    This class does not support ``__members__`` property.
    """
    def __init__(self, op: RedOpType) -> None:
        """__init__(self: torch._C._distributed_c10d.ReduceOp, arg0: c10d::ReduceOp::RedOpType) -> None"""

    SUM: RedOpType = ...
    AVG: RedOpType = ...
    PRODUCT: RedOpType = ...
    MIN: RedOpType = ...
    MAX: RedOpType = ...
    BAND: RedOpType = ...
    BOR: RedOpType = ...
    BXOR: RedOpType = ...
    PREMUL_SUM: RedOpType = ...
    UNUSED: RedOpType = ...
    class RedOpType(Enum):
        """
        Members:

        SUM

        AVG

        PRODUCT

        MIN

        MAX

        BAND

        BOR

        BXOR

        PREMUL_SUM
        """

class BroadcastOptions:
    rootRank: int
    rootTensor: int
    timeout: timedelta
    asyncOp: bool

class AllreduceOptions:
    reduceOp: ReduceOp
    timeout: timedelta
    asyncOp: bool
    sparseIndices: Tensor | None

class AllreduceCoalescedOptions(AllreduceOptions): ...

class ReduceOptions:
    reduceOp: ReduceOp
    rootRank: int
    rootTensor: int
    timeout: timedelta
    asyncOp: bool

class AllgatherOptions:
    timeout: timedelta
    asyncOp: bool

class GatherOptions:
    rootRank: int
    timeout: timedelta
    asyncOp: bool

class ScatterOptions:
    rootRank: int
    timeout: timedelta
    asyncOp: bool

class ReduceScatterOptions:
    reduceOp: ReduceOp
    timeout: timedelta
    asyncOp: bool

class BarrierOptions:
    device_ids: list[int]
    device: torch.device
    timeout: timedelta
    asyncOp: bool

class AllToAllOptions:
    timeout: timedelta
    asyncOp: bool

class Store:
    """
    Base class for all store implementations, such as the 3 provided by PyTorch
    distributed: (:class:`~torch.distributed.TCPStore`, :class:`~torch.distributed.FileStore`,
    and :class:`~torch.distributed.HashStore`).
    """
    def set(self, key: str, value: str):
        """
        set(self: torch._C._distributed_c10d.Store, arg0: str, arg1: str) -> None


        Inserts the key-value pair into the store based on the supplied ``key`` and
        ``value``. If ``key`` already exists in the store, it will overwrite the old
        value with the new supplied ``value``.

        Arguments:
            key (str): The key to be added to the store.
            value (str): The value associated with ``key`` to be added to the store.

        Example::
            >>> import torch.distributed as dist
            >>> from datetime import timedelta
            >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
            >>> store.set("first_key", "first_value")
            >>> # Should return "first_value"
            >>> store.get("first_key")
        """
    def get(self, key: str) -> bytes:
        """
        get(self: torch._C._distributed_c10d.Store, arg0: str) -> bytes


        Retrieves the value associated with the given ``key`` in the store. If ``key`` is not
        present in the store, the function will wait for ``timeout``, which is defined
        when initializing the store, before throwing an exception.

        Arguments:
            key (str): The function will return the value associated with this key.

        Returns:
            Value associated with ``key`` if ``key`` is in the store.

        Example::
            >>> import torch.distributed as dist
            >>> from datetime import timedelta
            >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
            >>> store.set("first_key", "first_value")
            >>> # Should return "first_value"
            >>> store.get("first_key")
        """
    def add(self, key: str, value: int) -> int:
        """
        add(self: torch._C._distributed_c10d.Store, arg0: str, arg1: typing.SupportsInt) -> int


        The first call to add for a given ``key`` creates a counter associated
        with ``key`` in the store, initialized to ``amount``. Subsequent calls to add
        with the same ``key`` increment the counter by the specified ``amount``.
        Calling :meth:`~torch.distributed.store.add` with a key that has already
        been set in the store by :meth:`~torch.distributed.store.set` will result
        in an exception.

        Arguments:
            key (str): The key in the store whose counter will be incremented.
            amount (int): The quantity by which the counter will be incremented.

        Example::
            >>> import torch.distributed as dist
            >>> from datetime import timedelta
            >>> # Using TCPStore as an example, other store types can also be used
            >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
            >>> store.add("first_key", 1)
            >>> store.add("first_key", 6)
            >>> # Should return 7
            >>> store.get("first_key")
        """
    def check(self, keys: list[str]) -> bool:
        """
        check(self: torch._C._distributed_c10d.Store, arg0: collections.abc.Sequence[str]) -> bool


        The call to check whether a given list of ``keys`` have value stored in
        the store. This call immediately returns in normal cases but still suffers
        from some edge deadlock cases, e.g, calling check after TCPStore has been destroyed.
        Calling :meth:`~torch.distributed.store.check` with a list of keys that
        one wants to check whether stored in the store or not.

        Arguments:
            keys (list[str]): The keys to query whether stored in the store.

        Example::
            >>> import torch.distributed as dist
            >>> from datetime import timedelta
            >>> # Using TCPStore as an example, other store types can also be used
            >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
            >>> store.add("first_key", 1)
            >>> # Should return 7
            >>> store.check(["first_key"])
        """
    def compare_set(self, key: str, expected_value: str, desired_value: str) -> bytes:
        """
        compare_set(self: torch._C._distributed_c10d.Store, arg0: str, arg1: str, arg2: str) -> bytes


        Inserts the key-value pair into the store based on the supplied ``key`` and
        performs comparison between ``expected_value`` and ``desired_value`` before inserting. ``desired_value``
        will only be set if ``expected_value`` for the ``key`` already exists in the store or if ``expected_value``
        is an empty string.

        Arguments:
            key (str): The key to be checked in the store.
            expected_value (str): The value associated with ``key`` to be checked before insertion.
            desired_value (str): The value associated with ``key`` to be added to the store.

        Example::
            >>> import torch.distributed as dist
            >>> from datetime import timedelta
            >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
            >>> store.set("key", "first_value")
            >>> store.compare_set("key", "first_value", "second_value")
            >>> # Should return "second_value"
            >>> store.get("key")
        """
    def delete_key(self, key: str) -> bool:
        """
        delete_key(self: torch._C._distributed_c10d.Store, arg0: str) -> bool


        Deletes the key-value pair associated with ``key`` from the store. Returns
        `true` if the key was successfully deleted, and `false` if it was not.

        .. warning::
            The ``delete_key`` API is only supported by the :class:`~torch.distributed.TCPStore` and :class:`~torch.distributed.HashStore`. Using this API
            with the :class:`~torch.distributed.FileStore` will result in an exception.

        Arguments:
            key (str): The key to be deleted from the store

        Returns:
            `True` if ``key`` was deleted, otherwise `False`.

        Example::
            >>> import torch.distributed as dist
            >>> from datetime import timedelta
            >>> # Using TCPStore as an example, HashStore can also be used
            >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
            >>> store.set("first_key")
            >>> # This should return true
            >>> store.delete_key("first_key")
            >>> # This should return false
            >>> store.delete_key("bad_key")
        """
    def num_keys(self) -> int:
        """
        num_keys(self: torch._C._distributed_c10d.Store) -> int


        Returns the number of keys set in the store. Note that this number will typically
        be one greater than the number of keys added by :meth:`~torch.distributed.store.set`
        and :meth:`~torch.distributed.store.add` since one key is used to coordinate all
        the workers using the store.

        .. warning::
            When used with the :class:`~torch.distributed.TCPStore`, ``num_keys`` returns the number of keys written to the underlying file. If the store is destructed and another store is created with the same file, the original keys will be retained.

        Returns:
            The number of keys present in the store.

        Example::
            >>> import torch.distributed as dist
            >>> from datetime import timedelta
            >>> # Using TCPStore as an example, other store types can also be used
            >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
            >>> store.set("first_key", "first_value")
            >>> # This should return 2
            >>> store.num_keys()
        """
    def set_timeout(self, timeout: timedelta):
        """
        set_timeout(self: torch._C._distributed_c10d.Store, arg0: datetime.timedelta) -> None


        Sets the store's default timeout. This timeout is used during initialization and in
        :meth:`~torch.distributed.store.wait` and :meth:`~torch.distributed.store.get`.

        Arguments:
            timeout (timedelta): timeout to be set in the store.

        Example::
            >>> import torch.distributed as dist
            >>> from datetime import timedelta
            >>> # Using TCPStore as an example, other store types can also be used
            >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
            >>> store.set_timeout(timedelta(seconds=10))
            >>> # This will throw an exception after 10 seconds
            >>> store.wait(["bad_key"])
        """
    @overload
    def wait(self, keys: list[str]):
        """
        wait(*args, **kwargs)
        Overloaded function.

        1. wait(self: torch._C._distributed_c10d.Store, arg0: collections.abc.Sequence[str]) -> None


        Waits for each key in ``keys`` to be added to the store. If not all keys are
        set before the ``timeout`` (set during store initialization), then ``wait``
        will throw an exception.

        Arguments:
            keys (list): List of keys on which to wait until they are set in the store.

        Example::
            >>> import torch.distributed as dist
            >>> from datetime import timedelta
            >>> # Using TCPStore as an example, other store types can also be used
            >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
            >>> # This will throw an exception after 30 seconds
            >>> store.wait(["bad_key"])


        2. wait(self: torch._C._distributed_c10d.Store, arg0: collections.abc.Sequence[str], arg1: datetime.timedelta) -> None


        Waits for each key in ``keys`` to be added to the store, and throws an exception
        if the keys have not been set by the supplied ``timeout``.

        Arguments:
            keys (list): List of keys on which to wait until they are set in the store.
            timeout (timedelta): Time to wait for the keys to be added before throwing an exception.

        Example::
            >>> import torch.distributed as dist
            >>> from datetime import timedelta
            >>> # Using TCPStore as an example, other store types can also be used
            >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
            >>> # This will throw an exception after 10 seconds
            >>> store.wait(["bad_key"], timedelta(seconds=10))
        """
    @overload
    def wait(self, keys: list[str], timeout: timedelta):
        """
        wait(*args, **kwargs)
        Overloaded function.

        1. wait(self: torch._C._distributed_c10d.Store, arg0: collections.abc.Sequence[str]) -> None


        Waits for each key in ``keys`` to be added to the store. If not all keys are
        set before the ``timeout`` (set during store initialization), then ``wait``
        will throw an exception.

        Arguments:
            keys (list): List of keys on which to wait until they are set in the store.

        Example::
            >>> import torch.distributed as dist
            >>> from datetime import timedelta
            >>> # Using TCPStore as an example, other store types can also be used
            >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
            >>> # This will throw an exception after 30 seconds
            >>> store.wait(["bad_key"])


        2. wait(self: torch._C._distributed_c10d.Store, arg0: collections.abc.Sequence[str], arg1: datetime.timedelta) -> None


        Waits for each key in ``keys`` to be added to the store, and throws an exception
        if the keys have not been set by the supplied ``timeout``.

        Arguments:
            keys (list): List of keys on which to wait until they are set in the store.
            timeout (timedelta): Time to wait for the keys to be added before throwing an exception.

        Example::
            >>> import torch.distributed as dist
            >>> from datetime import timedelta
            >>> # Using TCPStore as an example, other store types can also be used
            >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
            >>> # This will throw an exception after 10 seconds
            >>> store.wait(["bad_key"], timedelta(seconds=10))
        """
    def queue_pop(self, key: str, block: bool = ...) -> bytes:
        """
        queue_pop(self: torch._C._distributed_c10d.Store, key: str, block: bool = True) -> bytes


        Pops a value from the specified queue or waits until timeout if the queue is empty.

        See queue_push for more details.

        If block is False, a dist.QueueEmptyError will be raised if the queue is empty.

        Arguments:
            key (str): The key of the queue to pop from.
            block (bool): Whether to block waiting for the key or immediately return.
        """
    def queue_push(self, key: str, value: bytes | str) -> None:
        """
        queue_push(self: torch._C._distributed_c10d.Store, arg0: str, arg1: str) -> None


        Pushes a value into the specified queue.

        Using the same key for queues and set/get operations may result in unexpected
        behavior.

        wait/check operations are supported for queues.

        wait with queues will only wake one waiting worker rather than all.

        Arguments:
            key (str): The key of the queue to push to.
            value (str): The value to push into the queue.
        """
    def queue_len(self, key: str) -> int:
        """
        queue_len(self: torch._C._distributed_c10d.Store, arg0: str) -> int


        Returns the length of the specified queue.

        If the queue doesn't exist it returns 0.

        See queue_push for more details.

        Arguments:
            key (str): The key of the queue to get the length.
        """

class FileStore(Store):
    """
    A store implementation that uses a file to store the underlying key-value pairs.

    Arguments:
        file_name (str): path of the file in which to store the key-value pairs
        world_size (int, optional): The total number of processes using the store. Default is -1 (a negative value indicates a non-fixed number of store users).

    Example::
        >>> import torch.distributed as dist
        >>> store1 = dist.FileStore("/tmp/filestore", 2)
        >>> store2 = dist.FileStore("/tmp/filestore", 2)
        >>> # Use any of the store methods from either the client or server after initialization
        >>> store1.set("first_key", "first_value")
        >>> store2.get("first_key")


    """
    def __init__(self, path: str, numWorkers: int = ...) -> None:
        """
        __init__(self: torch._C._distributed_c10d.FileStore, file_name: str, world_size: typing.SupportsInt = -1) -> None

        Creates a new FileStore.
        """

class HashStore(Store):
    """
    A thread-safe store implementation based on an underlying hashmap. This store can be used
    within the same process (for example, by other threads), but cannot be used across processes.

    Example::
        >>> import torch.distributed as dist
        >>> store = dist.HashStore()
        >>> # store can be used from other threads
        >>> # Use any of the store methods after initialization
        >>> store.set("first_key", "first_value")

    """
    def __init__(self) -> None:
        """
        __init__(self: torch._C._distributed_c10d.HashStore) -> None

        Creates a new HashStore.
        """

class TCPStore(Store):
    """
    A TCP-based distributed key-value store implementation. The server store holds
    the data, while the client stores can connect to the server store over TCP and
    perform actions such as :meth:`~torch.distributed.store.set` to insert a key-value
    pair, :meth:`~torch.distributed.store.get` to retrieve a key-value pair, etc. There
    should always be one server store initialized because the client store(s) will wait for
    the server to establish a connection.

    Arguments:
        host_name (str): The hostname or IP Address the server store should run on.
        port (int): The port on which the server store should listen for incoming requests.
        world_size (int, optional): The total number of store users (number of clients + 1 for the server). Default is None (None indicates a non-fixed number of store users).
        is_master (bool, optional): True when initializing the server store and False for client stores. Default is False.
        timeout (timedelta, optional): Timeout used by the store during initialization and for methods such as :meth:`~torch.distributed.store.get` and :meth:`~torch.distributed.store.wait`. Default is timedelta(seconds=300)
        wait_for_workers (bool, optional): Whether to wait for all the workers to connect with the server store. This is only applicable when world_size is a fixed value. Default is True.
        multi_tenant (bool, optional): If True, all ``TCPStore`` instances in the current process with the same host/port will use the same underlying ``TCPServer``. Default is False.
        master_listen_fd (int, optional): If specified, the underlying ``TCPServer`` will listen on this file descriptor, which must be a socket already bound to ``port``. To bind an ephemeral port we recommend setting the port to 0 and reading ``.port``. Default is None (meaning the server creates a new socket and attempts to bind it to ``port``).
        use_libuv (bool, optional): If True, use libuv for ``TCPServer`` backend. Default is True.
    Example::
        >>> import torch.distributed as dist
        >>> from datetime import timedelta
        >>> # Run on process 1 (server)
        >>> server_store = dist.TCPStore("127.0.0.1", 1234, 2, True, timedelta(seconds=30))
        >>> # Run on process 2 (client)
        >>> client_store = dist.TCPStore("127.0.0.1", 1234, 2, False)
        >>> # Use any of the store methods from either the client or server after initialization
        >>> server_store.set("first_key", "first_value")
        >>> client_store.get("first_key")

    """
    def __init__(
        self,
        host_name: str,
        port: int,
        world_size: int | None = ...,
        is_master: bool = ...,
        timeout: timedelta = ...,
        wait_for_workers: bool = ...,
        multi_tenant: bool = ...,
        master_listen_fd: int | None = ...,
        use_libuv: bool | None = ...,
    ) -> None:
        """
        __init__(self: torch._C._distributed_c10d.TCPStore, host_name: str, port: typing.SupportsInt, world_size: typing.SupportsInt | None = None, is_master: bool = False, timeout: datetime.timedelta = datetime.timedelta(seconds=300), wait_for_workers: bool = True, multi_tenant: bool = False, master_listen_fd: typing.SupportsInt | None = None, use_libuv: bool = True) -> None

        Creates a new TCPStore.
        """
    @property
    def host(self) -> str:
        """Gets the hostname on which the store listens for requests."""
    @property
    def port(self) -> int:
        """Gets the port number on which the store listens for requests."""

class PrefixStore(Store):
    """
    A wrapper around any of the 3 key-value stores (:class:`~torch.distributed.TCPStore`,
    :class:`~torch.distributed.FileStore`, and :class:`~torch.distributed.HashStore`)
    that adds a prefix to each key inserted to the store.

    Arguments:
        prefix (str): The prefix string that is prepended to each key before being inserted into the store.
        store (torch.distributed.store): A store object that forms the underlying key-value store.

    """
    def __init__(self, prefix: str, store: Store) -> None:
        """
        __init__(self: torch._C._distributed_c10d.PrefixStore, prefix: str, store: torch._C._distributed_c10d.Store) -> None

        Creates a new PrefixStore.
        """
    @property
    def underlying_store(self) -> Store:
        """Gets the underlying store object that PrefixStore wraps around."""

class _ControlCollectives:
    """Base class for all ControlCollectives implementations."""
    def barrier(self, key: str, timeout: timedelta, blocking: bool) -> None:
        """
        barrier(self: torch._C._distributed_c10d._ControlCollectives, key: str, timeout: datetime.timedelta = datetime.timedelta(seconds=300), block: bool = True) -> None


        Blocks until all workers have entered this function.

        Arguments:
            key (str): The unique key used to identify this operation.
            timeout (duration): The timeout for this operation.
            block (bool): whether to block this working waiting on the results of the barrier.
        """
    def broadcast_send(self, key: str, data: str, timeout: timedelta) -> None:
        """
        broadcast_send(self: torch._C._distributed_c10d._ControlCollectives, key: str, data: str, timeout: datetime.timedelta = datetime.timedelta(seconds=300)) -> None


        Sends data to all other workers. Must be only called from one worker.

        Arguments:
            key (str): The unique key used to identify this operation.
            data (str): The data to send.
            timeout (duration): The timeout for this operation.
        """
    def broadcast_recv(self, key: str, timeout: timedelta) -> str:
        """
        broadcast_recv(self: torch._C._distributed_c10d._ControlCollectives, key: str, timeout: datetime.timedelta = datetime.timedelta(seconds=300)) -> bytes


        Receives data broadcasted from 1 worker.

        Arguments:
            key (str): The unique key used to identify this operation.
            timeout (duration): The timeout for this operation.
        """
    def gather_send(self, key: str, data: str, timeout: timedelta) -> None:
        """
        gather_send(self: torch._C._distributed_c10d._ControlCollectives, key: str, data: str, timeout: datetime.timedelta = datetime.timedelta(seconds=300)) -> None


        Sends data to one other worker.

        Arguments:
            key (str): The unique key used to identify this operation.
            data (str): The data to send.
            timeout (duration): The timeout for this operation.
        """
    def gather_recv(self, key: str, timeout: timedelta) -> str:
        """
        gather_recv(self: torch._C._distributed_c10d._ControlCollectives, key: str, data: str, timeout: datetime.timedelta = datetime.timedelta(seconds=300)) -> list[bytes]


        Receives data broadcasted from all workers. Must only be called by one worker.

        Arguments:
            key (str): The unique key used to identify this operation.
            timeout (duration): The timeout for this operation.
        """
    def scatter_send(self, key: str, data: str, timeout: timedelta) -> None:
        """
        scatter_send(self: torch._C._distributed_c10d._ControlCollectives, key: str, data: collections.abc.Sequence[str], timeout: datetime.timedelta = datetime.timedelta(seconds=300)) -> bytes


        Sends rank specific data to all other workers.

        Arguments:
            key (str): The unique key used to identify this operation.
            data (str): The data to send.
            timeout (duration): The timeout for this operation.
        """
    def scatter_recv(self, key: str, timeout: timedelta) -> str:
        """
        scatter_recv(self: torch._C._distributed_c10d._ControlCollectives, key: str, timeout: datetime.timedelta = datetime.timedelta(seconds=300)) -> bytes


        Receives rank specific data from one worker.

        Arguments:
            key (str): The unique key used to identify this operation.
            timeout (duration): The timeout for this operation.
        """
    def all_gather(self, key: str, data: str, timeout: timedelta) -> str:
        """
        all_gather(self: torch._C._distributed_c10d._ControlCollectives, key: str, data: str, timeout: datetime.timedelta = datetime.timedelta(seconds=300)) -> list[bytes]


        Sends data to all workers and receives data from all other workers.

        Arguments:
            key (str): The unique key used to identify this operation.
            data (str): The data to send.
            timeout (duration): The timeout for this operation.
        """
    def all_sum(self, key: str, data: int, timeout: timedelta) -> int:
        """
        all_sum(self: torch._C._distributed_c10d._ControlCollectives, key: str, data: typing.SupportsInt, timeout: datetime.timedelta = datetime.timedelta(seconds=300)) -> int


        Computes a sum across all workers and returns the final value.

        Arguments:
            key (str): The unique key used to identify this operation.
            data (int): The data to sum.
            timeout (duration): The timeout for this operation.
        """

class _StoreCollectives(_ControlCollectives):
    """
    An implementation of ControlCollectives that uses the provided store as the underlying
    communication mechanism.

    """
    def __init__(self, store: Store, rank: int, world_size: int) -> None:
        """__init__(self: torch._C._distributed_c10d._StoreCollectives, store: torch._C._distributed_c10d.Store, rank: typing.SupportsInt, world_size: typing.SupportsInt) -> None"""

class _DistributedBackendOptions:
    def __init__(self) -> None:
        """__init__(self: torch._C._distributed_c10d._DistributedBackendOptions) -> None"""
    @property
    def store(self) -> Store: ...
    @store.setter
    def store(self, store: Store) -> None: ...
    @property
    def group_rank(self) -> int: ...
    @group_rank.setter
    def group_rank(self, rank: int) -> None: ...
    @property
    def group_size(self) -> int: ...
    @group_size.setter
    def group_size(self, size: int) -> None: ...
    @property
    def timeout(self) -> timedelta: ...
    @timeout.setter
    def timeout(self, timeout: timedelta) -> None: ...
    @property
    def group_id(self) -> str: ...
    @group_id.setter
    def group_id(self, group_id: str) -> None: ...
    @property
    def global_ranks_in_group(self) -> list[int]: ...
    @global_ranks_in_group.setter
    def global_ranks_in_group(self, ranks: list[int]) -> None: ...

class Work:
    """
    A `Work` object represents the handle to a pending asynchronous operation in
    PyTorch's distributed package. It is returned by non-blocking collective operations,
    such as `dist.all_reduce(tensor, async_op=True)`.
    """
    def is_completed(self) -> bool:
        """is_completed(self: torch._C._distributed_c10d.Work) -> bool"""
    def is_success(self) -> bool:
        """is_success(self: torch._C._distributed_c10d.Work) -> bool"""
    def exception(self) -> Any:
        """exception(self: torch._C._distributed_c10d.Work) -> std::exception_ptr"""
    def wait(self, timeout: timedelta = ...) -> bool:
        """
        wait(self: torch._C._distributed_c10d.Work, timeout: datetime.timedelta = datetime.timedelta(0)) -> bool


        Returns:
            true/false.

        Example::
           try:
               work.wait(timeout)
           except:
               # some handling

        .. warning ::
            In normal cases, users do not need to set the timeout.
            calling wait() is the same as calling synchronize():
            Letting the current stream block on the completion of the NCCL work.
            However, if timeout is set, it will block the CPU thread until the NCCL work is completed
            or timed out. If timeout, exception will be thrown.
        """
    def block_current_stream(self) -> None:
        """
        block_current_stream(self: torch._C._distributed_c10d.Work) -> None


        Blocks the currently active GPU stream on the operation to
        complete. For GPU based collectives this is equivalent to
        synchronize. For CPU initiated collectives such as with Gloo this
        will block the CUDA stream until the operation is complete.

        This returns immediately in all cases.

        To check whether an operation was successful you should check the
        Work object result asynchronously.
        """
    def get_future(self) -> Future:
        """
        get_future(self: torch._C._distributed_c10d.Work) -> torch.Future


        Returns:
            A ``torch.futures.Future`` object which is associated with the completion of
            the ``Work``. As an example, a future object can be retrieved
            by ``fut = process_group.allreduce(tensors).get_future()``.

        Example::
            Below is an example of a simple allreduce DDP communication hook that uses
            ``get_future`` API to retrieve a Future associated with the completion of
            ``allreduce``.

            >>> def allreduce(process_group: dist.ProcessGroup, bucket: dist.GradBucket): -> torch.futures.Future
            >>>     group_to_use = process_group if process_group is not None else torch.distributed.group.WORLD
            >>>     tensor = bucket.buffer().div_(group_to_use.size())
            >>>     return torch.distributed.all_reduce(tensor, group=group_to_use, async_op=True).get_future()
            >>> ddp_model.register_comm_hook(state=None, hook=allreduce)

        .. warning ::
            ``get_future`` API supports NCCL, and partially GLOO and MPI backends
            (no support for peer-to-peer operations like send/recv) and will return a ``torch.futures.Future``.

            In the example above, ``allreduce`` work will be done on GPU using NCCL backend,
            ``fut.wait()`` will return after synchronizing the appropriate NCCL streams
            with PyTorch's current device streams to ensure we can have asynchronous CUDA
            execution and it does not wait for the entire operation to complete on GPU. Note that
            ``CUDAFuture``  does not support ``TORCH_NCCL_BLOCKING_WAIT`` flag or NCCL's ``barrier()``.
            In addition, if a callback function was added by ``fut.then()``, it will wait until
            ``WorkNCCL``'s NCCL streams synchronize with ``ProcessGroupNCCL``'s dedicated callback
            stream and invoke the callback inline after running the callback on the callback stream.
            ``fut.then()`` will return another ``CUDAFuture`` that holds the return value of the
            callback and a ``CUDAEvent`` that recorded the callback stream.

                1. For CPU work, ``fut.done()`` returns true when work has been completed and value()
                   tensors are ready.
                2. For GPU work, ``fut.done()`` returns true only whether the operation has been enqueued.
                3. For mixed CPU-GPU work (e.g. sending GPU tensors with GLOO), ``fut.done()`` returns
                   true when tensors have arrived on respective nodes, but not yet necessarily synched on
                   respective GPUs (similarly to GPU work).
        """
    def source_rank(self) -> int:
        """source_rank(self: torch._C._distributed_c10d.Work) -> int"""
    def result(self) -> list[Tensor]:
        """result(self: torch._C._distributed_c10d.Work) -> list[torch.Tensor]"""
    def synchronize(self) -> None:
        """synchronize(self: torch._C._distributed_c10d.Work) -> None"""
    def boxed(self) -> ScriptObject:
        """boxed(self: torch._C._distributed_c10d.Work) -> object"""
    @staticmethod
    def unbox(obj: ScriptObject) -> Work:
        """unbox(arg0: object) -> torch._C._distributed_c10d.Work"""

class Backend:
    class Options:
        """
        Base class for all backend options implementations, such as the nccl
        options :class:`~torch.distributed.ProcessGroupNCCL.Options`).
        """
        def __init__(self, backend: str, timeout: timedelta = ...) -> None:
            """__init__(self: torch._C._distributed_c10d.Backend.Options, backend: str, timeout: datetime.timedelta = datetime.timedelta(seconds=1800)) -> None"""
        @property
        def backend(self) -> str: ...

        global_ranks_in_group: list[int]
        group_name: str

    def __init__(self, rank: int, size: int) -> None: ...
    @property
    def supports_splitting(self) -> bool:
        """(test whether the backend supports splitting)"""
    @property
    def supports_coalescing(self) -> bool:
        """(test whether the backend supports coalescing)"""
    @property
    def supports_time_estimate(self) -> bool:
        """(test whether the backend supports collective time estimation)"""
    def set_timeout(self, timeout: timedelta) -> None:
        """
        set_timeout(self: torch._C._distributed_c10d.Backend, timeout: datetime.timedelta) -> None

        Sets the default timeout for all future operations.
        """
    @property
    def options(self) -> Options: ...
    def rank(self) -> int:
        """rank(self: torch._C._distributed_c10d.Backend) -> int"""
    def size(self) -> int:
        """size(self: torch._C._distributed_c10d.Backend) -> int"""
    def name(self) -> str:
        """name(self: torch._C._distributed_c10d.Backend) -> str"""
    def abort(self) -> None:
        """
        abort(self: torch._C._distributed_c10d.Backend) -> None

        abort all operations and connections if supported by the backend
        """
    def shutdown(self) -> None:
        """
        shutdown(self: torch._C._distributed_c10d.Backend) -> None

        shutdown the backend
        """
    def eager_connect_single_device(self, device: torch.device | None) -> None:
        """eager_connect_single_device(self: torch._C._distributed_c10d.Backend, arg0: torch.device) -> None"""
    def get_error(self) -> ErrorType: ...
    def supports_tensor_alloc(self, device: torch.device) -> bool:
        """supports_tensor_alloc(self: torch._C._distributed_c10d.Backend, device: torch.device) -> bool"""
    def allocate_tensor(self, size: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        """allocate_tensor(self: torch._C._distributed_c10d.Backend, size: typing.SupportsInt, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor"""
    @property
    def mem_allocator(self) -> Any: ...

class ProcessGroup:
    """
    A ProcessGroup is a communication primitive that allows for
    collective operations across a group of processes.

    This is a base class that provides the interface for all
    ProcessGroups. It is not meant to be used directly, but rather
    extended by subclasses.
    """
    class BackendType(Enum):
        """
        The type of the backend used for the process group.

        Members:

          UNDEFINED

          GLOO

          NCCL

          XCCL

          UCC

          MPI

          CUSTOM
        """

        UNDEFINED = ...
        GLOO = ...
        NCCL = ...
        UCC = ...
        MPI = ...
        XCCL = ...
        CUSTOM = ...

    def __init__(self, store: Store, rank: int, size: int) -> None:
        """
        __init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: torch._C._distributed_c10d.ProcessGroup, rank: typing.SupportsInt, size: typing.SupportsInt) -> None

        Create a new ProcessGroup instance.

        2. __init__(self: torch._C._distributed_c10d.ProcessGroup, store: torch._C._distributed_c10d.Store, rank: typing.SupportsInt, size: typing.SupportsInt) -> None

        Create a new ProcessGroup instance.
        """
    def rank(self) -> int:
        """
        rank(self: torch._C._distributed_c10d.ProcessGroup) -> int

        Get the rank of this process group.
        """
    def size(self) -> int:
        """
        size(self: torch._C._distributed_c10d.ProcessGroup) -> int

        Get the size of this process group.
        """
    def get_group_store(self) -> Store:
        """
        get_group_store(self: torch._C._distributed_c10d.ProcessGroup) -> torch._C._distributed_c10d.Store

        Get the store of this process group.
        """
    def split_group(
        self,
        new_ranks: list[int],
        timeout: timedelta | None = ...,
        opts: Backend.Options | None = ...,
        group_name: str | None = ...,
        group_desc: str | None = ...,
    ) -> ProcessGroup | None:
        """split_group(self: torch._C._distributed_c10d.ProcessGroup, ranks: collections.abc.Sequence[typing.SupportsInt], timeout: datetime.timedelta | None = None, opts: c10d::Backend::Options | None = None, group_name: str | None = None, group_desc: str | None = None) -> torch._C._distributed_c10d.ProcessGroup"""
    def merge_remote_group(
        self, store: Store, size: int, timeout: timedelta, group_name: str | None = ..., group_desc: str | None = ...
    ) -> ProcessGroup:
        """merge_remote_group(self: torch._C._distributed_c10d.ProcessGroup, store: torch._C._distributed_c10d.Store, size: typing.SupportsInt, timeout: datetime.timedelta = datetime.timedelta(seconds=1800), group_name: str | None = None, group_desc: str | None = None) -> torch._C._distributed_c10d.ProcessGroup"""
    def abort(self) -> None:
        """
        abort(self: torch._C._distributed_c10d.ProcessGroup) -> None

        abort all operations and connections if supported by the backend
        """
    def set_timeout(self, timeout: timedelta) -> None:
        """
        set_timeout(self: torch._C._distributed_c10d.ProcessGroup, timeout: datetime.timedelta) -> None

        Sets the default timeout for all future operations.
        """
    def shutdown(self) -> None:
        """
        shutdown(self: torch._C._distributed_c10d.ProcessGroup) -> None

        shutdown the process group
        """
    @overload
    def broadcast(self, tensors: list[Tensor], opts=...) -> Work:
        """
        broadcast(*args, **kwargs)
        Overloaded function.

        1. broadcast(self: torch._C._distributed_c10d.ProcessGroup, tensors: collections.abc.Sequence[torch.Tensor], opts: torch._C._distributed_c10d.BroadcastOptions = <torch._C._distributed_c10d.BroadcastOptions object at 0x113d36f30>) -> c10d::Work

        Broadcasts the tensor to all processes in the process group.

                      See :func:`torch.distributed.broadcast` for more details.

        2. broadcast(self: torch._C._distributed_c10d.ProcessGroup, tensor: torch.Tensor, root: typing.SupportsInt, timeout: datetime.timedelta | None = None) -> c10d::Work

        Broadcasts the tensor to all processes in the process group.

                      See :func:`torch.distributed.broadcast` for more details.
        """
    @overload
    def broadcast(self, tensor: Tensor, root: int, timeout: timedelta | None = ...) -> Work:
        """
        broadcast(*args, **kwargs)
        Overloaded function.

        1. broadcast(self: torch._C._distributed_c10d.ProcessGroup, tensors: collections.abc.Sequence[torch.Tensor], opts: torch._C._distributed_c10d.BroadcastOptions = <torch._C._distributed_c10d.BroadcastOptions object at 0x113d36f30>) -> c10d::Work

        Broadcasts the tensor to all processes in the process group.

                      See :func:`torch.distributed.broadcast` for more details.

        2. broadcast(self: torch._C._distributed_c10d.ProcessGroup, tensor: torch.Tensor, root: typing.SupportsInt, timeout: datetime.timedelta | None = None) -> c10d::Work

        Broadcasts the tensor to all processes in the process group.

                      See :func:`torch.distributed.broadcast` for more details.
        """
    @overload
    def allreduce(self, tensors: list[Tensor], opts: AllreduceOptions = ...) -> Work:
        """
        allreduce(*args, **kwargs)
        Overloaded function.

        1. allreduce(self: torch._C._distributed_c10d.ProcessGroup, tensors: collections.abc.Sequence[torch.Tensor], opts: torch._C._distributed_c10d.AllreduceOptions = <torch._C._distributed_c10d.AllreduceOptions object at 0x113d36fb0>) -> c10d::Work

        Allreduces the provided tensors across all processes in the process group.

                      See :func:`torch.distributed.all_reduce` for more details.

        2. allreduce(self: torch._C._distributed_c10d.ProcessGroup, tensors: collections.abc.Sequence[torch.Tensor], op: torch._C._distributed_c10d.ReduceOp = <RedOpType.SUM: 0>, timeout: datetime.timedelta | None = None) -> c10d::Work

        Allreduces the provided tensors across all processes in the process group.

                      See :func:`torch.distributed.all_reduce` for more details.

        3. allreduce(self: torch._C._distributed_c10d.ProcessGroup, tensor: torch.Tensor, op: torch._C._distributed_c10d.ReduceOp = <RedOpType.SUM: 0>, timeout: datetime.timedelta | None = None) -> c10d::Work

        Allreduces the provided tensors across all processes in the process group.

                      See :func:`torch.distributed.all_reduce` for more details.
        """
    @overload
    def allreduce(self, tensors: list[Tensor], op=..., timeout: timedelta | None = ...) -> Work:
        """
        allreduce(*args, **kwargs)
        Overloaded function.

        1. allreduce(self: torch._C._distributed_c10d.ProcessGroup, tensors: collections.abc.Sequence[torch.Tensor], opts: torch._C._distributed_c10d.AllreduceOptions = <torch._C._distributed_c10d.AllreduceOptions object at 0x113d36fb0>) -> c10d::Work

        Allreduces the provided tensors across all processes in the process group.

                      See :func:`torch.distributed.all_reduce` for more details.

        2. allreduce(self: torch._C._distributed_c10d.ProcessGroup, tensors: collections.abc.Sequence[torch.Tensor], op: torch._C._distributed_c10d.ReduceOp = <RedOpType.SUM: 0>, timeout: datetime.timedelta | None = None) -> c10d::Work

        Allreduces the provided tensors across all processes in the process group.

                      See :func:`torch.distributed.all_reduce` for more details.

        3. allreduce(self: torch._C._distributed_c10d.ProcessGroup, tensor: torch.Tensor, op: torch._C._distributed_c10d.ReduceOp = <RedOpType.SUM: 0>, timeout: datetime.timedelta | None = None) -> c10d::Work

        Allreduces the provided tensors across all processes in the process group.

                      See :func:`torch.distributed.all_reduce` for more details.
        """
    @overload
    def allreduce(self, tensor: Tensor, op=..., timeout: timedelta | None = ...) -> Work:
        """
        allreduce(*args, **kwargs)
        Overloaded function.

        1. allreduce(self: torch._C._distributed_c10d.ProcessGroup, tensors: collections.abc.Sequence[torch.Tensor], opts: torch._C._distributed_c10d.AllreduceOptions = <torch._C._distributed_c10d.AllreduceOptions object at 0x113d36fb0>) -> c10d::Work

        Allreduces the provided tensors across all processes in the process group.

                      See :func:`torch.distributed.all_reduce` for more details.

        2. allreduce(self: torch._C._distributed_c10d.ProcessGroup, tensors: collections.abc.Sequence[torch.Tensor], op: torch._C._distributed_c10d.ReduceOp = <RedOpType.SUM: 0>, timeout: datetime.timedelta | None = None) -> c10d::Work

        Allreduces the provided tensors across all processes in the process group.

                      See :func:`torch.distributed.all_reduce` for more details.

        3. allreduce(self: torch._C._distributed_c10d.ProcessGroup, tensor: torch.Tensor, op: torch._C._distributed_c10d.ReduceOp = <RedOpType.SUM: 0>, timeout: datetime.timedelta | None = None) -> c10d::Work

        Allreduces the provided tensors across all processes in the process group.

                      See :func:`torch.distributed.all_reduce` for more details.
        """
    def allreduce_coalesced(self, tensors: list[Tensor], opts=...) -> Work:
        """
        allreduce_coalesced(self: torch._C._distributed_c10d.ProcessGroup, tensors: collections.abc.Sequence[torch.Tensor], opts: torch._C._distributed_c10d.AllreduceCoalescedOptions = <torch._C._distributed_c10d.AllreduceCoalescedOptions object at 0x111385130>) -> c10d::Work

        Allreduces the provided tensors across all processes in the process group.

                      See :func:`torch.distributed.all_reduce` for more details.
        """
    def reduce_scatter_tensor_coalesced(
        self, outputTensors: list[Tensor], inputTensors: list[Tensor], opts: ReduceScatterOptions | None = ...
    ) -> Work:
        """
        reduce_scatter_tensor_coalesced(self: torch._C._distributed_c10d.ProcessGroup, outputs: collections.abc.Sequence[torch.Tensor], inputs: collections.abc.Sequence[torch.Tensor], opts: torch._C._distributed_c10d.ReduceScatterOptions = <torch._C._distributed_c10d.ReduceScatterOptions object at 0x1112d7d70>) -> c10d::Work

        Reduces and scatters the input tensors from all processes across the process group.

                      See :func:`torch.distributed.reduce_scatter` for more details.
        """
    @overload
    def reduce(self, tensors: list[Tensor], opts=...) -> Work:
        """
        reduce(*args, **kwargs)
        Overloaded function.

        1. reduce(self: torch._C._distributed_c10d.ProcessGroup, tensors: collections.abc.Sequence[torch.Tensor], opts: torch._C._distributed_c10d.ReduceOptions = <torch._C._distributed_c10d.ReduceOptions object at 0x113d37330>) -> c10d::Work

        Reduces the provided tensors across all processes in the process group.

                      See :func:`torch.distributed.reduce` for more details.

        2. reduce(self: torch._C._distributed_c10d.ProcessGroup, tensor: torch.Tensor, root: typing.SupportsInt, op: torch._C._distributed_c10d.ReduceOp = <RedOpType.SUM: 0>, timeout: datetime.timedelta | None = None) -> c10d::Work

        Reduces the provided tensors across all processes in the process group.

                      See :func:`torch.distributed.reduce` for more details.
        """
    @overload
    def reduce(self, tensor: Tensor, root: int, op=..., timeout: timedelta | None = ...) -> Work:
        """
        reduce(*args, **kwargs)
        Overloaded function.

        1. reduce(self: torch._C._distributed_c10d.ProcessGroup, tensors: collections.abc.Sequence[torch.Tensor], opts: torch._C._distributed_c10d.ReduceOptions = <torch._C._distributed_c10d.ReduceOptions object at 0x113d37330>) -> c10d::Work

        Reduces the provided tensors across all processes in the process group.

                      See :func:`torch.distributed.reduce` for more details.

        2. reduce(self: torch._C._distributed_c10d.ProcessGroup, tensor: torch.Tensor, root: typing.SupportsInt, op: torch._C._distributed_c10d.ReduceOp = <RedOpType.SUM: 0>, timeout: datetime.timedelta | None = None) -> c10d::Work

        Reduces the provided tensors across all processes in the process group.

                      See :func:`torch.distributed.reduce` for more details.
        """
    @overload
    def allgather(self, output_tensors: list[list[Tensor]], input_tensors: list[Tensor], opts=...) -> Work:
        """
        allgather(*args, **kwargs)
        Overloaded function.

        1. allgather(self: torch._C._distributed_c10d.ProcessGroup, output_tensors: collections.abc.Sequence[collections.abc.Sequence[torch.Tensor]], input_tensors: collections.abc.Sequence[torch.Tensor], opts: torch._C._distributed_c10d.AllgatherOptions = <torch._C._distributed_c10d.AllgatherOptions object at 0x113d37430>) -> c10d::Work

        Allgathers the input tensors from all processes across the process group.

                      See :func:`torch.distributed.all_gather` for more details.

        2. allgather(self: torch._C._distributed_c10d.ProcessGroup, output_tensors: collections.abc.Sequence[torch.Tensor], input_tensor: torch.Tensor, timeout: datetime.timedelta | None = None) -> c10d::Work

        Allgathers the input tensors from all processes across the process group.

                      See :func:`torch.distributed.all_gather` for more details.
        """
    @overload
    def allgather(self, output_tensors: list[Tensor], input_tensor: Tensor, timeout: timedelta | None = ...) -> Work:
        """
        allgather(*args, **kwargs)
        Overloaded function.

        1. allgather(self: torch._C._distributed_c10d.ProcessGroup, output_tensors: collections.abc.Sequence[collections.abc.Sequence[torch.Tensor]], input_tensors: collections.abc.Sequence[torch.Tensor], opts: torch._C._distributed_c10d.AllgatherOptions = <torch._C._distributed_c10d.AllgatherOptions object at 0x113d37430>) -> c10d::Work

        Allgathers the input tensors from all processes across the process group.

                      See :func:`torch.distributed.all_gather` for more details.

        2. allgather(self: torch._C._distributed_c10d.ProcessGroup, output_tensors: collections.abc.Sequence[torch.Tensor], input_tensor: torch.Tensor, timeout: datetime.timedelta | None = None) -> c10d::Work

        Allgathers the input tensors from all processes across the process group.

                      See :func:`torch.distributed.all_gather` for more details.
        """
    def allgather_coalesced(self, output_lists: list[list[Tensor]], input_list: list[Tensor], opts=...) -> Work:
        """
        allgather_coalesced(self: torch._C._distributed_c10d.ProcessGroup, output_lists: collections.abc.Sequence[collections.abc.Sequence[torch.Tensor]], input_list: collections.abc.Sequence[torch.Tensor], opts: torch._C._distributed_c10d.AllgatherOptions = <torch._C._distributed_c10d.AllgatherOptions object at 0x11129f6f0>) -> c10d::Work

        Allgathers the input tensors from all processes across the process group.

                      See :func:`torch.distributed.all_gather` for more details.
        """
    def allgather_into_tensor_coalesced(self, output_lists: list[Tensor], input_list: list[Tensor], opts=...) -> Work:
        """
        allgather_into_tensor_coalesced(self: torch._C._distributed_c10d.ProcessGroup, outputs: collections.abc.Sequence[torch.Tensor], inputs: collections.abc.Sequence[torch.Tensor], opts: torch._C._distributed_c10d.AllgatherOptions = <torch._C._distributed_c10d.AllgatherOptions object at 0x11137d070>) -> c10d::Work

        Allgathers the input tensors from all processes across the process group.

                      See :func:`torch.distributed.all_gather` for more details.
        """
    @overload
    def gather(self, output_tensors: list[list[Tensor]], input_tensors: list[Tensor], opts=...) -> Work:
        """
        gather(*args, **kwargs)
        Overloaded function.

        1. gather(self: torch._C._distributed_c10d.ProcessGroup, output_tensors: collections.abc.Sequence[collections.abc.Sequence[torch.Tensor]], input_tensors: collections.abc.Sequence[torch.Tensor], opts: torch._C._distributed_c10d.GatherOptions = <torch._C._distributed_c10d.GatherOptions object at 0x113d378b0>) -> c10d::Work

        Gathers the input tensors from all processes across the process group.

                      See :func:`torch.distributed.gather` for more details.

        2. gather(self: torch._C._distributed_c10d.ProcessGroup, output_tensors: collections.abc.Sequence[torch.Tensor], input_tensor: torch.Tensor, root: typing.SupportsInt, timeout: datetime.timedelta | None = None) -> c10d::Work

        Gathers the input tensors from all processes across the process group.

                      See :func:`torch.distributed.gather` for more details.
        """
    @overload
    def gather(
        self, output_tensors: list[Tensor], input_tensor: Tensor, root: int, timeout: timedelta | None = ...
    ) -> Work:
        """
        gather(*args, **kwargs)
        Overloaded function.

        1. gather(self: torch._C._distributed_c10d.ProcessGroup, output_tensors: collections.abc.Sequence[collections.abc.Sequence[torch.Tensor]], input_tensors: collections.abc.Sequence[torch.Tensor], opts: torch._C._distributed_c10d.GatherOptions = <torch._C._distributed_c10d.GatherOptions object at 0x113d378b0>) -> c10d::Work

        Gathers the input tensors from all processes across the process group.

                      See :func:`torch.distributed.gather` for more details.

        2. gather(self: torch._C._distributed_c10d.ProcessGroup, output_tensors: collections.abc.Sequence[torch.Tensor], input_tensor: torch.Tensor, root: typing.SupportsInt, timeout: datetime.timedelta | None = None) -> c10d::Work

        Gathers the input tensors from all processes across the process group.

                      See :func:`torch.distributed.gather` for more details.
        """
    @overload
    def scatter(self, output_tensors: list[Tensor], input_tensors: list[list[Tensor]], opts=...) -> Work:
        """
        scatter(*args, **kwargs)
        Overloaded function.

        1. scatter(self: torch._C._distributed_c10d.ProcessGroup, output_tensors: collections.abc.Sequence[torch.Tensor], input_tensors: collections.abc.Sequence[collections.abc.Sequence[torch.Tensor]], opts: torch._C._distributed_c10d.ScatterOptions = <torch._C._distributed_c10d.ScatterOptions object at 0x113d378f0>) -> c10d::Work

        Scatters the input tensors from all processes across the process group.

                      See :func:`torch.distributed.scatter` for more details.

        2. scatter(self: torch._C._distributed_c10d.ProcessGroup, output_tensor: torch.Tensor, input_tensors: collections.abc.Sequence[torch.Tensor], root: typing.SupportsInt, timeout: datetime.timedelta | None = None) -> c10d::Work

        Scatters the input tensors from all processes across the process group.

                      See :func:`torch.distributed.scatter` for more details.
        """
    @overload
    def scatter(
        self, output_tensor: Tensor, input_tensors: list[Tensor], root: int, timeout: timedelta | None = ...
    ) -> Work:
        """
        scatter(*args, **kwargs)
        Overloaded function.

        1. scatter(self: torch._C._distributed_c10d.ProcessGroup, output_tensors: collections.abc.Sequence[torch.Tensor], input_tensors: collections.abc.Sequence[collections.abc.Sequence[torch.Tensor]], opts: torch._C._distributed_c10d.ScatterOptions = <torch._C._distributed_c10d.ScatterOptions object at 0x113d378f0>) -> c10d::Work

        Scatters the input tensors from all processes across the process group.

                      See :func:`torch.distributed.scatter` for more details.

        2. scatter(self: torch._C._distributed_c10d.ProcessGroup, output_tensor: torch.Tensor, input_tensors: collections.abc.Sequence[torch.Tensor], root: typing.SupportsInt, timeout: datetime.timedelta | None = None) -> c10d::Work

        Scatters the input tensors from all processes across the process group.

                      See :func:`torch.distributed.scatter` for more details.
        """
    @overload
    def reduce_scatter(self, output_tensors: list[Tensor], input_tensors: list[list[Tensor]], opts=...) -> Work:
        """
        reduce_scatter(*args, **kwargs)
        Overloaded function.

        1. reduce_scatter(self: torch._C._distributed_c10d.ProcessGroup, output_tensors: collections.abc.Sequence[torch.Tensor], input_tensors: collections.abc.Sequence[collections.abc.Sequence[torch.Tensor]], opts: torch._C._distributed_c10d.ReduceScatterOptions = <torch._C._distributed_c10d.ReduceScatterOptions object at 0x113d37a70>) -> c10d::Work

        Reduces and scatters the input tensors from all processes across the process group.

                      See :func:`torch.distributed.reduce_scatter` for more details.

        2. reduce_scatter(self: torch._C._distributed_c10d.ProcessGroup, output: torch.Tensor, input: collections.abc.Sequence[torch.Tensor], op: torch._C._distributed_c10d.ReduceOp = <RedOpType.SUM: 0>, timeout: datetime.timedelta | None = None) -> c10d::Work

        Reduces and scatters the input tensors from all processes across the process group.

                      See :func:`torch.distributed.reduce_scatter` for more details.
        """
    @overload
    def reduce_scatter(
        self, output_tensors: Tensor, input_tensor: list[Tensor], op=..., timeout: timedelta | None = ...
    ) -> Work:
        """
        reduce_scatter(*args, **kwargs)
        Overloaded function.

        1. reduce_scatter(self: torch._C._distributed_c10d.ProcessGroup, output_tensors: collections.abc.Sequence[torch.Tensor], input_tensors: collections.abc.Sequence[collections.abc.Sequence[torch.Tensor]], opts: torch._C._distributed_c10d.ReduceScatterOptions = <torch._C._distributed_c10d.ReduceScatterOptions object at 0x113d37a70>) -> c10d::Work

        Reduces and scatters the input tensors from all processes across the process group.

                      See :func:`torch.distributed.reduce_scatter` for more details.

        2. reduce_scatter(self: torch._C._distributed_c10d.ProcessGroup, output: torch.Tensor, input: collections.abc.Sequence[torch.Tensor], op: torch._C._distributed_c10d.ReduceOp = <RedOpType.SUM: 0>, timeout: datetime.timedelta | None = None) -> c10d::Work

        Reduces and scatters the input tensors from all processes across the process group.

                      See :func:`torch.distributed.reduce_scatter` for more details.
        """
    @overload
    def alltoall_base(
        self,
        output_tensor: Tensor,
        input_tensor: Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        opts=...,
    ) -> Work:
        """
        alltoall_base(*args, **kwargs)
        Overloaded function.

        1. alltoall_base(self: torch._C._distributed_c10d.ProcessGroup, output: torch.Tensor, input: torch.Tensor, output_split_sizes: collections.abc.Sequence[typing.SupportsInt], input_split_sizes: collections.abc.Sequence[typing.SupportsInt], opts: torch._C._distributed_c10d.AllToAllOptions = <torch._C._distributed_c10d.AllToAllOptions object at 0x1112c7d30>) -> c10d::Work

        Alltoalls the input tensors from all processes across the process group.

                      See :func:`torch.distributed.all_to_all` for more details.

        2. alltoall_base(self: torch._C._distributed_c10d.ProcessGroup, output: torch.Tensor, input: torch.Tensor, output_split_sizes: collections.abc.Sequence[typing.SupportsInt], input_split_sizes: collections.abc.Sequence[typing.SupportsInt], timeout: datetime.timedelta | None = None) -> c10d::Work

        Alltoalls the input tensors from all processes across the process group.

                      See :func:`torch.distributed.all_to_all` for more details.
        """
    @overload
    def alltoall_base(
        self,
        output: Tensor,
        input: Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        timeout: timedelta | None = ...,
    ) -> Work:
        """
        alltoall_base(*args, **kwargs)
        Overloaded function.

        1. alltoall_base(self: torch._C._distributed_c10d.ProcessGroup, output: torch.Tensor, input: torch.Tensor, output_split_sizes: collections.abc.Sequence[typing.SupportsInt], input_split_sizes: collections.abc.Sequence[typing.SupportsInt], opts: torch._C._distributed_c10d.AllToAllOptions = <torch._C._distributed_c10d.AllToAllOptions object at 0x1112c7d30>) -> c10d::Work

        Alltoalls the input tensors from all processes across the process group.

                      See :func:`torch.distributed.all_to_all` for more details.

        2. alltoall_base(self: torch._C._distributed_c10d.ProcessGroup, output: torch.Tensor, input: torch.Tensor, output_split_sizes: collections.abc.Sequence[typing.SupportsInt], input_split_sizes: collections.abc.Sequence[typing.SupportsInt], timeout: datetime.timedelta | None = None) -> c10d::Work

        Alltoalls the input tensors from all processes across the process group.

                      See :func:`torch.distributed.all_to_all` for more details.
        """
    @overload
    def alltoall(self, output_tensor: list[Tensor], input_tensor: list[Tensor], opts=...) -> Work:
        """
        alltoall(self: torch._C._distributed_c10d.ProcessGroup, output_tensors: collections.abc.Sequence[torch.Tensor], input_tensors: collections.abc.Sequence[torch.Tensor], opts: torch._C._distributed_c10d.AllToAllOptions = <torch._C._distributed_c10d.AllToAllOptions object at 0x111257ef0>) -> c10d::Work

        Alltoalls the input tensors from all processes across the process group.

                      See :func:`torch.distributed.all_to_all` for more details.
        """
    @overload
    def alltoall(self, output: list[Tensor], input: list[Tensor], timeout: timedelta | None = ...) -> Work:
        """
        alltoall(self: torch._C._distributed_c10d.ProcessGroup, output_tensors: collections.abc.Sequence[torch.Tensor], input_tensors: collections.abc.Sequence[torch.Tensor], opts: torch._C._distributed_c10d.AllToAllOptions = <torch._C._distributed_c10d.AllToAllOptions object at 0x111257ef0>) -> c10d::Work

        Alltoalls the input tensors from all processes across the process group.

                      See :func:`torch.distributed.all_to_all` for more details.
        """
    def send(self, tensors: list[Tensor], dstRank: int, tag: int) -> Work:
        """
        send(self: torch._C._distributed_c10d.ProcessGroup, tensors: collections.abc.Sequence[torch.Tensor], dstRank: typing.SupportsInt, tag: typing.SupportsInt) -> c10d::Work

        Sends the tensor to the specified rank.

                      See :func:`torch.distributed.send` for more details.
        """
    def recv(self, tensors: list[Tensor], srcRank: int, tag: int) -> Work:
        """
        recv(self: torch._C._distributed_c10d.ProcessGroup, tensors: collections.abc.Sequence[torch.Tensor], srcRank: typing.SupportsInt, tag: typing.SupportsInt) -> c10d::Work

        Receives the tensor from the specified rank.

                      See :func:`torch.distributed.recv` for more details.
        """
    def recv_anysource(self, tensors: list[Tensor], tag: int) -> Work:
        """
        recv_anysource(self: torch._C._distributed_c10d.ProcessGroup, arg0: collections.abc.Sequence[torch.Tensor], arg1: typing.SupportsInt) -> c10d::Work

        Receives the tensor from any source.

                      See :func:`torch.distributed.recv` for more details.
        """
    @overload
    def barrier(self, opts=...) -> Work:
        """
        barrier(*args, **kwargs)
        Overloaded function.

        1. barrier(self: torch._C._distributed_c10d.ProcessGroup, opts: torch._C._distributed_c10d.BarrierOptions = <torch._C._distributed_c10d.BarrierOptions object at 0x113d3c0b0>) -> c10d::Work

        Blocks until all processes in the group enter the call, and
                      then all leave the call together.

                      See :func:`torch.distributed.barrier` for more details.

        2. barrier(self: torch._C._distributed_c10d.ProcessGroup, timeout: datetime.timedelta | None = None) -> c10d::Work

        Blocks until all processes in the group enter the call, and
                      then all leave the call together.

                      See :func:`torch.distributed.barrier` for more details.
        """
    @overload
    def barrier(self, timeout: timedelta | None = ...) -> Work:
        """
        barrier(*args, **kwargs)
        Overloaded function.

        1. barrier(self: torch._C._distributed_c10d.ProcessGroup, opts: torch._C._distributed_c10d.BarrierOptions = <torch._C._distributed_c10d.BarrierOptions object at 0x113d3c0b0>) -> c10d::Work

        Blocks until all processes in the group enter the call, and
                      then all leave the call together.

                      See :func:`torch.distributed.barrier` for more details.

        2. barrier(self: torch._C._distributed_c10d.ProcessGroup, timeout: datetime.timedelta | None = None) -> c10d::Work

        Blocks until all processes in the group enter the call, and
                      then all leave the call together.

                      See :func:`torch.distributed.barrier` for more details.
        """
    def boxed(self) -> ScriptObject:
        """boxed(self: torch._C._distributed_c10d.ProcessGroup) -> object"""
    @staticmethod
    def unbox(obj: ScriptObject) -> ProcessGroup:
        """unbox(arg0: object) -> torch._C._distributed_c10d.ProcessGroup"""
    def name(self) -> str:
        """
        name(self: torch._C._distributed_c10d.ProcessGroup) -> str

        Get the name of this process group.
        """
    @property
    def bound_device_id(self) -> torch.device | None: ...
    @bound_device_id.setter
    def bound_device_id(self, device: torch.device | None) -> None: ...
    @property
    def group_name(self) -> str:
        """(Gets this process group name. It's cluster unique)"""
    @property
    def group_desc(self) -> str:
        """Gets this process group description"""

class FakeProcessGroup(Backend):
    def __init__(self, rank: int, world_size: int) -> None:
        """__init__(self: torch._C._distributed_c10d.FakeProcessGroup, rank: typing.SupportsInt, world_size: typing.SupportsInt, options: torch._C._distributed_c10d.FakeProcessGroup.Options = <torch._C._distributed_c10d.FakeProcessGroup.Options object at 0x11130caf0>) -> None"""

class FakeWork(Work):
    seq_id: int
    def __init__(self) -> None:
        """__init__(self: torch._C._distributed_c10d.FakeWork) -> None"""
    def wait(self, timeout: timedelta = ...) -> bool:
        """wait(self: torch._C._distributed_c10d.FakeWork, timeout: datetime.timedelta = datetime.timedelta(0)) -> bool"""
    def getFuture(self) -> Future:
        """getFuture(self: torch._C._distributed_c10d.FakeWork) -> c10::ivalue::Future"""

class ProcessGroupGloo(Backend):
    class Device: ...

    class Options(Backend.Options):
        """
        Base class for all backend options implementations, such as the nccl
        options :class:`~torch.distributed.ProcessGroupNCCL.Options`).
        """

        devices: list[ProcessGroupGloo.Device]
        threads: int
        def __init__(self) -> None:
            """__init__(self: torch._C._distributed_c10d.Backend.Options, backend: str, timeout: datetime.timedelta = datetime.timedelta(seconds=1800)) -> None"""

    def __init__(self, store: Store, rank: int, size: int, timeout: timedelta) -> None:
        """
        __init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: torch._C._distributed_c10d.ProcessGroupGloo, store: torch._C._distributed_c10d.Store, rank: typing.SupportsInt, size: typing.SupportsInt, options: torch._C._distributed_c10d.ProcessGroupGloo._Options) -> None

        Create a new ProcessGroupGloo instance.

        2. __init__(self: torch._C._distributed_c10d.ProcessGroupGloo, store: torch._C._distributed_c10d.Store, rank: typing.SupportsInt, size: typing.SupportsInt, timeout: datetime.timedelta = datetime.timedelta(seconds=1800)) -> None

        Create a new ProcessGroupGloo instance.
        """
    @staticmethod
    def create_device(hostname=..., interface=..., lazy_init=...) -> Device:
        """create_device(hostname: str = '', interface: str = '', lazy_init: bool | None = None) -> torch._C._distributed_c10d.ProcessGroupGloo.Device"""
    @staticmethod
    def create_default_device(lazy_init=...) -> Device:
        """create_default_device(lazy_init: bool | None = None) -> torch._C._distributed_c10d.ProcessGroupGloo.Device"""
    @property
    def options(self) -> Options:
        """Return the options used to create this ProcessGroupGloo instance."""

class _ProcessGroupWrapper(Backend):
    def __init__(self, pg: Backend, gloo_pg: ProcessGroupGloo) -> None:
        """__init__(self: torch._C._distributed_c10d._ProcessGroupWrapper, backend: torch._C._distributed_c10d.Backend, gloo_backend: torch._C._distributed_c10d.Backend) -> None"""

    wrapped_pg: Backend

class ErrorType(Enum):
    """
    Members:

    SUCCESS

    TIMEOUT

    COMM_ERROR

    REMOTE_ERROR
    """

    SUCCESS = ...
    TIMEOUT = ...
    COMM_ERROR = ...
    REMOTE_ERROR = ...

class ProcessGroupNCCL(Backend):
    class NCCLConfig:
        blocking: int
        cga_cluster_size: int
        min_ctas: int
        max_ctas: int
        def unsafe_get_ptr(self) -> int: ...

    class Options(Backend.Options):
        config: ProcessGroupNCCL.NCCLConfig
        is_high_priority_stream: bool
        split_from: ProcessGroupNCCL
        split_color: int
        def __init__(self, is_high_priority_stream: bool = ...) -> None: ...

    def __init__(self, store: Store, rank: int, size: int, options: Options) -> None: ...
    def perform_nocolor_split(self, device: torch.device) -> None: ...
    def register_mem_pool(self, pool: torch.cuda.MemPool) -> None: ...
    def deregister_mem_pool(self, pool: torch.cuda.MemPool) -> None: ...
    def comm_split_count(self) -> int: ...
    def abort(self) -> None: ...
    @property
    def uid(self) -> int: ...
    @property
    def options(self) -> Options: ...
    @staticmethod
    def get_build_nccl_version(self) -> tuple[int, int, int]: ...
    @staticmethod
    def get_runtime_nccl_version(self) -> tuple[int, int, int]: ...

class ProcessGroupUCC(Backend):
    def __init__(self, store: Store, rank: int, size: int, timeout: timedelta) -> None: ...

class ProcessGroupMPI(Backend):
    def __init__(self, rank: int, size: int, pgComm: int) -> None: ...
    @staticmethod
    def create(ranks: list[int]) -> ProcessGroupMPI: ...

class _SymmetricMemory:
    @staticmethod
    def set_group_info(group_name: str, rank: int, world_size: int, store: Store) -> None:
        """set_group_info(arg0: str, arg1: typing.SupportsInt, arg2: typing.SupportsInt, arg3: c10d::Store) -> None"""
    @staticmethod
    def empty_strided_p2p(
        size: torch.types._size,
        stride: torch.types._size,
        dtype: torch.dtype,
        device: torch.device,
        group_name: str | None = ...,
        alloc_id: int | None = ...,
    ) -> torch.Tensor:
        """empty_strided_p2p(size: Tuple[int, ...], stride: Tuple[int, ...], dtype: torch.dtype, device: torch.device, group_name: str | None = None, alloc_id: typing.SupportsInt | None = None) -> torch.Tensor"""
    @staticmethod
    def has_multicast_support(device_type: DeviceType, device_idx: int) -> bool:
        """has_multicast_support(arg0: c10::DeviceType, arg1: typing.SupportsInt) -> bool"""
    @staticmethod
    def set_backend(name: str) -> None:
        """set_backend(arg0: str) -> None"""
    @staticmethod
    def get_backend(device: torch.device) -> str | None:
        """get_backend(arg0: torch.device) -> str | None"""
    @staticmethod
    def get_mempool_allocator(device: torch.device) -> Any:
        """get_mempool_allocator(arg0: torch.device) -> c10::Allocator"""
    @property
    def rank(self) -> int: ...
    @property
    def world_size(self) -> int: ...
    @staticmethod
    def rendezvous(tensor: torch.Tensor, group_name: str | None = ...) -> _SymmetricMemory:
        """rendezvous(tensor: torch.Tensor, group_name: str | None = None) -> torch._C._distributed_c10d._SymmetricMemory"""
    def get_buffer(
        self, rank: int, sizes: torch.types._size, dtype: torch.dtype, storage_offset: int | None = ...
    ) -> torch.Tensor:
        """get_buffer(self: torch._C._distributed_c10d._SymmetricMemory, rank: typing.SupportsInt, sizes: Tuple[int, ...], dtype: torch.dtype, storage_offset: typing.SupportsInt = 0) -> torch.Tensor"""
    def get_signal_pad(
        self,
        rank: int,
        sizes: torch.types._size = ...,
        dtype: torch.dtype | None = ...,
        storage_offset: int | None = ...,
    ) -> torch.Tensor:
        """get_signal_pad(self: torch._C._distributed_c10d._SymmetricMemory, rank: typing.SupportsInt, sizes: Tuple[int, ...] = [], dtype: torch.dtype | None = None, storage_offset: typing.SupportsInt = 0) -> torch.Tensor"""
    def barrier(self, channel: int = ..., timeout_ms: int = ...) -> None:
        """barrier(self: torch._C._distributed_c10d._SymmetricMemory, channel: typing.SupportsInt = 0, timeout_ms: typing.SupportsInt = 0) -> None"""
    def put_signal(self, dst_rank: int, channel: int = ..., timeout_ms: int = ...) -> None:
        """put_signal(self: torch._C._distributed_c10d._SymmetricMemory, dst_rank: typing.SupportsInt, channel: typing.SupportsInt = 0, timeout_ms: typing.SupportsInt = 0) -> None"""
    def wait_signal(self, src_rank: int, channel: int = ..., timeout_ms: int = ...) -> None:
        """wait_signal(self: torch._C._distributed_c10d._SymmetricMemory, src_rank: typing.SupportsInt, channel: typing.SupportsInt = 0, timeout_ms: typing.SupportsInt = 0) -> None"""
    def get_remote_tensor(self, peer: int, sizes: torch.types._size, dtype: torch.dtype) -> torch.Tensor:
        """get_remote_tensor(self: torch._C._distributed_c10d._SymmetricMemory, peer: typing.SupportsInt, sizes: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor"""
    @staticmethod
    def memset32(tensor: torch.Tensor, offset: int, val: int, count: int = ...) -> torch.Tensor:
        """memset32(input: torch.Tensor, offset: typing.SupportsInt, val: typing.SupportsInt, count: typing.SupportsInt = 1) -> torch.Tensor"""
    @staticmethod
    def stream_write_value32(tensor: torch.Tensor, offset: int, val: int) -> torch.Tensor:
        """stream_write_value32(input: torch.Tensor, offset: typing.SupportsInt, val: typing.SupportsInt) -> torch.Tensor"""
    @property
    def buffer_ptrs(self) -> list[int]: ...
    @property
    def buffer_ptrs_dev(self) -> int: ...
    @property
    def signal_pad_ptrs(self) -> list[int]: ...
    @property
    def signal_pad_ptrs_dev(self) -> int: ...
    @property
    def multicast_ptr(self) -> int: ...
    @property
    def buffer_size(self) -> int: ...
    @property
    def signal_pad_size(self) -> int: ...

class ProcessGroupXCCL(Backend):
    class Options(Backend.Options):
        def __init__(self) -> None: ...

    def __init__(self, store: Store, rank: int, size: int, options: Options) -> None: ...
    @property
    def options(self) -> Options: ...
