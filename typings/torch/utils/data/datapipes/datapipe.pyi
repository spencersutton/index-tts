from collections import UserList
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Literal, TypeVar

from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
UNTRACABLE_DATAFRAME_PIPES: Any

class DataChunk(UserList[_T]):
    items: list[_T]
    def __init__(self, items: Iterable[_T]) -> None: ...
    def as_str(self, indent: str = ...) -> str: ...
    def __iter__(self) -> Iterator[_T]: ...
    def raw_iterator(self) -> Iterator[_T]: ...

class MapDataPipe(Dataset[_T_co], metaclass=_DataPipeMeta):
    """
    Map-style DataPipe.

    All datasets that represent a map from keys to data samples should subclass this.
    Subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given, unique key. Subclasses can also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    These DataPipes can be invoked in two ways, using the class constructor or applying their
    functional form onto an existing `MapDataPipe` (recommend, available to most but not all DataPipes).

    Note:
        :class:`~torch.utils.data.DataLoader` by default constructs an index
        sampler that yields integral indices. To make it work with a map-style
        DataPipe with non-integral indices/keys, a custom sampler must be provided.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper, Mapper
        >>> dp = SequenceWrapper(range(10))
        >>> map_dp_1 = dp.map(lambda x: x + 1)  # Using functional form (recommended)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> map_dp_2 = Mapper(dp, lambda x: x + 1)  # Using class constructor
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> batch_dp = map_dp_1.batch(batch_size=2)
        >>> list(batch_dp)
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    """

    functions: dict[str, Callable] = ...
    reduce_ex_hook: Callable | None = ...
    getstate_hook: Callable | None = ...
    str_hook: Callable | None = ...
    repr_hook: Callable | None = ...
    def __getattr__(self, attribute_name: Any): ...
    @classmethod
    def register_function(cls, function_name: Any, function: Any) -> None: ...
    @classmethod
    def register_datapipe_as_function(cls, function_name: Any, cls_to_register: Any): ...
    def __getstate__(self):
        """
        Serialize `lambda` functions when `dill` is available.

        If this doesn't cover your custom DataPipe's use case, consider writing custom methods for
        `__getstate__` and `__setstate__`, or use `pickle.dumps` for serialization.
        """
    def __reduce_ex__(self, *args: Any, **kwargs: Any): ...
    @classmethod
    def set_getstate_hook(cls, hook_fn: Any) -> None: ...
    @classmethod
    def set_reduce_ex_hook(cls, hook_fn: Any) -> None: ...
    def batch(self, batch_size: int, drop_last: bool = ..., wrapper_class: type[DataChunk] = ...) -> MapDataPipe: ...
    def concat(self, *datapipes: MapDataPipe) -> MapDataPipe: ...
    def map(self, fn: Callable = ...) -> MapDataPipe: ...
    def shuffle(self, *, indices: list | None = ...) -> IterDataPipe: ...
    def zip(self, *datapipes: MapDataPipe[_T_co]) -> MapDataPipe: ...

class IterDataPipe(IterableDataset[_T_co], metaclass=_IterDataPipeMeta):
    """
    Iterable-style DataPipe.

    All DataPipes that represent an iterable of data samples should subclass this.
    This style of DataPipes is particularly useful when data come from a stream, or
    when the number of samples is too large to fit them all in memory. ``IterDataPipe`` is lazily initialized and its
    elements are computed only when ``next()`` is called on the iterator of an ``IterDataPipe``.

    All subclasses should overwrite :meth:`__iter__`, which would return an
    iterator of samples in this DataPipe. Calling ``__iter__`` of an ``IterDataPipe`` automatically invokes its
    method ``reset()``, which by default performs no operation. When writing a custom ``IterDataPipe``, users should
    override ``reset()`` if necessary. The common usages include resetting buffers, pointers,
    and various state variables within the custom ``IterDataPipe``.

    Note:
        Only `one` iterator can be valid for each ``IterDataPipe`` at a time,
        and the creation a second iterator will invalidate the first one. This constraint is necessary because
        some ``IterDataPipe`` have internal buffers, whose states can become invalid if there are multiple iterators.
        The code example below presents details on how this constraint looks in practice.
        If you have any feedback related to this constraint, please see `GitHub IterDataPipe Single Iterator Issue`_.

    These DataPipes can be invoked in two ways, using the class constructor or applying their
    functional form onto an existing ``IterDataPipe`` (recommended, available to most but not all DataPipes).
    You can chain multiple `IterDataPipe` together to form a pipeline that will perform multiple
    operations in succession.

    .. _GitHub IterDataPipe Single Iterator Issue:
        https://github.com/pytorch/data/issues/45

    Note:
        When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
        item in the DataPipe will be yielded from the :class:`~torch.utils.data.DataLoader`
        iterator. When :attr:`num_workers > 0`, each worker process will have a
        different copy of the DataPipe object, so it is often desired to configure
        each copy independently to avoid having duplicate data returned from the
        workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
        process, returns information about the worker. It can be used in either the
        dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
        :attr:`worker_init_fn` option to modify each copy's behavior.

    Examples:
        General Usage:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
            >>> dp = IterableWrapper(range(10))
            >>> map_dp_1 = Mapper(dp, lambda x: x + 1)  # Using class constructor
            >>> map_dp_2 = dp.map(
            ...     lambda x: x + 1
            ... )  # Using functional form (recommended)
            >>> list(map_dp_1)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> list(map_dp_2)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> filter_dp = map_dp_1.filter(lambda x: x % 2 == 0)
            >>> list(filter_dp)
            [2, 4, 6, 8, 10]
        Single Iterator Constraint Example:
            >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
            >>> source_dp = IterableWrapper(range(10))
            >>> it1 = iter(source_dp)
            >>> list(it1)
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            >>> it1 = iter(source_dp)
            >>> it2 = iter(
            ...     source_dp
            ... )  # The creation of a new iterator invalidates `it1`
            >>> next(it2)
            0
            >>> next(it1)  # Further usage of `it1` will raise a `RunTimeError`
    """

    functions: dict[str, Callable] = ...
    reduce_ex_hook: Callable | None = ...
    getstate_hook: Callable | None = ...
    str_hook: Callable | None = ...
    repr_hook: Callable | None = ...
    _number_of_samples_yielded: int = ...
    _snapshot_state: _SnapshotState = ...
    _fast_forward_iterator: Iterator | None = ...
    def __getattr__(self, attribute_name: Any): ...
    @classmethod
    def register_function(cls, function_name: Any, function: Any) -> None: ...
    @classmethod
    def register_datapipe_as_function(
        cls, function_name: Any, cls_to_register: Any, enable_df_api_tracing: bool = ...
    ): ...
    def __getstate__(self):
        """
        Serialize `lambda` functions when `dill` is available.

        If this doesn't cover your custom DataPipe's use case, consider writing custom methods for
        `__getstate__` and `__setstate__`, or use `pickle.dumps` for serialization.
        """
    def __reduce_ex__(self, *args: Any, **kwargs: Any): ...
    @classmethod
    def set_getstate_hook(cls, hook_fn: Any) -> None: ...
    @classmethod
    def set_reduce_ex_hook(cls, hook_fn: Any) -> None: ...
    def batch(self, batch_size: int, drop_last: bool = ..., wrapper_class: type[DataChunk] = ...) -> IterDataPipe: ...
    def collate(
        self,
        conversion: Callable[..., Any] | dict[str | Any, Callable | Any] | None = ...,
        collate_fn: Callable | None = ...,
    ) -> IterDataPipe: ...
    def concat(self, *datapipes: IterDataPipe) -> IterDataPipe: ...
    def demux(
        self,
        num_instances: int,
        classifier_fn: Callable[[_T_co], int | None],
        drop_none: bool = ...,
        buffer_size: int = ...,
    ) -> list[IterDataPipe]: ...
    def filter(self, filter_fn: Callable, input_col=...) -> IterDataPipe: ...
    def fork(
        self, num_instances: int, buffer_size: int = ..., copy: Literal["shallow", "deep"] | None = ...
    ) -> list[IterDataPipe]: ...
    def groupby(
        self,
        group_key_fn: Callable[[_T_co], Any],
        *,
        keep_key: bool = ...,
        buffer_size: int = ...,
        group_size: int | None = ...,
        guaranteed_group_size: int | None = ...,
        drop_remaining: bool = ...,
    ) -> IterDataPipe: ...
    def list_files(
        self,
        masks: str | list[str] = ...,
        *,
        recursive: bool = ...,
        abspath: bool = ...,
        non_deterministic: bool = ...,
        length: int = ...,
    ) -> IterDataPipe: ...
    def map(self, fn: Callable, input_col=..., output_col=...) -> IterDataPipe: ...
    def mux(self, *datapipes) -> IterDataPipe: ...
    def open_files(self, mode: str = ..., encoding: str | None = ..., length: int = ...) -> IterDataPipe: ...
    def read_from_stream(self, chunk: int | None = ...) -> IterDataPipe: ...
    def routed_decode(self, *handlers: Callable, key_fn: Callable = ...) -> IterDataPipe: ...
    def sharding_filter(self, sharding_group_filter=...) -> IterDataPipe: ...
    def shuffle(self, *, buffer_size: int = ..., unbatch_level: int = ...) -> IterDataPipe: ...
    def unbatch(self, unbatch_level: int = ...) -> IterDataPipe: ...
    def zip(self, *datapipes: IterDataPipe) -> IterDataPipe: ...

class DFIterDataPipe(IterDataPipe):
    def __iter__(self): ...

class _DataPipeSerializationWrapper:
    def __init__(self, datapipe) -> None: ...
    def __getstate__(self): ...
    def __setstate__(self, state): ...
    def __len__(self) -> int: ...

class _IterDataPipeSerializationWrapper(_DataPipeSerializationWrapper, IterDataPipe):
    def __iter__(self): ...

class _MapDataPipeSerializationWrapper(_DataPipeSerializationWrapper, MapDataPipe):
    def __getitem__(self, idx): ...
