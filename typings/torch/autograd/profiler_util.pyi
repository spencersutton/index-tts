from collections import UserList, defaultdict
from typing import Self
from warnings import deprecated

__all__ = [
    "EventList",
    "FormattedTimesMixin",
    "FunctionEvent",
    "FunctionEventAvg",
    "Interval",
    "Kernel",
    "MemRecordsAcc",
    "StringTable",
]

class EventList(UserList):
    """A list of Events (for pretty printing)."""
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def self_cpu_time_total(self) -> int: ...
    def table(
        self,
        sort_by=...,
        row_limit=...,
        max_src_column_width=...,
        max_name_column_width=...,
        max_shapes_column_width=...,
        header=...,
        top_level_events_only=...,
        time_unit=...,
    ):
        """
        Print an EventList as a nicely formatted table.

        Args:
            sort_by (str, optional): Attribute used to sort entries. By default
                they are printed in the same order as they were registered.
                Valid keys include: ``cpu_time``, ``cuda_time``, ``xpu_time``,
                ``cpu_time_total``, ``cuda_time_total``, ``xpu_time_total``,
                ``cpu_memory_usage``, ``cuda_memory_usage``, ``xpu_memory_usage``,
                ``self_cpu_memory_usage``, ``self_cuda_memory_usage``,
                ``self_xpu_memory_usage``, ``count``.
            top_level_events_only(bool, optional): Boolean flag to determine the
                selection of events to display. If true, the profiler will only
                display events at top level like top-level invocation of python
                `lstm`, python `add` or other functions, nested events like low-level
                cpu/cuda/xpu ops events are omitted for profiler result readability.
            time_unit(str, optional): A time unit to be used for all values in the
                table. Valid options are: ``s``, ``ms`` and ``us``.

        Returns:
            A string containing the table.
        """
    def export_chrome_trace(self, path) -> None:
        """
        Export an EventList as a Chrome tracing tools file.

        The checkpoint can be later loaded and inspected under ``chrome://tracing`` URL.

        Args:
            path (str): Path where the trace will be written.
        """
    def supported_export_stacks_metrics(self) -> list[str]: ...
    def export_stacks(self, path: str, metric: str) -> None: ...
    def key_averages(self, group_by_input_shapes=..., group_by_stack_n=..., group_by_overload_name=...) -> EventList:
        """
        Averages all function events over their keys.

        Args:
            group_by_input_shapes: group entries by
                (event name, input shapes) rather than just event name.
                This is useful to see which input shapes contribute to the runtime
                the most and may help with size-specific optimizations or
                choosing the best candidates for quantization (aka fitting a roof line)

            group_by_stack_n: group by top n stack trace entries

            group_by_overload_name: Differentiate operators by their overload name e.g. aten::add.Tensor
            and aten::add.out will be aggregated separately

        Returns:
            An EventList containing FunctionEventAvg objects.
        """
    def total_average(self) -> FunctionEventAvg:
        """
        Averages all events.

        Returns:
            A FunctionEventAvg object.
        """

class FormattedTimesMixin:
    """
    Helpers for FunctionEvent and FunctionEventAvg.

    The subclass should define `*_time_total` and `count` attributes.
    """

    cpu_time_str = ...
    device_time_str = ...
    cpu_time_total_str = ...
    device_time_total_str = ...
    self_cpu_time_total_str = ...
    self_device_time_total_str = ...
    @property
    def cpu_time(self) -> float: ...
    @property
    def device_time(self) -> float: ...
    @property
    @deprecated("`cuda_time` is deprecated, please use `device_time` instead.", category=FutureWarning)
    def cuda_time(self) -> float: ...

class Interval:
    def __init__(self, start, end) -> None: ...
    def elapsed_us(self):
        """Returns the length of the interval"""

Kernel = ...

class FunctionEvent(FormattedTimesMixin):
    """Profiling information about a single function."""
    def __init__(
        self,
        id,
        name,
        thread,
        start_us,
        end_us,
        overload_name=...,
        fwd_thread=...,
        input_shapes=...,
        stack=...,
        scope=...,
        use_device=...,
        cpu_memory_usage=...,
        device_memory_usage=...,
        is_async=...,
        is_remote=...,
        sequence_nr=...,
        node_id=...,
        device_type=...,
        device_index=...,
        device_resource_id=...,
        is_legacy=...,
        flops=...,
        trace_name=...,
        concrete_inputs=...,
        kwinputs=...,
        is_user_annotation=...,
    ) -> None: ...
    def append_kernel(self, name, device, duration) -> None: ...
    def append_cpu_child(self, child) -> None:
        """
        Append a CPU child of type FunctionEvent.

        One is supposed to append only direct children to the event to have
        correct self cpu time being reported.
        """
    def set_cpu_parent(self, parent) -> None:
        """
        Set the immediate CPU parent of type FunctionEvent.

        One profiling FunctionEvent should have only one CPU parent such that
        the child's range interval is completely inside the parent's. We use
        this connection to determine the event is from top-level op or not.
        """
    @property
    def self_cpu_memory_usage(self) -> int: ...
    @property
    def self_device_memory_usage(self) -> int: ...
    @property
    @deprecated(
        "`self_cuda_memory_usage` is deprecated. Use `self_device_memory_usage` instead.", category=FutureWarning
    )
    def self_cuda_memory_usage(self) -> int: ...
    @property
    def cpu_time_total(self) -> Literal[0]: ...
    @property
    def self_cpu_time_total(self) -> int: ...
    @property
    def device_time_total(self) -> int: ...
    @property
    @deprecated("`cuda_time_total` is deprecated. Use `device_time_total` instead.", category=FutureWarning)
    def cuda_time_total(self) -> int: ...
    @property
    def self_device_time_total(self) -> int: ...
    @property
    @deprecated("`self_cuda_time_total` is deprecated. Use `self_device_time_total` instead.", category=FutureWarning)
    def self_cuda_time_total(self) -> int: ...
    @property
    def key(self) -> str: ...

class FunctionEventAvg(FormattedTimesMixin):
    """Used to average stats over multiple FunctionEvent objects."""
    def __init__(self) -> None: ...
    def add(self, other) -> Self: ...
    def __iadd__(self, other) -> Self: ...

class StringTable(defaultdict):
    def __missing__(self, key): ...

class MemRecordsAcc:
    """Acceleration structure for accessing mem_records in interval."""
    def __init__(self, mem_records) -> None: ...
    def in_interval(self, start_us, end_us) -> Generator[Any, Any, None]:
        """
        Return all records in the given interval
        To maintain backward compatibility, convert us to ns in function
        """

MEMORY_EVENT_NAME = ...
OUT_OF_MEMORY_EVENT_NAME = ...
