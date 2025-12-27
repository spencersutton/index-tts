from warnings import deprecated

__all__ = ["profile"]

@deprecated(
    "`torch.autograd.profiler_legacy.profile` is deprecated and will be removed in a future release. "
    "Please use `torch.profiler` instead.",
    category=None,
)
class profile:
    """DEPRECATED: use torch.profiler instead."""
    def __init__(
        self,
        enabled=...,
        *,
        use_cuda=...,
        record_shapes=...,
        with_flops=...,
        profile_memory=...,
        with_stack=...,
        with_modules=...,
    ) -> None: ...
    def config(self) -> ProfilerConfig: ...
    def __enter__(self) -> Self | None: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False] | None: ...
    def table(
        self,
        sort_by=...,
        row_limit=...,
        max_src_column_width=...,
        max_name_column_width=...,
        max_shapes_column_width=...,
        header=...,
        top_level_events_only=...,
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
    def export_stacks(self, path: str, metric: str = ...) -> None: ...
    def key_averages(self, group_by_input_shape=..., group_by_stack_n=...) -> EventList:
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
    @property
    def self_cpu_time_total(self) -> int:
        """Return CPU time as the sum of self times across all events."""
