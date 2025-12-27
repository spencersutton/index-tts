import datetime
from collections.abc import Callable
from enum import Enum
from types import TracebackType

class Aggregation(Enum):
    """
            These are types of aggregations that can be used to accumulate stats.


    Members:

      VALUE :
                VALUE returns the last value to be added.


      MEAN :
                MEAN computes the arithmetic mean of all the added values.


      COUNT :
                COUNT returns the total number of added values.


      SUM :
                SUM returns the sum of the added values.


      MAX :
                MAX returns the max of the added values.


      MIN :
                MIN returns the min of the added values.

    """

    VALUE = ...
    MEAN = ...
    COUNT = ...
    SUM = ...
    MAX = ...
    MIN = ...

class Stat:
    """
    Stat is used to compute summary statistics in a performant way over
    fixed intervals. Stat logs the statistics as an Event once every
    ``window_size`` duration. When the window closes the stats are logged
    via the event handlers as a ``torch.monitor.Stat`` event.

    ``window_size`` should be set to something relatively high to avoid a
    huge number of events being logged. Ex: 60s. Stat uses millisecond
    precision.

    If ``max_samples`` is set, the stat will cap the number of samples per
    window by discarding `add` calls once ``max_samples`` adds have
    occurred. If it's not set, all ``add`` calls during the window will be
    included. This is an optional field to make aggregations more directly
    comparable across windows when the number of samples might vary.

    When the Stat is destructed it will log any remaining data even if the
    window hasn't elapsed.
    """

    name: str
    count: int
    def __init__(self, name: str, aggregations: list[Aggregation], window_size: int, max_samples: int = ...) -> None:
        """
        __init__(self: torch._C._monitor.Stat, name: str, aggregations: collections.abc.Sequence[torch._C._monitor.Aggregation], window_size: datetime.timedelta, max_samples: typing.SupportsInt = 9223372036854775807) -> None


        Constructs the ``Stat``.
        """
    def add(self, v: float) -> None:
        """
        add(self: torch._C._monitor.Stat, v: typing.SupportsFloat) -> None


        Adds a value to the stat to be aggregated according to the
        configured stat type and aggregations.
        """
    def get(self) -> dict[Aggregation, float]:
        """
        get(self: torch._C._monitor.Stat) -> dict[torch._C._monitor.Aggregation, float]


        Returns the current value of the stat, primarily for testing
        purposes. If the stat has logged and no additional values have been
        added this will be zero.
        """

class Event:
    """
    Event represents a specific typed event to be logged. This can represent
    high-level data points such as loss or accuracy per epoch or more
    low-level aggregations such as through the Stats provided through this
    library.

    All Events of the same type should have the same name so downstream
    handlers can correctly process them.
    """

    name: str
    timestamp: datetime.datetime
    data: dict[str, int | float | bool | str]
    def __init__(self, name: str, timestamp: datetime.datetime, data: dict[str, int | float | bool | str]) -> None:
        """
        __init__(self: torch._C._monitor.Event, name: str, timestamp: datetime.datetime, data: collections.abc.Mapping[str, data_value_t]) -> None


        Constructs the ``Event``.
        """

def log_event(e: Event) -> None:
    """
    log_event(event: torch._C._monitor.Event) -> None


    log_event logs the specified event to all of the registered event
    handlers. It's up to the event handlers to log the event out to the
    corresponding event sink.

    If there are no event handlers registered this method is a no-op.
    """

class EventHandlerHandle:
    """
    EventHandlerHandle is a wrapper type returned by
    ``register_event_handler`` used to unregister the handler via
    ``unregister_event_handler``. This cannot be directly initialized.
    """

def register_event_handler(handler: Callable[[Event], None]) -> EventHandlerHandle:
    """
    register_event_handler(callback: collections.abc.Callable[[torch._C._monitor.Event], None]) -> torch._C._monitor.EventHandlerHandle


    register_event_handler registers a callback to be called whenever an
    event is logged via ``log_event``. These handlers should avoid blocking
    the main thread since that may interfere with training as they run
    during the ``log_event`` call.
    """

def unregister_event_handler(handle: EventHandlerHandle) -> None:
    """
    unregister_event_handler(handler: torch._C._monitor.EventHandlerHandle) -> None


    unregister_event_handler unregisters the ``EventHandlerHandle`` returned
    after calling ``register_event_handler``. After this returns the event
    handler will no longer receive events.
    """

class _WaitCounterTracker:
    def __enter__(self) -> None:
        """__enter__(self: torch._C._monitor._WaitCounterTracker) -> None"""
    def __exit__(
        self,
        exc_type: type[BaseException] | None = ...,
        exc_value: BaseException | None = ...,
        traceback: TracebackType | None = ...,
    ) -> None:
        """__exit__(self: torch._C._monitor._WaitCounterTracker, *args) -> None"""

class _WaitCounter:
    """
    WaitCounter represents a named duration counter.
    Multiple units of work can be tracked by the same WaitCounter. Depending
    on the backend, the WaitCounter may track the number of units of work,
    their duration etc.
    """
    def __init__(self, key: str) -> None:
        """__init__(self: torch._C._monitor._WaitCounter, key: str) -> None"""
    def guard(self) -> _WaitCounterTracker:
        """
        guard(self: torch._C._monitor._WaitCounter) -> torch._C._monitor._WaitCounterTracker


        Creates a guard that manages a single unit of work.
        """
