import dataclasses
from collections import UserDict, UserString
from collections.abc import Callable, Sequence, Set as AbstractSet
from typing import TYPE_CHECKING, Any, Literal

from gradio.blocks import Block, BlockContext, Component
from gradio.components import Timer
from gradio.data_classes import FileDataDict
from gradio_client.documentation import document

"""Contains all of the events that can be triggered in a gr.Blocks() app, with the exception
of the on-page-load event, which is defined in gr.Blocks().load()."""
if TYPE_CHECKING: ...

def set_cancel_events(
    triggers: Sequence[EventListenerMethod], cancels: None | dict[str, Any] | list[dict[str, Any]]
):  # -> None:
    ...

@document()
class Dependency(UserDict):
    def __init__(self, trigger, key_vals, dep_index, fn, associated_timer: Timer | None = ...) -> None: ...
    def __call__(self, *args, **kwargs): ...

@document()
class EventData:
    def __init__(self, target: Block | None, _data: Any) -> None: ...
    def __getattr__(self, name):  # -> Any:
        ...

@document()
class SelectData(EventData):
    def __init__(self, target: Block | None, data: Any) -> None: ...

@document()
class KeyUpData(EventData):
    def __init__(self, target: Block | None, data: Any) -> None: ...

@document()
class DeletedFileData(EventData):
    def __init__(self, target: Block | None, data: FileDataDict) -> None: ...

@document()
class LikeData(EventData):
    def __init__(self, target: Block | None, data: Any) -> None: ...

@document()
class RetryData(EventData):
    def __init__(self, target: Block | None, data: Any) -> None: ...

@document()
class UndoData(EventData):
    def __init__(self, target: Block | None, data: Any) -> None: ...

@document()
class EditData(EventData):
    def __init__(self, target: Block | None, data: Any) -> None: ...

@document()
class DownloadData(EventData):
    def __init__(self, target: Block | None, data: FileDataDict) -> None: ...

@document()
class CopyData(EventData):
    def __init__(self, target: Block | None, data: Any) -> None: ...

@dataclasses.dataclass
class EventListenerMethod:
    block: Block | None
    event_name: str

if TYPE_CHECKING:
    type EventListenerCallable = Callable[
        [
            Callable[..., Any] | None,
            Component | Sequence[Component] | None,
            Block | Sequence[Block] | Sequence[Component] | Component | None,
            str | None | Literal[False],
            bool,
            Literal["full", "minimal", "hidden"],
            Component | Sequence[Component] | None,
            bool | None,
            bool,
            int,
            bool,
            bool,
            dict[str, Any] | list[dict[str, Any]] | None,
            float | None,
            Literal["once", "multiple", "always_last"] | None,
            str | None,
            int | None | Literal["default"],
            str | None,
            bool,
        ],
        Dependency,
    ]

class EventListener(UserString):
    def __new__(cls, event_name, *_args, **_kwargs):  # -> Self:
        ...
    def __init__(
        self,
        event_name: str,
        has_trigger: bool = ...,
        config_data: Callable[..., dict[str, Any]] = ...,
        show_progress: Literal["full", "minimal", "hidden"] = ...,
        callback: Callable | None = ...,
        trigger_after: int | None = ...,
        trigger_only_on_success: bool = ...,
        trigger_only_on_failure: bool = ...,
        doc: str = ...,
        connection: Literal["sse", "stream"] = ...,
        event_specific_args: list[dict[str, str]] | None = ...,
    ) -> None: ...
    def set_doc(self, component: str):  # -> None:
        ...
    def copy(self):  # -> EventListener:
        ...

@document()
def on(
    triggers: Sequence[EventListenerCallable] | EventListenerCallable | None = ...,
    fn: Callable[..., Any] | None | Literal["decorator"] = ...,
    inputs: Component
    | BlockContext
    | Sequence[Component | BlockContext]
    | AbstractSet[Component | BlockContext]
    | None = ...,
    outputs: Component
    | BlockContext
    | Sequence[Component | BlockContext]
    | AbstractSet[Component | BlockContext]
    | None = ...,
    *,
    api_visibility: Literal["public", "private", "undocumented"] = ...,
    api_name: str | None = ...,
    api_description: str | None | Literal[False] = ...,
    scroll_to_output: bool = ...,
    show_progress: Literal["full", "minimal", "hidden"] = ...,
    show_progress_on: Component | Sequence[Component] | None = ...,
    queue: bool = ...,
    batch: bool = ...,
    max_batch_size: int = ...,
    preprocess: bool = ...,
    postprocess: bool = ...,
    cancels: dict[str, Any] | list[dict[str, Any]] | None = ...,
    trigger_mode: Literal["once", "multiple", "always_last"] | None = ...,
    js: str | Literal[True] | None = ...,
    concurrency_limit: int | None | Literal["default"] = ...,
    concurrency_id: str | None = ...,
    time_limit: int | None = ...,
    stream_every: float = ...,
    key: int | str | tuple[int | str, ...] | None = ...,
    validator: Callable | None = ...,
) -> Dependency: ...
@document()
def api(
    fn: Callable | Literal["decorator"] = ...,
    *,
    api_name: str | None = ...,
    api_description: str | None = ...,
    queue: bool = ...,
    batch: bool = ...,
    max_batch_size: int = ...,
    concurrency_limit: int | None | Literal["default"] = ...,
    concurrency_id: str | None = ...,
    api_visibility: Literal["public", "private", "undocumented"] = ...,
    time_limit: int | None = ...,
    stream_every: float = ...,
) -> Dependency: ...

class Events:
    change = ...
    input = ...
    click = ...
    double_click = ...
    submit = ...
    stop = ...
    edit = ...
    clear = ...
    play = ...
    pause = ...
    end = ...
    start_recording = ...
    pause_recording = ...
    stop_recording = ...
    focus = ...
    blur = ...
    upload = ...
    release = ...
    select = ...
    stream = ...
    like = ...
    example_select = ...
    option_select = ...
    load = ...
    key_up = ...
    apply = ...
    delete = ...
    tick = ...
    undo = ...
    retry = ...
    expand = ...
    collapse = ...
    download = ...
    copy = ...

all_events = ...
