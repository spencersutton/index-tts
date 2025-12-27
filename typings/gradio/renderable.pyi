from collections.abc import Callable, Sequence
from typing import Literal

from gradio.blocks import Block
from gradio.components import Component
from gradio.events import EventListenerCallable
from gradio_client.documentation import document

class Renderable:
    def __init__(
        self,
        fn: Callable,
        inputs: Sequence[Component],
        triggers: list[tuple[Block | None, str]],
        concurrency_limit: int | None | Literal["default"],
        concurrency_id: str | None,
        trigger_mode: Literal["once", "multiple", "always_last"] | None,
        queue: bool,
        show_progress: Literal["full", "minimal", "hidden"],
    ) -> None: ...
    def apply(self, *args, **kwargs):  # -> None:
        ...

@document()
def render(
    inputs: Sequence[Component] | Component | None = ...,
    triggers: Sequence[EventListenerCallable] | EventListenerCallable | None = ...,
    *,
    queue: bool = ...,
    trigger_mode: Literal["once", "multiple", "always_last"] | None = ...,
    concurrency_limit: int | None | Literal["default"] = ...,
    concurrency_id: str | None = ...,
    show_progress: Literal["full", "minimal", "hidden"] = ...,
):  # -> Callable[..., Any]:

    ...
