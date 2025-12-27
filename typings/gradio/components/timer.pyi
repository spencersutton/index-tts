from gradio.components.base import Component
from gradio.events import Dependency
from gradio_client.documentation import document

"""gr.Timer() component."""

@document()
class Timer(Component):
    EVENTS = ...
    def __init__(self, value: float = ..., *, active: bool = ..., render: bool = ...) -> None: ...
    def preprocess(self, payload: float | None) -> float | None: ...
    def postprocess(self, value: float | None) -> float | None: ...
    def api_info(self) -> dict: ...
    def example_payload(self): ...
    def example_value(self): ...
    def breaks_grouping(self) -> bool: ...
    def tick(
        self,
        fn: Callable[..., Any] | None = ...,
        inputs: Block | Sequence[Block] | set[Block] | None = ...,
        outputs: Block | Sequence[Block] | None = ...,
        api_name: str | None = ...,
        scroll_to_output: bool = ...,
        show_progress: Literal[full, minimal, hidden] = ...,
        show_progress_on: Component | Sequence[Component] | None = ...,
        queue: bool | None = ...,
        batch: bool = ...,
        max_batch_size: int = ...,
        preprocess: bool = ...,
        postprocess: bool = ...,
        cancels: dict[str, Any] | list[dict[str, Any]] | None = ...,
        every: Timer | float | None = ...,
        trigger_mode: Literal[once, multiple, always_last] | None = ...,
        js: str | Literal[True] | None = ...,
        concurrency_limit: int | None | Literal[default] = ...,
        concurrency_id: str | None = ...,
        api_visibility: Literal[public, private, undocumented] = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        api_description: str | None | Literal[False] = ...,
        validator: Callable[..., Any] | None = ...,
    ) -> Dependency: ...
