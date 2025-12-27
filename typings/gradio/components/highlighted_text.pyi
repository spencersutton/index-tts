from collections.abc import Callable, Sequence
from typing import Any, Literal

from gradio.components import Timer
from gradio.components.base import Component
from gradio.data_classes import GradioModel, GradioRootModel
from gradio.events import Dependency
from gradio.i18n import I18nData
from gradio_client.documentation import document

"""gr.HighlightedText() component."""

class HighlightedToken(GradioModel):
    token: str
    class_or_confidence: str | float | None = ...

class HighlightedTextData(GradioRootModel):
    root: list[HighlightedToken]

@document()
class HighlightedText(Component):
    data_model = HighlightedTextData
    EVENTS = ...
    def __init__(
        self,
        value: list[tuple[str, str | float | None]] | dict | Callable | None = ...,
        *,
        color_map: dict[str, str] | None = ...,
        show_legend: bool = ...,
        show_inline_category: bool = ...,
        combine_adjacent: bool = ...,
        adjacent_separator: str = ...,
        label: str | I18nData | None = ...,
        every: Timer | float | None = ...,
        inputs: Component | Sequence[Component] | set[Component] | None = ...,
        show_label: bool | None = ...,
        container: bool = ...,
        scale: int | None = ...,
        min_width: int = ...,
        visible: bool | Literal["hidden"] = ...,
        elem_id: str | None = ...,
        elem_classes: list[str] | str | None = ...,
        render: bool = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
        interactive: bool | None = ...,
        rtl: bool = ...,
    ) -> None: ...
    def example_payload(self) -> Any: ...
    def example_value(self) -> Any: ...
    def preprocess(self, payload: HighlightedTextData | None) -> list[tuple[str, str | float | None]] | None: ...
    def postprocess(self, value: list[tuple[str, str | float | None]] | dict | None) -> HighlightedTextData | None: ...
    def change(
        self,
        fn: Callable[..., Any] | None = ...,
        inputs: Block | Sequence[Block] | set[Block] | None = ...,
        outputs: Block | Sequence[Block] | None = ...,
        api_name: str | None = ...,
        scroll_to_output: bool = ...,
        show_progress: Literal["full", "minimal", "hidden"] = ...,
        show_progress_on: Component | Sequence[Component] | None = ...,
        queue: bool | None = ...,
        batch: bool = ...,
        max_batch_size: int = ...,
        preprocess: bool = ...,
        postprocess: bool = ...,
        cancels: dict[str, Any] | list[dict[str, Any]] | None = ...,
        every: Timer | float | None = ...,
        trigger_mode: Literal["once", "multiple", "always_last"] | None = ...,
        js: str | Literal[True] | None = ...,
        concurrency_limit: int | None | Literal["default"] = ...,
        concurrency_id: str | None = ...,
        api_visibility: Literal["public", "private", "undocumented"] = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        api_description: str | None | Literal[False] = ...,
        validator: Callable[..., Any] | None = ...,
    ) -> Dependency: ...
    def select(
        self,
        fn: Callable[..., Any] | None = ...,
        inputs: Block | Sequence[Block] | set[Block] | None = ...,
        outputs: Block | Sequence[Block] | None = ...,
        api_name: str | None = ...,
        scroll_to_output: bool = ...,
        show_progress: Literal["full", "minimal", "hidden"] = ...,
        show_progress_on: Component | Sequence[Component] | None = ...,
        queue: bool | None = ...,
        batch: bool = ...,
        max_batch_size: int = ...,
        preprocess: bool = ...,
        postprocess: bool = ...,
        cancels: dict[str, Any] | list[dict[str, Any]] | None = ...,
        every: Timer | float | None = ...,
        trigger_mode: Literal["once", "multiple", "always_last"] | None = ...,
        js: str | Literal[True] | None = ...,
        concurrency_limit: int | None | Literal["default"] = ...,
        concurrency_id: str | None = ...,
        api_visibility: Literal["public", "private", "undocumented"] = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        api_description: str | None | Literal[False] = ...,
        validator: Callable[..., Any] | None = ...,
    ) -> Dependency: ...
