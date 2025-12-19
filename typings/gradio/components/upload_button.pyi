from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

from gradio.components import Timer
from gradio.components.base import Component
from gradio.data_classes import FileData, ListFiles
from gradio.events import Dependency
from gradio.i18n import I18nData
from gradio_client.documentation import document

"""gr.UploadButton() component."""
if TYPE_CHECKING: ...

@document()
class UploadButton(Component):
    EVENTS = ...
    def __init__(
        self,
        label: str = ...,
        value: str | I18nData | list[str] | Callable | None = ...,
        *,
        every: Timer | float | None = ...,
        inputs: Component | Sequence[Component] | set[Component] | None = ...,
        variant: Literal["primary", "secondary", "stop"] = ...,
        visible: bool | Literal["hidden"] = ...,
        size: Literal["sm", "md", "lg"] = ...,
        icon: str | None = ...,
        scale: int | None = ...,
        min_width: int | None = ...,
        interactive: bool = ...,
        elem_id: str | None = ...,
        elem_classes: list[str] | str | None = ...,
        render: bool = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
        type: Literal["filepath", "binary"] = ...,
        file_count: Literal["single", "multiple", "directory"] = ...,
        file_types: list[str] | None = ...,
    ) -> None: ...
    def api_info(self) -> dict[str, list[str]]: ...
    def example_payload(self) -> Any: ...
    def example_value(self) -> Any: ...
    def preprocess(self, payload: ListFiles | FileData | None) -> bytes | str | list[bytes] | list[str] | None: ...
    def postprocess(self, value: str | list[str] | None) -> ListFiles | FileData | None: ...
    @property
    def skip_api(self): ...

    if TYPE_CHECKING: ...
    def click(
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
    def upload(
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
