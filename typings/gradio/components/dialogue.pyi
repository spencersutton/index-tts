from collections.abc import Callable
from typing import Literal

from gradio.components.base import Component, server
from gradio.data_classes import GradioModel, GradioRootModel
from gradio.events import Dependency
from gradio_client.documentation import document

class DialogueLine(GradioModel):
    speaker: str
    text: str

class DialogueModel(GradioRootModel):
    root: list[DialogueLine] | str

@document()
class Dialogue(Component):
    EVENTS = ...
    data_model = DialogueModel
    def __init__(
        self,
        value: list[dict[str, str]] | Callable | None = ...,
        *,
        type: Literal["list", "text"] = ...,
        speakers: list[str] | None = ...,
        formatter: Callable | None = ...,
        unformatter: Callable | None = ...,
        tags: list[str] | None = ...,
        separator: str = ...,
        color_map: dict[str, str] | None = ...,
        label: str | None = ...,
        info: str | None = ...,
        placeholder: str | None = ...,
        show_label: bool | None = ...,
        container: bool = ...,
        scale: int | None = ...,
        min_width: int = ...,
        interactive: bool | None = ...,
        visible: bool | Literal["hidden"] = ...,
        elem_id: str | None = ...,
        autofocus: bool = ...,
        autoscroll: bool = ...,
        elem_classes: list[str] | str | None = ...,
        render: bool = ...,
        key: int | str | None = ...,
        max_lines: int | None = ...,
        buttons: list[Literal["copy"]] | None = ...,
        submit_btn: str | bool | None = ...,
        ui_mode: Literal["dialogue", "text", "both"] = ...,
    ) -> None: ...
    def preprocess(self, payload: DialogueModel) -> str | list[dict[str, str]]: ...
    @staticmethod
    def default_formatter(speaker: str, text: str) -> str: ...
    @staticmethod
    def default_unformatter(line: str, default_speaker: str) -> dict[str, str]: ...
    @server
    async def format(self, value: list[dict] | str): ...
    @server
    async def unformat(self, payload: dict): ...
    def postprocess(self, value: list[dict[str, str]] | str | None) -> DialogueModel | None: ...
    def as_example(self, value): ...
    def example_payload(self): ...
    def example_value(self): ...
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
    def input(
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
    def submit(
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
