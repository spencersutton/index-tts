from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import PIL.Image
from gradio.components import Timer
from gradio.components.base import Component
from gradio.data_classes import FileData, GradioModel
from gradio.events import Dependency
from gradio.i18n import I18nData
from gradio_client.documentation import document

"""gr.AnnotatedImage() component."""

class Annotation(GradioModel):
    image: FileData
    label: str

class AnnotatedImageData(GradioModel):
    image: FileData
    annotations: list[Annotation]

@document()
class AnnotatedImage(Component):
    EVENTS = ...
    data_model = AnnotatedImageData
    def __init__(
        self,
        value: (
            tuple[
                np.ndarray | PIL.Image.Image | str,
                list[tuple[np.ndarray | tuple[int, int, int, int], str]],
            ]
            | None
        ) = ...,
        *,
        format: str = ...,
        show_legend: bool = ...,
        height: int | str | None = ...,
        width: int | str | None = ...,
        color_map: dict[str, str] | None = ...,
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
        buttons: list[Literal["fullscreen"]] | None = ...,
    ) -> None: ...
    def preprocess(self, payload: AnnotatedImageData | None) -> tuple[str, list[tuple[str, str]]] | None: ...
    def postprocess(
        self,
        value: (
            tuple[
                np.ndarray | PIL.Image.Image | str,
                Sequence[tuple[np.ndarray | tuple[int, int, int, int], str]],
            ]
            | None
        ),
    ) -> AnnotatedImageData | None: ...
    def example_payload(self) -> Any: ...
    def example_value(self) -> Any: ...
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
