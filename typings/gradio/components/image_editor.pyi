import dataclasses
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import PIL.Image
from gradio.components import Timer
from gradio.components.base import Component, server
from gradio.data_classes import FileData, GradioModel
from gradio.events import Dependency
from gradio.i18n import I18nData
from gradio_client.documentation import document
from typing_extensions import TypedDict

r"""gr.ImageEditor() component."""

type ImageType = np.ndarray | PIL.Image.Image | str

class EditorValue(TypedDict):
    background: ImageType | None
    layers: list[ImageType]
    composite: ImageType | None

class EditorExampleValue(TypedDict):
    background: str | None
    layers: list[str | None] | None
    composite: str | None

class EditorData(GradioModel):
    background: FileData | None = ...
    layers: list[FileData] = ...
    composite: FileData | None = ...
    id: str | None = ...

class EditorDataBlobs(GradioModel):
    background: bytes | None
    layers: list[bytes | None]
    composite: bytes | None

class BlobData(TypedDict):
    type: str
    index: int | None
    file: bytes
    id: str

class AcceptBlobs(GradioModel):
    data: BlobData
    files: list[tuple[str, bytes]]

@document()
@dataclasses.dataclass
class Eraser:
    default_size: int | Literal["auto"] = ...

@document()
@dataclasses.dataclass
class Brush(Eraser):
    colors: list[str | tuple[str, float]] | str | tuple[str, float] | None = ...
    default_color: str | tuple[str, float] | None = ...
    color_mode: Literal["fixed", "defaults"] = ...
    def __post_init__(self): ...

@document()
@dataclasses.dataclass
class LayerOptions:
    allow_additional_layers: bool = ...
    layers: list[str] | None = ...
    disabled: bool = ...
    def __post_init__(self): ...

@document()
@dataclasses.dataclass
class WebcamOptions:
    mirror: bool = ...
    constraints: dict[str, Any] | None = ...

@document()
@dataclasses.dataclass
class WatermarkOptions:
    watermark: str | Path | PIL.Image.Image | np.ndarray | None = ...
    position: tuple[int, int] | Literal["top-left", "top-right", "bottom-left", "bottom-right"] = ...
    def __post_init__(self): ...

@document()
class ImageEditor(Component):
    EVENTS = ...
    data_model = EditorData
    def __init__(
        self,
        value: EditorValue | ImageType | None = ...,
        *,
        height: int | str | None = ...,
        width: int | str | None = ...,
        image_mode: Literal["1", "L", "P", "RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV", "I", "F"] = ...,
        sources: (
            Iterable[Literal["upload", "webcam", "clipboard"]] | Literal["upload", "webcam", "clipboard"] | None
        ) = ...,
        type: Literal["numpy", "pil", "filepath"] = ...,
        label: str | I18nData | None = ...,
        every: Timer | float | None = ...,
        inputs: Component | Sequence[Component] | set[Component] | None = ...,
        show_label: bool | None = ...,
        buttons: list[Literal["download", "share", "fullscreen"]] | None = ...,
        container: bool = ...,
        scale: int | None = ...,
        min_width: int = ...,
        interactive: bool | None = ...,
        visible: bool | Literal["hidden"] = ...,
        elem_id: str | None = ...,
        elem_classes: list[str] | str | None = ...,
        render: bool = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
        placeholder: str | None = ...,
        _selectable: bool = ...,
        transforms: Iterable[Literal["crop", "resize"]] | None = ...,
        eraser: Eraser | None | Literal[False] = ...,
        brush: Brush | None | Literal[False] = ...,
        format: str = ...,
        layers: bool | LayerOptions = ...,
        canvas_size: tuple[int, int] = ...,
        fixed_canvas: bool = ...,
        webcam_options: WebcamOptions | None = ...,
    ) -> None: ...
    def convert_and_format_image(self, file: FileData | None | bytes) -> np.ndarray | PIL.Image.Image | str | None: ...
    def preprocess(self, payload: EditorData | None) -> EditorValue | None: ...
    def postprocess(self, value: EditorValue | ImageType | None) -> EditorData | None: ...
    def example_payload(self) -> Any: ...
    def example_value(self) -> Any: ...
    @server
    def accept_blobs(self, data: AcceptBlobs): ...
    def clear(
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
    def apply(
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
