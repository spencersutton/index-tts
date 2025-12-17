from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import PIL.Image
from gradio import components
from gradio.components import Timer
from gradio.components.audio import WaveformOptions
from gradio.components.image_editor import Brush, Eraser, LayerOptions, WebcamOptions
from gradio.components.textbox import InputHTMLAttributes
from gradio.i18n import I18nData

if TYPE_CHECKING: ...

class TextArea(components.Textbox):
    is_template = ...
    def __init__(
        self,
        value: str | Callable | None = ...,
        *,
        lines: int = ...,
        max_lines: int = ...,
        placeholder: str | None = ...,
        label: str | I18nData | None = ...,
        info: str | I18nData | None = ...,
        every: Timer | float | None = ...,
        inputs: (components.Component | Sequence[components.Component] | set[components.Component] | None) = ...,
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
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
        type: Literal["text", "password", "email"] = ...,
        text_align: Literal["left", "right"] | None = ...,
        rtl: bool = ...,
        buttons: list[Literal["fullscreen", "copy"]] | None = ...,
        max_length: int | None = ...,
        submit_btn: str | bool | None = ...,
        stop_btn: str | bool | None = ...,
        html_attributes: InputHTMLAttributes | None = ...,
    ) -> None: ...

    if TYPE_CHECKING: ...

class Sketchpad(components.ImageEditor):
    is_template = ...
    def __init__(
        self,
        value: str | PIL.Image.Image | np.ndarray | None = ...,
        *,
        height: int | str | None = ...,
        width: int | str | None = ...,
        image_mode: Literal["1", "L", "P", "RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV", "I", "F"] = ...,
        sources: Iterable[Literal["upload", "webcam", "clipboard"]] = ...,
        type: Literal["numpy", "pil", "filepath"] = ...,
        label: str | I18nData | None = ...,
        every: Timer | float | None = ...,
        inputs: (components.Component | Sequence[components.Component] | set[components.Component] | None) = ...,
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
        webcam_options: WebcamOptions | None = ...,
        _selectable: bool = ...,
        transforms: Iterable[Literal["crop"]] = ...,
        eraser: Eraser | None = ...,
        brush: Brush | None = ...,
        format: str = ...,
        canvas_size: tuple[int, int] = ...,
        fixed_canvas: bool = ...,
        layers: LayerOptions | bool = ...,
    ) -> None: ...

    if TYPE_CHECKING: ...

class Paint(components.ImageEditor):
    is_template = ...
    def __init__(
        self,
        value: str | PIL.Image.Image | np.ndarray | None = ...,
        *,
        height: int | str | None = ...,
        width: int | str | None = ...,
        image_mode: Literal["1", "L", "P", "RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV", "I", "F"] = ...,
        sources: Iterable[Literal["upload", "webcam", "clipboard"]] = ...,
        type: Literal["numpy", "pil", "filepath"] = ...,
        label: str | I18nData | None = ...,
        every: Timer | float | None = ...,
        inputs: (components.Component | Sequence[components.Component] | set[components.Component] | None) = ...,
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
        webcam_options: WebcamOptions | None = ...,
        _selectable: bool = ...,
        transforms: Iterable[Literal["crop"]] = ...,
        eraser: Eraser | None = ...,
        brush: Brush | None = ...,
        format: str = ...,
        layers: LayerOptions | bool = ...,
        canvas_size: tuple[int, int] = ...,
        fixed_canvas: bool = ...,
        placeholder: str | None = ...,
    ) -> None: ...

    if TYPE_CHECKING: ...

class ImageMask(components.ImageEditor):
    is_template = ...
    def __init__(
        self,
        value: str | PIL.Image.Image | np.ndarray | None = ...,
        *,
        height: int | str | None = ...,
        width: int | str | None = ...,
        image_mode: Literal["1", "L", "P", "RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV", "I", "F"] = ...,
        sources: Iterable[Literal["upload", "webcam", "clipboard"]] = ...,
        type: Literal["numpy", "pil", "filepath"] = ...,
        label: str | I18nData | None = ...,
        every: Timer | float | None = ...,
        inputs: (components.Component | Sequence[components.Component] | set[components.Component] | None) = ...,
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
        transforms: Iterable[Literal["crop"]] = ...,
        eraser: Eraser | None = ...,
        brush: Brush | None = ...,
        format: str = ...,
        layers: LayerOptions | bool = ...,
        canvas_size: tuple[int, int] = ...,
        fixed_canvas: bool = ...,
        webcam_options: WebcamOptions | None = ...,
    ) -> None: ...

    if TYPE_CHECKING: ...

class PlayableVideo(components.Video):
    is_template = ...
    def __init__(
        self,
        value: (str | Path | tuple[str | Path, str | Path | None] | Callable | None) = ...,
        *,
        format: Literal["mp4"] = ...,
        sources: (list[Literal["upload", "webcam"]] | Literal["upload", "webcam"] | None) = ...,
        height: int | str | None = ...,
        width: int | str | None = ...,
        label: str | I18nData | None = ...,
        every: Timer | float | None = ...,
        inputs: (components.Component | Sequence[components.Component] | set[components.Component] | None) = ...,
        show_label: bool | None = ...,
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
        webcam_options: WebcamOptions | None = ...,
        include_audio: bool | None = ...,
        autoplay: bool = ...,
        buttons: list[Literal["download", "share"]] | None = ...,
        loop: bool = ...,
        streaming: bool = ...,
        watermark: str | Path | None = ...,
        subtitles: str | Path | None = ...,
    ) -> None: ...

    if TYPE_CHECKING: ...

class Microphone(components.Audio):
    is_template = ...
    def __init__(
        self,
        value: str | Path | tuple[int, np.ndarray] | Callable | None = ...,
        *,
        sources: (list[Literal["upload", "microphone"]] | Literal["upload", "microphone"] | None) = ...,
        type: Literal["numpy", "filepath"] = ...,
        label: str | I18nData | None = ...,
        every: Timer | float | None = ...,
        inputs: (components.Component | Sequence[components.Component] | set[components.Component] | None) = ...,
        show_label: bool | None = ...,
        container: bool = ...,
        scale: int | None = ...,
        min_width: int = ...,
        interactive: bool | None = ...,
        visible: bool | Literal["hidden"] = ...,
        streaming: bool = ...,
        elem_id: str | None = ...,
        elem_classes: list[str] | str | None = ...,
        render: bool = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
        format: Literal["wav", "mp3"] = ...,
        autoplay: bool = ...,
        buttons: list[Literal["download", "share"]] | None = ...,
        editable: bool = ...,
        waveform_options: WaveformOptions | dict | None = ...,
        loop: bool = ...,
        recording: bool = ...,
        subtitles: str | Path | None = ...,
    ) -> None: ...

    if TYPE_CHECKING: ...

class Files(components.File):
    is_template = ...
    def __init__(
        self,
        value: str | list[str] | Callable | None = ...,
        *,
        file_count: Literal["multiple"] = ...,
        file_types: list[str] | None = ...,
        type: Literal["filepath", "binary"] = ...,
        label: str | I18nData | None = ...,
        every: Timer | float | None = ...,
        inputs: (components.Component | Sequence[components.Component] | set[components.Component] | None) = ...,
        show_label: bool | None = ...,
        container: bool = ...,
        scale: int | None = ...,
        min_width: int = ...,
        height: float | None = ...,
        interactive: bool | None = ...,
        visible: bool | Literal["hidden"] = ...,
        elem_id: str | None = ...,
        elem_classes: list[str] | str | None = ...,
        render: bool = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
        allow_reordering: bool = ...,
    ) -> None: ...

    if TYPE_CHECKING: ...

class Numpy(components.Dataframe):
    is_template = ...
    def __init__(
        self,
        value: list[list[Any]] | Callable | None = ...,
        *,
        headers: list[str] | None = ...,
        row_count: int | tuple[int, str] = ...,
        row_limits: tuple[int | None, int | None] | None = ...,
        col_count: int | tuple[int, str] | None = ...,
        column_count: int | tuple[int, str] | None = ...,
        column_limits: tuple[int | None, int | None] | None = ...,
        datatype: (
            Literal["str", "number", "bool", "date", "markdown", "html"]
            | Sequence[Literal["str", "number", "bool", "date", "markdown", "html"]]
        ) = ...,
        type: Literal["numpy"] = ...,
        latex_delimiters: list[dict[str, str | bool]] | None = ...,
        label: str | I18nData | None = ...,
        show_label: bool | None = ...,
        every: Timer | float | None = ...,
        inputs: (components.Component | Sequence[components.Component] | set[components.Component] | None) = ...,
        max_height: int = ...,
        scale: int | None = ...,
        min_width: int = ...,
        interactive: bool | None = ...,
        visible: bool | Literal["hidden"] = ...,
        elem_id: str | None = ...,
        elem_classes: list[str] | str | None = ...,
        render: bool = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
        wrap: bool = ...,
        line_breaks: bool = ...,
        column_widths: list[str | int] | None = ...,
        show_row_numbers: bool = ...,
        show_search: Literal["none", "search", "filter"] = ...,
        static_columns: list[int] | None = ...,
        pinned_columns: int | None = ...,
        max_chars: int | None = ...,
        buttons: list[Literal["fullscreen", "copy"]] | None = ...,
    ) -> None: ...

    if TYPE_CHECKING: ...

class Matrix(components.Dataframe):
    is_template = ...
    def __init__(
        self,
        value: list[list[Any]] | Callable | None = ...,
        *,
        headers: list[str] | None = ...,
        row_count: int | tuple[int, str] = ...,
        row_limits: tuple[int | None, int | None] | None = ...,
        col_count: int | tuple[int, str] | None = ...,
        column_count: int | tuple[int, str] | None = ...,
        column_limits: tuple[int | None, int | None] | None = ...,
        datatype: (
            Literal["str", "number", "bool", "date", "markdown", "html"]
            | Sequence[Literal["str", "number", "bool", "date", "markdown", "html"]]
        ) = ...,
        type: Literal["array"] = ...,
        latex_delimiters: list[dict[str, str | bool]] | None = ...,
        label: str | I18nData | None = ...,
        show_label: bool | None = ...,
        every: Timer | float | None = ...,
        inputs: (components.Component | Sequence[components.Component] | set[components.Component] | None) = ...,
        max_height: int = ...,
        scale: int | None = ...,
        min_width: int = ...,
        interactive: bool | None = ...,
        visible: bool | Literal["hidden"] = ...,
        elem_id: str | None = ...,
        elem_classes: list[str] | str | None = ...,
        render: bool = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
        wrap: bool = ...,
        line_breaks: bool = ...,
        column_widths: list[str | int] | None = ...,
        show_row_numbers: bool = ...,
        show_search: Literal["none", "search", "filter"] = ...,
        pinned_columns: int | None = ...,
        max_chars: int | None = ...,
        buttons: list[Literal["fullscreen", "copy"]] | None = ...,
        static_columns: list[int] | None = ...,
    ) -> None: ...

    if TYPE_CHECKING: ...

class List(components.Dataframe):
    is_template = ...
    def __init__(
        self,
        value: list[list[Any]] | Callable | None = ...,
        *,
        headers: list[str] | None = ...,
        row_count: int | tuple[int, str] = ...,
        row_limits: tuple[int | None, int | None] | None = ...,
        col_count: Literal[1] = ...,
        column_count: Literal[1] | None = ...,
        column_limits: tuple[int | None, int | None] | None = ...,
        datatype: (
            Literal["str", "number", "bool", "date", "markdown", "html"]
            | Sequence[Literal["str", "number", "bool", "date", "markdown", "html"]]
        ) = ...,
        type: Literal["array"] = ...,
        latex_delimiters: list[dict[str, str | bool]] | None = ...,
        label: str | I18nData | None = ...,
        show_label: bool | None = ...,
        every: Timer | float | None = ...,
        inputs: (components.Component | Sequence[components.Component] | set[components.Component] | None) = ...,
        max_height: int = ...,
        scale: int | None = ...,
        min_width: int = ...,
        interactive: bool | None = ...,
        visible: bool | Literal["hidden"] = ...,
        elem_id: str | None = ...,
        elem_classes: list[str] | str | None = ...,
        render: bool = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
        wrap: bool = ...,
        line_breaks: bool = ...,
        column_widths: list[str | int] | None = ...,
        show_row_numbers: bool = ...,
        show_search: Literal["none", "search", "filter"] = ...,
        pinned_columns: int | None = ...,
        max_chars: int | None = ...,
        buttons: list[Literal["fullscreen", "copy"]] | None = ...,
        static_columns: list[int] | None = ...,
    ) -> None: ...

    if TYPE_CHECKING: ...

Mic = Microphone
