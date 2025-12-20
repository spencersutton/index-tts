from collections.abc import Callable, Sequence, Set as AbstractSet
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from gradio.components import Timer
from gradio.components.base import Component
from gradio.data_classes import GradioModel
from gradio.events import Dependency
from gradio.i18n import I18nData
from gradio_client.documentation import document

if TYPE_CHECKING: ...

class PlotData(GradioModel):
    columns: list[str]
    data: list[list[Any]]
    datatypes: dict[str, Literal["quantitative", "nominal", "temporal"]]
    mark: str

class NativePlot(Component):
    EVENTS = ...
    def __init__(
        self,
        value: pd.DataFrame | Callable | None = ...,
        x: str | None = ...,
        y: str | None = ...,
        *,
        color: str | None = ...,
        title: str | None = ...,
        x_title: str | None = ...,
        y_title: str | None = ...,
        color_title: str | None = ...,
        x_bin: str | float | None = ...,
        y_aggregate: Literal["sum", "mean", "median", "min", "max", "count"] | None = ...,
        color_map: dict[str, str] | None = ...,
        colors_in_legend: list[str] | None = ...,
        x_lim: list[float | None] | None = ...,
        y_lim: list[float | None] = ...,
        x_label_angle: float = ...,
        y_label_angle: float = ...,
        x_axis_labels_visible: bool | Literal["hidden"] = ...,
        caption: str | I18nData | None = ...,
        sort: Literal["x", "y", "-x", "-y"] | list[str] | None = ...,
        tooltip: Literal["axis", "none", "all"] | list[str] = ...,
        height: int | None = ...,
        label: str | I18nData | None = ...,
        show_label: bool | None = ...,
        container: bool = ...,
        scale: int | None = ...,
        min_width: int = ...,
        every: Timer | float | None = ...,
        inputs: Component | Sequence[Component] | AbstractSet[Component] | None = ...,
        visible: bool | Literal["hidden"] = ...,
        elem_id: str | None = ...,
        elem_classes: list[str] | str | None = ...,
        render: bool = ...,
        buttons: list[Literal["fullscreen", "export"]] | None = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
    ) -> None: ...
    def get_block_name(self) -> str: ...
    def get_mark(self) -> str: ...
    def preprocess(self, payload: PlotData | None) -> PlotData | None: ...
    def postprocess(self, value: pd.DataFrame | dict | None) -> PlotData | None: ...
    def example_payload(self) -> Any: ...
    def example_value(self) -> Any: ...
    def api_info(self) -> dict[str, Any]: ...

    if TYPE_CHECKING: ...
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
    def double_click(
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

@document()
class BarPlot(NativePlot):
    def get_block_name(self) -> str: ...
    def get_mark(self) -> str: ...

    if TYPE_CHECKING: ...

@document()
class LinePlot(NativePlot):
    def get_block_name(self) -> str: ...
    def get_mark(self) -> str: ...

    if TYPE_CHECKING: ...

@document()
class ScatterPlot(NativePlot):
    def get_block_name(self) -> str: ...
    def get_mark(self) -> str: ...

    if TYPE_CHECKING: ...
