from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from gradio.components import Timer
from gradio.components.base import Component
from gradio.data_classes import GradioModel
from gradio.events import Dependency
from gradio.i18n import I18nData
from gradio_client.documentation import document

"""gr.Plot() component."""
if TYPE_CHECKING: ...

class PlotData(GradioModel):
    type: Literal["altair", "bokeh", "plotly", "matplotlib"]
    plot: str

class AltairPlotData(PlotData):
    chart: Literal["bar", "line", "scatter"]
    type: Literal["altair"] = ...

@document()
class Plot(Component):
    data_model = PlotData
    EVENTS = ...
    def __init__(
        self,
        value: Any | None = ...,
        *,
        format: str = ...,
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
    ) -> None: ...
    def get_config(self): ...
    def preprocess(self, payload: PlotData | None) -> PlotData | None: ...
    def example_payload(self) -> Any: ...
    def example_value(self) -> Any: ...
    def postprocess(self, value: Any) -> PlotData | None: ...

    if TYPE_CHECKING: ...
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

class AltairPlot:
    @staticmethod
    def create_legend(position, title): ...
    @staticmethod
    def create_scale(limit): ...
