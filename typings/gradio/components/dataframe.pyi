from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import polars as pl
from gradio.components import Timer
from gradio.components.base import Component
from gradio.data_classes import GradioModel
from gradio.events import Dependency
from gradio.i18n import I18nData
from gradio_client.documentation import document
from pandas.io.formats.style import Styler

"""gr.Dataframe() component"""
if TYPE_CHECKING: ...

class DataframeData(GradioModel):
    headers: list[Any]
    data: list[list[Any]] | list[tuple[Any, ...]]
    metadata: dict[str, list[Any] | None] | None = ...

@document()
class Dataframe(Component):
    EVENTS = ...
    data_model = DataframeData
    def __init__(
        self,
        value: pd.DataFrame
        | Styler
        | np.ndarray
        | pl.DataFrame
        | list
        | list[list]
        | dict
        | str
        | Callable
        | None = ...,
        *,
        headers: list[str] | None = ...,
        row_count: int | None = ...,
        row_limits: tuple[int | None, int | None] | None = ...,
        col_count: None = ...,
        column_count: int | None = ...,
        column_limits: tuple[int | None, int | None] | None = ...,
        datatype: Literal["str", "number", "bool", "date", "markdown", "html", "image", "auto"]
        | Sequence[Literal["str", "number", "bool", "date", "markdown", "html"]] = ...,
        type: Literal["pandas", "numpy", "array", "polars"] = ...,
        latex_delimiters: list[dict[str, str | bool]] | None = ...,
        label: str | I18nData | None = ...,
        show_label: bool | None = ...,
        every: Timer | float | None = ...,
        inputs: Component | Sequence[Component] | set[Component] | None = ...,
        max_height: int | str = ...,
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
        buttons: list[Literal["fullscreen", "copy"]] | None = ...,
        show_row_numbers: bool = ...,
        max_chars: int | None = ...,
        show_search: Literal["none", "search", "filter"] = ...,
        pinned_columns: int | None = ...,
        static_columns: list[int] | None = ...,
    ) -> None: ...
    def preprocess(self, payload: DataframeData) -> pd.DataFrame | np.ndarray | pl.DataFrame | list[list]: ...
    @staticmethod
    def is_empty(
        value: pd.DataFrame | Styler | np.ndarray | pl.DataFrame | list | list[list] | dict | str | None,
    ) -> bool: ...
    def get_headers(
        self, value: pd.DataFrame | Styler | np.ndarray | pl.DataFrame | list | list[list] | dict | str | None
    ) -> list[str]: ...
    @staticmethod
    def get_cell_data(
        value: pd.DataFrame | Styler | np.ndarray | pl.DataFrame | list | list[list] | dict | str | None,
    ) -> list[list[Any]]: ...
    @staticmethod
    def get_metadata(
        value: pd.DataFrame | Styler | np.ndarray | pl.DataFrame | list | list[list] | dict | str | None,
    ) -> dict[str, list[list]] | None: ...
    def postprocess(
        self, value: pd.DataFrame | Styler | np.ndarray | pl.DataFrame | list | list[list] | dict | str | None
    ) -> DataframeData: ...
    def set_auto_datatype(self, value): ...
    def process_example(
        self, value: pd.DataFrame | Styler | np.ndarray | pl.DataFrame | list | list[list] | dict | str | None
    ): ...
    def example_payload(self) -> Any: ...
    def example_value(self) -> Any: ...

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
    def edit(
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
