from collections.abc import Callable, Sequence
from typing import Any, Literal

from diffusers import DiffusionPipeline
from gradio.blocks import Blocks
from gradio.components import Button, ClearButton, Component, DeepLinkButton, DuplicateButton
from gradio.events import Dependency
from gradio.flagging import FlaggingCallback
from gradio.i18n import I18nData
from gradio.layouts import Accordion, Column
from gradio_client.documentation import document
from transformers.pipelines.base import Pipeline

"""
This file defines two useful high-level abstractions to build Gradio apps: Interface and TabbedInterface.
"""

@document("launch", "load", "from_pipeline", "integrate", "queue")
class Interface(Blocks):
    @classmethod
    def from_pipeline(cls, pipeline: Pipeline | DiffusionPipeline, **kwargs) -> Interface: ...
    def __init__(
        self,
        fn: Callable,
        inputs: str | Component | Sequence[str | Component] | None,
        outputs: str | Component | Sequence[str | Component] | None,
        examples: list[Any] | list[list[Any]] | str | None = ...,
        *,
        cache_examples: bool | None = ...,
        cache_mode: Literal["eager", "lazy"] | None = ...,
        examples_per_page: int = ...,
        example_labels: list[str] | None = ...,
        preload_example: int | Literal[False] = ...,
        live: bool = ...,
        title: str | I18nData | None = ...,
        description: str | None = ...,
        article: str | None = ...,
        flagging_mode: Literal["never", "auto", "manual"] | None = ...,
        flagging_options: list[str] | list[tuple[str, str]] | None = ...,
        flagging_dir: str = ...,
        flagging_callback: FlaggingCallback | None = ...,
        analytics_enabled: bool | None = ...,
        batch: bool = ...,
        max_batch_size: int = ...,
        api_visibility: Literal["public", "private", "undocumented"] = ...,
        api_name: str | None = ...,
        api_description: str | None | Literal[False] = ...,
        _api_mode: bool = ...,
        allow_duplication: bool = ...,
        concurrency_limit: int | None | Literal["default"] = ...,
        additional_inputs: str | Component | Sequence[str | Component] | None = ...,
        additional_inputs_accordion: str | Accordion | None = ...,
        submit_btn: str | Button = ...,
        stop_btn: str | Button = ...,
        clear_btn: str | Button | None = ...,
        delete_cache: tuple[int, int] | None = ...,
        show_progress: Literal["full", "minimal", "hidden"] = ...,
        fill_width: bool = ...,
        time_limit: int | None = ...,
        stream_every: float = ...,
        deep_link: str | DeepLinkButton | bool | None = ...,
        validator: Callable | None = ...,
        **kwargs,
    ) -> None: ...
    def render_title_description(self) -> None: ...
    def render_flag_btns(self) -> list[Button]: ...
    def render_input_column(
        self,
    ) -> tuple[
        Button | None,
        ClearButton | None,
        Button | None,
        list[Button] | None,
        Column,
    ]: ...
    def render_output_column(
        self, _submit_btn_in: Button | None
    ) -> tuple[
        Button | None,
        ClearButton | None,
        DuplicateButton | None,
        Button | None,
        list | None,
    ]: ...
    def render_article(self):  # -> None:
        ...
    def attach_submit_events(self, _submit_btn: Button | None, _stop_btn: Button | None) -> Dependency: ...
    def attach_clear_events(self, _clear_btn: ClearButton, input_component_column: Column | None):  # -> None:
        ...
    def attach_flagging_events(
        self, flag_btns: list[Button] | None, _clear_btn: ClearButton, _submit_event: Dependency
    ):  # -> None:
        ...
    def render_examples(self):  # -> None:
        ...

@document()
class TabbedInterface(Blocks):
    def __init__(
        self,
        interface_list: Sequence[Blocks],
        tab_names: list[str] | None = ...,
        title: str | None = ...,
        analytics_enabled: bool | None = ...,
    ) -> None: ...

def close_all(verbose: bool = ...) -> None: ...
