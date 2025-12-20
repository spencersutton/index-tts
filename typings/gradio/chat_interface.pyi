from collections.abc import Callable, Generator
from typing import Literal

from gradio.blocks import Blocks
from gradio.components import Chatbot, Component, MultimodalTextbox, Textbox
from gradio.components.chatbot import MessageDict, NormalizedMessageDict
from gradio.components.multimodal_textbox import MultimodalPostprocess, MultimodalValue
from gradio.events import SelectData
from gradio.i18n import I18nData
from gradio.layouts import Accordion
from gradio_client.documentation import document

"""
This file defines a useful high-level abstraction to build Gradio chatbots: ChatInterface.
"""

@document()
class ChatInterface(Blocks):
    def __init__(
        self,
        fn: Callable,
        *,
        multimodal: bool = ...,
        chatbot: Chatbot | None = ...,
        textbox: Textbox | MultimodalTextbox | None = ...,
        additional_inputs: str | Component | list[str | Component] | None = ...,
        additional_inputs_accordion: str | Accordion | None = ...,
        additional_outputs: Component | list[Component] | None = ...,
        editable: bool = ...,
        examples: list[str] | list[MultimodalValue] | list[list] | None = ...,
        example_labels: list[str] | None = ...,
        example_icons: list[str] | None = ...,
        run_examples_on_click: bool = ...,
        cache_examples: bool | None = ...,
        cache_mode: Literal["eager", "lazy"] | None = ...,
        title: str | I18nData | None = ...,
        description: str | None = ...,
        flagging_mode: Literal["never", "manual"] | None = ...,
        flagging_options: list[str] | tuple[str, ...] | None = ...,
        flagging_dir: str = ...,
        analytics_enabled: bool | None = ...,
        autofocus: bool = ...,
        autoscroll: bool = ...,
        submit_btn: str | bool | None = ...,
        stop_btn: str | bool | None = ...,
        concurrency_limit: int | None | Literal["default"] = ...,
        delete_cache: tuple[int, int] | None = ...,
        show_progress: Literal["full", "minimal", "hidden"] = ...,
        fill_height: bool = ...,
        fill_width: bool = ...,
        api_name: str | None = ...,
        api_description: str | None | Literal[False] = ...,
        api_visibility: Literal["public", "private", "undocumented"] = ...,
        save_history: bool = ...,
        validator: Callable | None = ...,
    ) -> None: ...
    @staticmethod
    def serialize_components(conversation: list[NormalizedMessageDict]) -> list[NormalizedMessageDict]: ...
    def option_clicked(
        self, history: list[MessageDict], option: SelectData
    ) -> tuple[list[MessageDict], str | MultimodalPostprocess]: ...
    def example_populated(self, example: SelectData):  # -> Any:
        ...
    def example_clicked(
        self, example: SelectData
    ) -> Generator[tuple[list[MessageDict], str | MultimodalPostprocess]]: ...
