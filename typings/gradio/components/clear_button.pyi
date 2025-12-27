from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from gradio.components import Button, Component, Timer
from gradio_client.documentation import document

"""Predefined buttons with bound events that can be included in a gr.Blocks for convenience."""

@document("add")
class ClearButton(Button):
    is_template = ...
    def __init__(
        self,
        components: None | Sequence[Component] | Component = ...,
        *,
        value: str = ...,
        every: Timer | float | None = ...,
        inputs: Component | Sequence[Component] | set[Component] | None = ...,
        variant: Literal["primary", "secondary", "stop"] = ...,
        size: Literal["sm", "md", "lg"] = ...,
        icon: str | Path | None = ...,
        link: str | None = ...,
        visible: bool | Literal["hidden"] = ...,
        interactive: bool = ...,
        elem_id: str | None = ...,
        elem_classes: list[str] | str | None = ...,
        render: bool = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
        scale: int | None = ...,
        min_width: int | None = ...,
        api_name: str | None = ...,
        api_visibility: Literal["public", "private", "undocumented"] = ...,
    ) -> None: ...
    def add(self, components: None | Component | Sequence[Component]) -> ClearButton: ...
    def preprocess(self, payload: str | None) -> str | None: ...
    def postprocess(self, value: str | None) -> str | None: ...
    def example_payload(self) -> Any: ...
    def example_value(self) -> Any: ...
