from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from gradio.components import Button, Component, Timer
from gradio_client.documentation import document

"""gr.DuplicateButton() component"""
if TYPE_CHECKING: ...

@document()
class DuplicateButton(Button):
    is_template = ...
    def __init__(
        self,
        value: str = ...,
        *,
        every: Timer | float | None = ...,
        inputs: Component | Sequence[Component] | set[Component] | None = ...,
        variant: Literal["primary", "secondary", "stop", "huggingface"] = ...,
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
        _activate: bool = ...,
    ) -> None: ...
    def activate(self): ...

    if TYPE_CHECKING: ...
