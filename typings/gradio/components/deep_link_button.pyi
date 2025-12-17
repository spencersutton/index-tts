from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from gradio.components import Timer
from gradio.components.base import Component
from gradio.components.button import Button
from gradio_client.documentation import document

"""Predefined button to copy a shareable link to the current Gradio Space."""
if TYPE_CHECKING: ...

@document()
class DeepLinkButton(Button):
    is_template = ...
    n_created = ...
    def __init__(
        self,
        value: str = ...,
        copied_value: str = ...,
        *,
        inputs: Component | Sequence[Component] | set[Component] | None = ...,
        variant: Literal["primary", "secondary"] = ...,
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
        every: Timer | float | None = ...,
    ) -> None: ...
    def activate(self): ...
    def get_share_link(self, value: str = ..., copied_value: str = ...): ...

    if TYPE_CHECKING: ...
