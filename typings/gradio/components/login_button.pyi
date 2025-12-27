from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from gradio.components import Button, Component, Timer
from gradio_client.documentation import document

"""Predefined button to sign in with Hugging Face in a Gradio Space."""

@document()
class LoginButton(Button):
    is_template = ...
    def __init__(
        self,
        value: str = ...,
        logout_value: str = ...,
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
    ) -> None: ...
    def activate(self): ...

_js_handle_redirect = ...
