from typing import Literal

from gradio.blocks import BlockContext
from gradio.component_meta import ComponentMeta
from gradio_client.documentation import document

@document()
class Row(BlockContext, metaclass=ComponentMeta):
    EVENTS = ...
    def __init__(
        self,
        *,
        variant: Literal["default", "panel", "compact"] = ...,
        visible: bool | Literal["hidden"] = ...,
        elem_id: str | None = ...,
        elem_classes: list[str] | str | None = ...,
        scale: int | None = ...,
        render: bool = ...,
        height: int | str | None = ...,
        max_height: int | str | None = ...,
        min_height: int | str | None = ...,
        equal_height: bool = ...,
        show_progress: bool = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
    ) -> None: ...
    @staticmethod
    def update(visible: bool | None = ...): ...
