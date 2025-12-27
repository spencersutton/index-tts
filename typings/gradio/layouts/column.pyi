from typing import Literal

from gradio.blocks import BlockContext
from gradio.component_meta import ComponentMeta
from gradio_client.documentation import document

@document()
class Column(BlockContext, metaclass=ComponentMeta):
    EVENTS = ...
    def __init__(
        self,
        *,
        scale: int = ...,
        min_width: int = ...,
        variant: Literal["default", "panel", "compact"] = ...,
        visible: bool | Literal["hidden"] = ...,
        elem_id: str | None = ...,
        elem_classes: list[str] | str | None = ...,
        render: bool = ...,
        show_progress: bool = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
    ) -> None: ...
