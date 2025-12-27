from typing import Literal

from gradio.blocks import BlockContext
from gradio.component_meta import ComponentMeta
from gradio_client.documentation import document

@document()
class Group(BlockContext, metaclass=ComponentMeta):
    EVENTS = ...
    def __init__(
        self,
        *,
        visible: bool | Literal["hidden"] = ...,
        elem_id: str | None = ...,
        elem_classes: list[str] | str | None = ...,
        render: bool = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
    ) -> None: ...
