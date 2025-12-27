from gradio.blocks import Block, BlockContext
from gradio.component_meta import ComponentMeta

class Form(BlockContext, metaclass=ComponentMeta):
    EVENTS = ...
    def __init__(
        self,
        *,
        scale: int = ...,
        min_width: int = ...,
        render: bool = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
    ) -> None: ...
    def add_child(self, child: Block): ...
