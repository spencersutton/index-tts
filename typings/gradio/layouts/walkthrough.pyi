from gradio.blocks import BlockContext
from gradio.component_meta import ComponentMeta
from gradio.events import Dependency
from gradio.i18n import I18nData
from gradio_client.documentation import document

@document()
class Walkthrough(BlockContext, metaclass=ComponentMeta):
    EVENTS = ...
    def __init__(
        self,
        *,
        selected: int | None = ...,
        visible: bool = ...,
        elem_id: str | None = ...,
        elem_classes: list[str] | str | None = ...,
        render: bool = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
    ) -> None: ...
    def get_block_name(self): ...

    if TYPE_CHECKING: ...
    def change(
        self,
        fn: Callable[..., Any] | None = ...,
        inputs: Block | Sequence[Block] | set[Block] | None = ...,
        outputs: Block | Sequence[Block] | None = ...,
        api_name: str | None = ...,
        scroll_to_output: bool = ...,
        show_progress: Literal[full, minimal, hidden] = ...,
        show_progress_on: Component | Sequence[Component] | None = ...,
        queue: bool | None = ...,
        batch: bool = ...,
        max_batch_size: int = ...,
        preprocess: bool = ...,
        postprocess: bool = ...,
        cancels: dict[str, Any] | list[dict[str, Any]] | None = ...,
        every: Timer | float | None = ...,
        trigger_mode: Literal[once, multiple, always_last] | None = ...,
        js: str | Literal[True] | None = ...,
        concurrency_limit: int | None | Literal[default] = ...,
        concurrency_id: str | None = ...,
        api_visibility: Literal[public, private, undocumented] = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        api_description: str | None | Literal[False] = ...,
        validator: Callable[..., Any] | None = ...,
    ) -> Dependency: ...
    def select(
        self,
        fn: Callable[..., Any] | None = ...,
        inputs: Block | Sequence[Block] | set[Block] | None = ...,
        outputs: Block | Sequence[Block] | None = ...,
        api_name: str | None = ...,
        scroll_to_output: bool = ...,
        show_progress: Literal[full, minimal, hidden] = ...,
        show_progress_on: Component | Sequence[Component] | None = ...,
        queue: bool | None = ...,
        batch: bool = ...,
        max_batch_size: int = ...,
        preprocess: bool = ...,
        postprocess: bool = ...,
        cancels: dict[str, Any] | list[dict[str, Any]] | None = ...,
        every: Timer | float | None = ...,
        trigger_mode: Literal[once, multiple, always_last] | None = ...,
        js: str | Literal[True] | None = ...,
        concurrency_limit: int | None | Literal[default] = ...,
        concurrency_id: str | None = ...,
        api_visibility: Literal[public, private, undocumented] = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        api_description: str | None | Literal[False] = ...,
        validator: Callable[..., Any] | None = ...,
    ) -> Dependency: ...

@document()
class Step(BlockContext, metaclass=ComponentMeta):
    EVENTS = ...
    def __init__(
        self,
        label: str | I18nData | None = ...,
        visible: bool = ...,
        interactive: bool = ...,
        *,
        id: int | None = ...,
        elem_id: str | None = ...,
        elem_classes: list[str] | str | None = ...,
        scale: int | None = ...,
        render: bool = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
    ) -> None: ...
    def get_expected_parent(self) -> type[Walkthrough]: ...
    def get_block_name(self): ...

    if TYPE_CHECKING: ...
    def select(
        self,
        fn: Callable[..., Any] | None = ...,
        inputs: Block | Sequence[Block] | set[Block] | None = ...,
        outputs: Block | Sequence[Block] | None = ...,
        api_name: str | None = ...,
        scroll_to_output: bool = ...,
        show_progress: Literal[full, minimal, hidden] = ...,
        show_progress_on: Component | Sequence[Component] | None = ...,
        queue: bool | None = ...,
        batch: bool = ...,
        max_batch_size: int = ...,
        preprocess: bool = ...,
        postprocess: bool = ...,
        cancels: dict[str, Any] | list[dict[str, Any]] | None = ...,
        every: Timer | float | None = ...,
        trigger_mode: Literal[once, multiple, always_last] | None = ...,
        js: str | Literal[True] | None = ...,
        concurrency_limit: int | None | Literal[default] = ...,
        concurrency_id: str | None = ...,
        api_visibility: Literal[public, private, undocumented] = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        api_description: str | None | Literal[False] = ...,
        validator: Callable[..., Any] | None = ...,
    ) -> Dependency: ...
