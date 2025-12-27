from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal

from gradio import routes
from gradio.components import Component
from gradio.events import EventData
from gradio.i18n import I18nData
from gradio_client.documentation import document

"""
Defines helper methods useful for loading and caching Interface examples.
"""

LOG_FILE = ...

def create_examples(
    examples: list[Any] | list[list[Any]] | str,
    inputs: Component | Sequence[Component],
    outputs: Component | Sequence[Component] | None = ...,
    fn: Callable | None = ...,
    cache_examples: bool | None = ...,
    cache_mode: Literal["eager", "lazy"] | None = ...,
    examples_per_page: int = ...,
    _api_mode: bool = ...,
    label: str | I18nData | None = ...,
    elem_id: str | None = ...,
    run_on_click: bool = ...,
    preprocess: bool = ...,
    postprocess: bool = ...,
    api_visibility: Literal["public", "private", "undocumented"] = ...,
    api_name: str | None = ...,
    api_description: str | None | Literal[False] = ...,
    batch: bool = ...,
    *,
    example_labels: list[str] | None = ...,
    visible: bool | Literal["hidden"] = ...,
    preload: int | Literal[False] = ...,
):  # -> Examples:

    ...

@document()
class Examples:
    def __init__(
        self,
        examples: list[Any] | list[list[Any]] | str,
        inputs: Component | Sequence[Component],
        outputs: Component | Sequence[Component] | None = ...,
        fn: Callable | None = ...,
        cache_examples: bool | None = ...,
        cache_mode: Literal["eager", "lazy"] | None = ...,
        examples_per_page: int = ...,
        _api_mode: bool = ...,
        label: str | I18nData | None = ...,
        elem_id: str | None = ...,
        run_on_click: bool = ...,
        preprocess: bool = ...,
        postprocess: bool = ...,
        api_visibility: Literal["public", "private", "undocumented"] = ...,
        api_name: str | None = ...,
        api_description: str | None | Literal[False] = ...,
        batch: bool = ...,
        *,
        example_labels: list[str] | None = ...,
        visible: bool | Literal["hidden"] = ...,
        preload: int | Literal[False] = ...,
        _initiated_directly: bool = ...,
    ) -> None: ...
    def create(self) -> None: ...
    async def cache(self, example_id: int | None = ...) -> None: ...
    def load_from_cache(self, example_id: int) -> list[Any]: ...

async def merge_generated_values_into_output(
    components: Sequence[Component], generated_values: list, output: list
):  # -> list[Any]:
    ...

class TrackedIterable:
    def __init__(
        self,
        iterable: Iterable | None,
        index: float | None,
        length: float | None,
        desc: str | None,
        unit: str | None,
        _tqdm=...,
        progress: float | None = ...,
    ) -> None: ...

@document("__call__", "tqdm")
class Progress(Iterable):
    def __init__(self, track_tqdm: bool = ...) -> None: ...
    def __len__(self) -> int:  # -> int | float | None:
        ...
    def __iter__(self):  # -> Self:
        ...
    def __next__(self):  # -> Self:

        ...
    def __call__(
        self,
        progress: float | tuple[int, int | None] | None,
        desc: str | None = ...,
        total: float | None = ...,
        unit: str = ...,
        _tqdm=...,
    ):  # -> float | tuple[int, int | None] | None:

        ...
    def tqdm(
        self,
        iterable: Iterable | None,
        desc: str | None = ...,
        total: float | None = ...,
        unit: str = ...,
        _tqdm=...,
    ):  # -> Self | Iterator[Any]:

        ...
    def update(self, n: float = ...):  # -> None:

        ...
    def close(self, _tqdm):  # -> None:

        ...

def patch_tqdm() -> None: ...
def create_tracker(
    fn, track_tqdm
):  # -> tuple[Progress, Any] | tuple[Progress, _Wrapped[..., Any, ..., AsyncGenerator[Any, Any]] | _Wrapped[..., Any, ..., CoroutineType[Any, Any, Any]] | _Wrapped[..., Any, ..., Generator[Any, Any, None]] | _Wrapped[..., Any, ..., Any]]:
    ...
def special_args(
    fn: Callable,
    inputs: list[Any] | None = ...,
    request: routes.Request | None = ...,
    event_data: EventData | None = ...,
    component_props: dict[int, dict[str, Any]] | None = ...,
) -> tuple[list, int | None, int | None, list[int]]: ...
def update(
    elem_id: str | None = ...,
    elem_classes: list[str] | str | None = ...,
    visible: bool | Literal["hidden"] | None = ...,
    **kwargs: Any,
) -> dict[str, Any]: ...
@document()
def validate(is_valid: bool, message: str):  # -> dict[str, str | bool]:

    ...
@document()
def skip() -> dict: ...
def log_message(
    message: str,
    title: str,
    level: Literal["info", "warning", "success"] = ...,
    duration: float | None = ...,
    visible: bool = ...,
):  # -> None:
    ...
@document(documentation_group="modals")
def Warning(message: str = ..., duration: float | None = ..., visible: bool = ..., title: str = ...):  # -> None:

    ...
@document(documentation_group="modals")
def Info(message: str = ..., duration: float | None = ..., visible: bool = ..., title: str = ...):  # -> None:

    ...
@document(documentation_group="modals")
def Success(message: str = ..., duration: float | None = ..., visible: bool = ..., title: str = ...):  # -> None:

    ...
