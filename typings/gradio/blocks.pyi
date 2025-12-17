import weakref
from collections.abc import AsyncIterator, Callable, Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import fastapi
from gradio.block_function import BlockFunction
from gradio.blocks_events import BlocksEvents, BlocksMeta
from gradio.components.base import Component
from gradio.data_classes import APIInfo, BlocksConfigDict
from gradio.events import EventData, EventListener, EventListenerMethod
from gradio.i18n import I18n, I18nData
from gradio.renderable import Renderable
from gradio.routes import App, Request
from gradio.state_holder import SessionState
from gradio.themes import ThemeClass as Theme
from gradio_client.documentation import document

if TYPE_CHECKING: ...

class Block:
    def __init__(
        self,
        *,
        elem_id: str | None = ...,
        elem_classes: list[str] | str | None = ...,
        render: bool = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
        visible: bool | Literal["hidden"] = ...,
        proxy_url: str | None = ...,
    ) -> None: ...
    def unique_key(self) -> int | None: ...
    @property
    def stateful(self) -> bool: ...
    @property
    def skip_api(self) -> bool: ...
    @property
    def constructor_args(self) -> dict[str, Any]: ...
    @property
    def events(self) -> list[EventListener]: ...
    def render(self):  # -> Component | Block | Self:

        ...
    def unrender(self):  # -> Self:

        ...
    def get_block_name(self) -> str: ...
    def get_block_class(self) -> str: ...
    def get_expected_parent(self) -> type[BlockContext] | None: ...
    def breaks_grouping(self) -> bool: ...
    def get_config(self, cls: type[Block] | None = ...) -> dict[str, Any]: ...
    @classmethod
    def recover_kwargs(cls, props: dict[str, Any], additional_keys: list[str] | None = ...):  # -> dict[Any, Any]:

        ...
    async def async_move_resource_to_block_cache(self, url_or_file_path: str | Path | None) -> str | None: ...
    def move_resource_to_block_cache(self, url_or_file_path: str | Path | None) -> str | None: ...
    def serve_static_file(self, url_or_file_path: str | Path | dict | None) -> dict | None: ...

class BlockContext(Block):
    def __init__(
        self,
        elem_id: str | None = ...,
        elem_classes: list[str] | str | None = ...,
        visible: bool | Literal["hidden"] = ...,
        render: bool = ...,
        key: int | str | tuple[int | str, ...] | None = ...,
        preserved_by_key: list[str] | str | None = ...,
    ) -> None: ...

    TEMPLATE_DIR = ...
    FRONTEND_DIR = ...
    @property
    def skip_api(self):  # -> Literal[True]:
        ...
    @classmethod
    def get_component_class_id(cls) -> str: ...
    @property
    def component_class_id(self):  # -> str:
        ...
    def add_child(self, child: Block):  # -> None:
        ...
    def __enter__(self):  # -> Self:
        ...
    def add(self, child: Block):  # -> None:
        ...
    def fill_expected_parents(self):  # -> None:
        ...
    def __exit__(self, exc_type: type[BaseException] | None = ..., *args):  # -> None:
        ...
    def postprocess(self, y): ...

def postprocess_update_dict(
    block: Component | BlockContext, update_dict: dict, postprocess: bool = ...
):  # -> dict[Any, Any]:

    ...
def convert_component_dict_to_list(outputs_ids: list[int], predictions: dict) -> list | dict: ...

class BlocksConfig:
    def __init__(self, root_block: Blocks) -> None: ...
    def set_event_trigger(
        self,
        targets: Sequence[EventListenerMethod],
        fn: Callable | None,
        inputs: (
            Component | BlockContext | Sequence[Component | BlockContext] | AbstractSet[Component | BlockContext] | None
        ),
        outputs: (
            Component | BlockContext | Sequence[Component | BlockContext] | AbstractSet[Component | BlockContext] | None
        ),
        preprocess: bool = ...,
        postprocess: bool = ...,
        scroll_to_output: bool = ...,
        show_progress: Literal["full", "minimal", "hidden"] = ...,
        show_progress_on: Component | Sequence[Component] | None = ...,
        api_name: str | None = ...,
        api_description: str | None | Literal[False] = ...,
        js: str | Literal[True] | None = ...,
        no_target: bool = ...,
        queue: bool = ...,
        batch: bool = ...,
        max_batch_size: int = ...,
        cancels: list[int] | None = ...,
        collects_event_data: bool | None = ...,
        trigger_after: int | None = ...,
        trigger_only_on_success: bool = ...,
        trigger_only_on_failure: bool = ...,
        trigger_mode: Literal["once", "multiple", "always_last"] | None = ...,
        concurrency_limit: int | None | Literal["default"] = ...,
        concurrency_id: str | None = ...,
        api_visibility: Literal["public", "private", "undocumented"] = ...,
        renderable: Renderable | None = ...,
        is_cancel_function: bool = ...,
        connection: Literal["stream", "sse"] = ...,
        time_limit: float | None = ...,
        stream_every: float = ...,
        event_specific_args: list[str] | None = ...,
        js_implementation: str | None = ...,
        key: str | int | tuple[int | str, ...] | None = ...,
        validator: Callable | None = ...,
        component_prop_inputs: list[int] | None = ...,
    ) -> tuple[BlockFunction, int]: ...
    @staticmethod
    def config_for_block(
        _id: int, rendered_ids: list[int], block: Block | Component, renderable: Renderable | None = ...
    ) -> dict: ...
    def get_config(self, renderable: Renderable | None = ...):  # -> dict[str, dict[Any, Any] | list[Any]]:
        ...
    def __copy__(self):  # -> BlocksConfig:
        ...
    def attach_load_events(self, rendered_in: Renderable | None = ...):  # -> None:

        ...

@document("launch", "queue", "integrate", "load", "unload")
class Blocks(BlockContext, BlocksEvents, metaclass=BlocksMeta):
    instances: weakref.WeakSet = ...
    @classmethod
    def get_instances(cls) -> list[Blocks]: ...
    def __init__(
        self,
        analytics_enabled: bool | None = ...,
        mode: str = ...,
        title: str | I18nData = ...,
        fill_height: bool = ...,
        fill_width: bool = ...,
        delete_cache: tuple[int, int] | None = ...,
        **kwargs,
    ) -> None: ...
    @property
    def blocks(self) -> dict[int, Component | Block]: ...
    @blocks.setter
    def blocks(self, value: dict[int, Component | Block]):  # -> None:
        ...
    @property
    def fns(self) -> dict[int, BlockFunction]: ...
    def get_component(self, id: int) -> Component | BlockContext: ...
    @classmethod
    def from_config(cls, config: BlocksConfigDict, fns: list[Callable], proxy_url: str) -> Blocks: ...
    @property
    def expects_oauth(self):  # -> bool:

        ...
    def unload(self, fn: Callable[..., Any]) -> None: ...
    def render(self):  # -> Self:
        ...
    def is_callable(self, fn_index: int = ...) -> bool: ...
    def __call__(self, *inputs, fn_index: int = ..., api_name: str | None = ...):  # -> Any:

        ...
    async def call_function(
        self,
        block_fn: BlockFunction | int,
        processed_input: list[Any],
        iterator: AsyncIterator[Any] | None = ...,
        requests: Request | list[Request] | None = ...,
        event_id: str | None = ...,
        event_data: EventData | None = ...,
        in_event_listener: bool = ...,
        state: SessionState | None = ...,
    ):  # -> dict[str, Any | tuple[Literal[_Keywords.FINISHED_ITERATING], ...] | _Keywords | float | bool | SyncToAsyncIterator | AsyncIterator[Any] | None]:

        ...
    def serialize_data(self, fn_index: int, inputs: list[Any]) -> list[Any]: ...
    def deserialize_data(self, fn_index: int, outputs: list[Any]) -> list[Any]: ...
    def validate_inputs(self, block_fn: BlockFunction, inputs: list[Any]):  # -> None:
        ...
    async def preprocess_data(
        self, block_fn: BlockFunction, inputs: list[Any], state: SessionState | None, explicit_call: bool = ...
    ):  # -> list[Any]:
        ...
    def validate_outputs(self, block_fn: BlockFunction, predictions: Any | list[Any]):  # -> None:
        ...
    async def postprocess_data(
        self, block_fn: BlockFunction, predictions: list | dict, state: SessionState | None
    ) -> list[Any]: ...
    async def handle_streaming_outputs(
        self,
        block_fn: BlockFunction,
        data: list,
        session_hash: str | None,
        run: int | None,
        root_path: str | None = ...,
        final: bool = ...,
    ) -> list: ...
    def handle_streaming_diffs(
        self,
        block_fn: BlockFunction,
        data: list,
        session_hash: str | None,
        run: int | None,
        final: bool,
        simple_format: bool = ...,
    ) -> list: ...
    async def process_api(
        self,
        block_fn: BlockFunction | int,
        inputs: list[Any],
        state: SessionState | None = ...,
        request: Request | list[Request] | None = ...,
        iterator: AsyncIterator | None = ...,
        session_hash: str | None = ...,
        event_id: str | None = ...,
        event_data: EventData | None = ...,
        in_event_listener: bool = ...,
        simple_format: bool = ...,
        explicit_call: bool = ...,
        root_path: str | None = ...,
    ) -> dict[str, Any]: ...
    def get_state_ids_to_track(self, block_fn: BlockFunction, state: SessionState | None) -> tuple[list[int], list]: ...
    def create_limiter(self):  # -> None:
        ...
    def get_config(self):  # -> dict[str, str]:
        ...
    def get_config_file(self) -> BlocksConfigDict: ...
    def transpile_to_js(self, quiet: bool = ...):  # -> None:
        ...
    def __enter__(self):  # -> Self:
        ...
    def __exit__(self, exc_type: type[BaseException] | None = ..., *args):  # -> None:
        ...
    def clear(self):  # -> Self:

        ...
    @document()
    def queue(
        self,
        status_update_rate: float | Literal["auto"] = ...,
        api_open: bool | None = ...,
        max_size: int | None = ...,
        *,
        default_concurrency_limit: int | None | Literal["not_set"] = ...,
    ):  # -> Self:

        ...
    def validate_queue_settings(self):  # -> None:
        ...
    def validate_navbar_settings(self):  # -> None:

        ...
    def launch(
        self,
        inline: bool | None = ...,
        inbrowser: bool = ...,
        share: bool | None = ...,
        debug: bool = ...,
        max_threads: int = ...,
        auth: (Callable[[str, str], bool] | tuple[str, str] | list[tuple[str, str]] | None) = ...,
        auth_message: str | None = ...,
        prevent_thread_lock: bool = ...,
        show_error: bool = ...,
        server_name: str | None = ...,
        server_port: int | None = ...,
        *,
        height: int = ...,
        width: int | str = ...,
        favicon_path: str | Path | None = ...,
        ssl_keyfile: str | None = ...,
        ssl_certfile: str | None = ...,
        ssl_keyfile_password: str | None = ...,
        ssl_verify: bool = ...,
        quiet: bool = ...,
        footer_links: list[Literal["api", "gradio", "settings"] | dict[str, str]] | None = ...,
        allowed_paths: list[str] | None = ...,
        blocked_paths: list[str] | None = ...,
        root_path: str | None = ...,
        app_kwargs: dict[str, Any] | None = ...,
        state_session_capacity: int = ...,
        share_server_address: str | None = ...,
        share_server_protocol: Literal["http", "https"] | None = ...,
        share_server_tls_certificate: str | None = ...,
        auth_dependency: Callable[[fastapi.Request], str | None] | None = ...,
        max_file_size: str | int | None = ...,
        enable_monitoring: bool | None = ...,
        strict_cors: bool = ...,
        node_server_name: str | None = ...,
        node_port: int | None = ...,
        ssr_mode: bool | None = ...,
        pwa: bool | None = ...,
        mcp_server: bool | None = ...,
        _frontend: bool = ...,
        i18n: I18n | None = ...,
        theme: Theme | str | None = ...,
        css: str | None = ...,
        css_paths: str | Path | Sequence[str | Path] | None = ...,
        js: str | Literal[True] | None = ...,
        head: str | None = ...,
        head_paths: str | Path | Sequence[str | Path] | None = ...,
    ) -> tuple[App, str, str]: ...
    def integrate(self, comet_ml=..., wandb: ModuleType | None = ..., mlflow: ModuleType | None = ...) -> None: ...
    def close(self, verbose: bool = ...) -> None: ...
    def block_thread(self) -> None: ...
    def run_startup_events(self):  # -> None:

        ...
    async def run_extra_startup_events(self):  # -> None:
        ...
    def get_api_info(self, all_endpoints: bool = ...) -> APIInfo: ...
    @staticmethod
    def get_event_targets(original_mapping: dict[int, Block], _targets: list, trigger: str) -> list: ...
    @document()
    def route(self, name: str, path: str | None = ..., show_in_navbar: bool = ...) -> Blocks: ...
