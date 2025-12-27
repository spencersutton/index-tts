from collections.abc import Callable
from typing import Literal

from gradio.blocks import Blocks
from gradio.chat_interface import ChatInterface
from gradio.components.chatbot import NormalizedMessageDict
from gradio.components.login_button import LoginButton
from gradio.components.multimodal_textbox import MultimodalValue
from gradio.interface import Interface
from gradio_client.documentation import document
from huggingface_hub.inference._providers import PROVIDER_T

"""This module should not be used directly as its API is subject to change. Instead,
use the `gr.Blocks.load()` or `gr.load()` functions."""

@document()
def load(
    name: str,
    src: Callable[[str, str | None], Blocks] | Literal["models", "spaces", "huggingface"] | None = ...,
    token: str | None = ...,
    accept_token: bool | LoginButton = ...,
    provider: PROVIDER_T | None = ...,
    **kwargs,
) -> Blocks: ...
def load_blocks_from_huggingface(
    name: str, src: str, token: str | None = ..., alias: str | None = ..., provider: PROVIDER_T | None = ..., **kwargs
) -> Blocks: ...
def from_model(
    model_name: str, token: str | None, alias: str | None, provider: PROVIDER_T | None = ..., **kwargs
) -> Blocks: ...
def from_spaces(
    space_name: str, token: str | None, alias: str | None, provider: PROVIDER_T | None = ..., **kwargs
) -> Blocks: ...
def make_event_data_fn(client, endpoint):  # -> Callable[..., Any]:

    ...
def from_spaces_blocks(space: str, token: str | None) -> Blocks: ...
def from_spaces_interface(
    model_name: str, config: dict, alias: str | None, token: str | None, iframe_url: str, **kwargs
) -> Interface: ...

TEXT_FILE_EXTENSIONS = ...
IMAGE_FILE_EXTENSIONS = ...

def format_conversation(history: list[NormalizedMessageDict], new_message: str | MultimodalValue) -> list[dict]: ...
@document()
def load_chat(
    base_url: str,
    model: str,
    token: str | None = ...,
    *,
    file_types: Literal["text_encoded", "image"] | list[Literal["text_encoded", "image"]] | None = ...,
    system_message: str | None = ...,
    streaming: bool = ...,
    **kwargs,
) -> ChatInterface: ...
@document()
def load_openapi(
    openapi_spec: str | dict,
    base_url: str,
    *,
    paths: list[str] | None = ...,
    exclude_paths: list[str] | None = ...,
    methods: list[Literal["get", "post", "put", "patch", "delete"]] | None = ...,
    auth_token: str | None = ...,
    **interface_kwargs,
) -> Blocks: ...
