import typing

RUFF_INSTALLED = ...

def find_first_non_return_key(some_dict):  # -> None:

    ...
def format(code: str, type: str):  # -> str | Any:

    ...
def get_param_name(param):  # -> str | Any | None:

    ...
def format_none(value):  # -> Literal['None']:

    ...
def format_value(value):  # -> str:

    ...
def get_parameter_docstring(docstring: str, parameter_name: str):  # -> str | Any | None:

    ...
def get_return_docstring(docstring: str):  # -> str | Any | None:

    ...
def add_value(obj: dict, key: str, value: typing.Any):  # -> dict[Any, Any]:

    ...
def set_deep(dictionary: dict, keys: list[str], value: typing.Any):  # -> None:

    ...
def get_deep(dictionary: dict, keys: list[str], default=...):  # -> dict[Any, Any] | None:

    ...
def get_type_arguments(type_hint) -> tuple: ...
def get_container_name(arg):  # -> str:

    ...
def format_type(_type: list[typing.Any]):  # -> LiteralString | str | None:

    ...
def get_type_hints(param, module):  # -> tuple[Any | LiteralString | str | None, dict[Any, Any], list[Any]]:

    ...
def extract_docstrings(module):  # -> tuple[dict[Any, Any], Literal['complex', 'simple']]:
    ...

class AdditionalInterface(typing.TypedDict):
    refs: list[str]
    source: str

def make_js(
    interfaces: dict[str, AdditionalInterface] | None = ..., user_fn_refs: dict[str, list[str]] | None = ...
):  # -> str:

    ...
def render_additional_interfaces(interfaces):  # -> str:

    ...
def render_additional_interfaces_markdown(interfaces):  # -> str:

    ...
def render_version_badge(pypi_exists, local_version, name):  # -> str:

    ...
def render_github_badge(repo):  # -> str:

    ...
def render_discuss_badge(space):  # -> str:

    ...
def render_class_events(events: dict, name):  # -> str:

    ...
def make_user_fn(
    class_name, user_fn_input_type, user_fn_input_description, user_fn_output_type, user_fn_output_description
):  # -> str:

    ...
def format_description(description): ...
def make_user_fn_markdown(
    user_fn_input_type, user_fn_input_description, user_fn_output_type, user_fn_output_description
):  # -> str:

    ...
def render_class_events_markdown(events):  # -> str:

    ...
def render_class_docs(exports, docs):  # -> str:

    ...

html = ...

def render_param_table(params):  # -> str:

    ...
def render_class_docs_markdown(exports, docs):  # -> str:

    ...
def make_space(
    docs: dict,
    name: str,
    description: str,
    local_version: str | None,
    demo: str,
    space: str | None,
    repo: str | None,
    pypi_exists: bool,
    suppress_demo_check: bool = ...,
):  # -> str:
    ...
def make_markdown(docs, name, description, local_version, demo, space, repo, pypi_exists):  # -> str:
    ...
