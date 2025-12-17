from typing import Any

from huggingface_hub.inference._common import RequestParameters
from huggingface_hub.inference._providers._common import BaseConversationalTask, TaskProviderHelper

class HyperbolicTextToImageTask(TaskProviderHelper):
    def __init__(self) -> None: ...
    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = ...) -> Any: ...

class HyperbolicTextGenerationTask(BaseConversationalTask):
    def __init__(self, task: str) -> None: ...
