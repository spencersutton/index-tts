from typing import Literal

from ._common import TaskProviderHelper

logger = ...
type PROVIDER_T = Literal[
    "black-forest-labs",
    "cerebras",
    "clarifai",
    "cohere",
    "fal-ai",
    "featherless-ai",
    "fireworks-ai",
    "groq",
    "hf-inference",
    "hyperbolic",
    "nebius",
    "novita",
    "nscale",
    "openai",
    "publicai",
    "replicate",
    "sambanova",
    "scaleway",
    "together",
    "zai-org",
]
type PROVIDER_OR_POLICY_T = PROVIDER_T | Literal["auto"]
PROVIDERS: dict[PROVIDER_T, dict[str, TaskProviderHelper]] = ...

def get_provider_helper(provider: PROVIDER_OR_POLICY_T | None, task: str, model: str | None) -> TaskProviderHelper: ...
