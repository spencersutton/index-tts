from typing import Any

"""
Defines internal helper methods for handling transformers and diffusers pipelines.
These are used by load_from_pipeline method in pipelines.py.
"""

def handle_transformers_pipeline(pipeline: Any) -> dict[str, Any] | None: ...
def handle_diffusers_pipeline(pipeline: Any) -> dict[str, Any] | None: ...
def handle_transformers_js_pipeline(pipeline: Any) -> dict[str, Any]: ...
