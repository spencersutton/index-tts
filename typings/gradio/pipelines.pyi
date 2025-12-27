import diffusers
import transformers

"""This module should not be used directly as its API is subject to change. Instead,
please use the `gr.Interface.from_pipeline()` function."""

def load_from_pipeline(pipeline: transformers.Pipeline | diffusers.DiffusionPipeline) -> dict: ...
def load_from_js_pipeline(pipeline) -> dict: ...
