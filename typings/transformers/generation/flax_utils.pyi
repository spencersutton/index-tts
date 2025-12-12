import flax
import jax.numpy as jnp
from typing import Optional
from ..utils import ModelOutput
from .configuration_utils import GenerationConfig
from .flax_logits_process import FlaxLogitsProcessorList

logger = ...

@flax.struct.dataclass
class FlaxGreedySearchOutput(ModelOutput):
    sequences: Optional[jnp.ndarray] = ...

@flax.struct.dataclass
class FlaxSampleOutput(ModelOutput):
    sequences: Optional[jnp.ndarray] = ...

@flax.struct.dataclass
class FlaxBeamSearchOutput(ModelOutput):
    sequences: Optional[jnp.ndarray] = ...
    scores: Optional[jnp.ndarray] = ...

@flax.struct.dataclass
class GreedyState:
    cur_len: jnp.ndarray
    sequences: jnp.ndarray
    running_token: jnp.ndarray
    is_sent_finished: jnp.ndarray
    model_kwargs: dict[str, jnp.ndarray]

@flax.struct.dataclass
class SampleState:
    cur_len: jnp.ndarray
    sequences: jnp.ndarray
    running_token: jnp.ndarray
    is_sent_finished: jnp.ndarray
    prng_key: jnp.ndarray
    model_kwargs: dict[str, jnp.ndarray]

@flax.struct.dataclass
class BeamSearchState:
    cur_len: jnp.ndarray
    running_sequences: jnp.ndarray
    running_scores: jnp.ndarray
    sequences: jnp.ndarray
    scores: jnp.ndarray
    is_sent_finished: jnp.ndarray
    model_kwargs: dict[str, jnp.ndarray]

class FlaxGenerationMixin:
    def prepare_inputs_for_generation(self, *args, **kwargs): ...
    def generate(
        self,
        input_ids: jnp.ndarray,
        generation_config: Optional[GenerationConfig] = ...,
        prng_key: Optional[jnp.ndarray] = ...,
        trace: bool = ...,
        params: Optional[dict[str, jnp.ndarray]] = ...,
        logits_processor: Optional[FlaxLogitsProcessorList] = ...,
        **kwargs,
    ):  # -> FlaxGreedySearchOutput | FlaxSampleOutput | FlaxBeamSearchOutput:

        ...
