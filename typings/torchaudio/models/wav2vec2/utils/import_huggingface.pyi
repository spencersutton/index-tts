from typing import Any

from torch.nn import Module

from ..model import Wav2Vec2Model

"""Import Hugging Face transformers's wav2vec2.0 pretrained weights to torchaudios's format.
"""
_LG = ...

def transform_wavlm_encoder_state(state: dict[str, Any], encoder_num_layers: int) -> None: ...
def import_huggingface_model(original: Module) -> Wav2Vec2Model: ...
