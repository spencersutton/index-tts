from torch.nn import Module

from ..model import Wav2Vec2Model

"""Import fariseq's wav2vec2.0 pretrained weights to torchaudios's format.

For this module to work, you need `fairseq`.
"""

def import_fairseq_model(original: Module) -> Wav2Vec2Model: ...
