from torch.nn import Module

from ..model import Wav2Vec2Model

def import_fairseq_model(original: Module) -> Wav2Vec2Model: ...
