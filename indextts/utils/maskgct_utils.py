from collections.abc import ItemsView, KeysView, ValuesView

import torch
from transformers import Wav2Vec2BertModel

from indextts.utils.maskgct.models.codec.kmeans.repcodec_model import RepCodec


class JsonHParams:
    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = JsonHParams(**v)
            self[k] = v

    def keys(self) -> KeysView[str]:
        return self.__dict__.keys()

    def items(self) -> ItemsView[str, object]:
        return self.__dict__.items()

    def values(self) -> ValuesView[object]:
        return self.__dict__.values()

    def __len__(self) -> int:
        return len(self.__dict__)

    def __getitem__(self, key) -> object:
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        return setattr(self, key, value)

    def __contains__(self, key) -> bool:
        return key in self.__dict__

    def __repr__(self) -> str:
        return self.__dict__.__repr__()


def build_semantic_model(
    path_: str = "./models/tts/maskgct/ckpt/wav2vec2bert_stats.pt",
) -> tuple[Wav2Vec2BertModel, torch.Tensor, torch.Tensor]:
    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model.eval()
    stat_mean_var = torch.load(path_)
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    return semantic_model, semantic_mean, semantic_std


def build_semantic_codec() -> RepCodec:
    return RepCodec().eval()
