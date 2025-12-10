from pathlib import Path

import torch
from torch import Tensor
from transformers import Wav2Vec2BertModel


def build_semantic_model(
    path_: Path = Path("./models/tts/maskgct/ckpt/wav2vec2bert_stats.pt"),
) -> tuple[Wav2Vec2BertModel, Tensor, Tensor]:
    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model.eval()
    stat_mean_var = torch.load(path_)
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    return semantic_model, semantic_mean, semantic_std
