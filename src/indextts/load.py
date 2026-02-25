from collections.abc import Mapping
from pathlib import Path
from typing import cast

import huggingface_hub as hf
import safetensors.torch
import torch
import transformers
from torch import Tensor, nn

from bigvgan import BigVGANInference as BigVGAN
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.s2mel.campplus.DTDNN import CAMPPlus
from indextts.s2mel.flow_matching import CFM
from indextts.s2mel.length_regulator import InterpolateRegulator
from indextts.util import Timer
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.repcodec_model import RepCodec

CHECKPOINT_DIR = Path("./checkpoints")


def _restore_model_weights[T: nn.Module](module: type[T], device: torch.device, name: str) -> T:
    path = CHECKPOINT_DIR / name
    print(f">> Restoring {name} weights from: {path} to {device}...")
    data = safetensors.torch.load_file(path, device=str(device))
    with torch.device("meta"):
        model = module()
    model.load_state_dict(data, assign=True)
    return model.eval()


def load_feature_extractor() -> transformers.SeamlessM4TFeatureExtractor:
    return transformers.SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")


def load_unified_voice(device: torch.device) -> UnifiedVoice:
    return _restore_model_weights(UnifiedVoice, device, "gpt.safetensors")


def load_campplus(device: torch.device) -> CAMPPlus:
    with Timer() as t:
        path = hf.hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
        data = torch.load(path, map_location=device)
        data = cast(Mapping[str, object], data)

        with torch.device("meta"):
            model = CAMPPlus()
        model.load_state_dict(data, assign=True)
        model = model.eval()

    print(f">> campplus_model weights restored in {t:.2f} seconds from: {path}")
    return model


def load_tokenizer(normalizer: TextNormalizer) -> TextTokenizer:
    with Timer() as t:
        path = Path(hf.hf_hub_download("IndexTeam/IndexTTS-2", "bpe.model"))
        tokenizer = TextTokenizer(path, normalizer)

    print(f">> bpe model restored in {t:.2f} seconds from: {path}")
    return tokenizer


def load_semantic_codec(device: torch.device) -> RepCodec:
    return _restore_model_weights(RepCodec, device, "semantic_codec.safetensors")


def load_semantic_model(device: torch.device) -> transformers.Wav2Vec2BertModel:
    with Timer() as t:
        model = transformers.Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
        model = model.eval().to(device)
    print(f">> semantic_model weights restored in {t:.2f} seconds")
    return model


def load_semantic_stats(device: torch.device) -> tuple[Tensor, Tensor]:
    with Timer() as t:
        path = hf.hf_hub_download("amphion/dualcodec", "w2vbert2_mean_var_stats_emilia.pt")
        data = torch.load(path)
        data = cast(dict[str, Tensor], data)
        mean = data["mean"].to(device)
        std = data["var"].sqrt().to(device)

    print(f">> semantic_mean and semantic_var weights restored in {t:.2f} seconds from: {path}")
    return mean, std


def load_cfm(device: torch.device) -> CFM:
    return _restore_model_weights(CFM, device, "cfm.safetensors")


def load_length_regulator(device: torch.device) -> InterpolateRegulator:
    return _restore_model_weights(InterpolateRegulator, device, "length_regulator.safetensors")


def load_bigvgan(device: torch.device, use_cuda_kernel: bool) -> BigVGAN:
    with Timer() as t:
        model = BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel=use_cuda_kernel)
        model.remove_weight_norm()
        model = model.eval().to(device)

    print(f">> bigvgan weights restored in {t:.2f} seconds.")
    return model


if __name__ == "__main__":
    path = CHECKPOINT_DIR / "gpt.safetensors"

    if not path.exists():
        pt_path = hf.hf_hub_download("IndexTeam/IndexTTS-2", filename="gpt.pth")
        data = torch.load(pt_path, map_location="cpu")
        data = cast(dict[str, Tensor], data)
        safetensors.torch.save_file(data, path)

    path = CHECKPOINT_DIR / "semantic_codec.safetensors"
    if not path.exists():
        pt_path = hf.hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        data = safetensors.torch.load_file(pt_path)
        for k in list(data.keys()):
            if k.startswith("decoder."):
                del data[k]
        safetensors.torch.save_file(data, path)

    lr_path = CHECKPOINT_DIR / "length_regulator.safetensors"
    cfm_path = CHECKPOINT_DIR / "cfm.safetensors"
    if not lr_path.exists() or not cfm_path.exists():
        path = hf.hf_hub_download("IndexTeam/IndexTTS-2", "s2mel.pth")
        s2mel_data = torch.load(path, map_location="cpu", weights_only=False)
        s2mel_data = cast(dict[str, dict[str, dict[str, Tensor]]], s2mel_data)
        if not lr_path.exists():
            data = s2mel_data["net"]["length_regulator"]
            del data["embedding.weight"]
            del data["mask_token"]
            safetensors.torch.save_file(data, lr_path)

        if not cfm_path.exists():
            data = s2mel_data["net"]["cfm"]
            del data["estimator.x_embedder.bias"]
            del data["estimator.x_embedder.weight_g"]
            del data["estimator.x_embedder.weight_v"]
            del data["estimator.cond_embedder.weight"]
            del data["estimator.content_mask_embedder.weight"]
            safetensors.torch.save_file(data, cfm_path)
