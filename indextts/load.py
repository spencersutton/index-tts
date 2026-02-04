from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import huggingface_hub as hf
import safetensors.torch
import torch
import transformers
from torch import Tensor

from BigVGANInference.bigvganinference import BigVGANInference as BigVGAN
from indextts.config import UnifiedVoiceConfig
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.flow_matching import CFM
from indextts.s2mel.modules.length_regulator import InterpolateRegulator
from indextts.util import Timer
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.repcodec_model import RepCodec


def extract_features() -> transformers.SeamlessM4TFeatureExtractor:
    with Timer() as t:
        model = transformers.SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    print(f">> feature extractor restored in {t:.2f} seconds")
    return model


def gpt(device: torch.device, cfg: UnifiedVoiceConfig, use_accel: bool, use_fp16: bool) -> UnifiedVoice:
    with Timer() as t:
        path = "./checkpoints/gpt.safetensors"
        data = safetensors.torch.load_file(path, device=str(device))

        with torch.device("meta"):
            model = UnifiedVoice(cfg=cfg, use_accel=use_accel)
        model.load_state_dict(data, assign=True)

        if use_fp16:
            model = model.half()

    print(f">> GPT weights restored in {t:.2f} seconds from: {path}")
    return model.eval()


def campplus_model(device: torch.device) -> CAMPPlus:
    with Timer() as t:
        path = hf.hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
        data: Mapping[str, Any] = torch.load(path, map_location=device)  # pyright: ignore

        with torch.device("meta"):
            model = CAMPPlus()
        model.load_state_dict(data, assign=True)
        model = model.eval().to(device)

    print(f">> campplus_model weights restored in {t:.2f} seconds from: {path}")
    return model


def tokenizer(normalizer: TextNormalizer) -> TextTokenizer:
    with Timer() as t:
        path = Path(hf.hf_hub_download("IndexTeam/IndexTTS-2", "bpe.model"))
        tokenizer = TextTokenizer(path, normalizer)

    print(f">> bpe model restored in {t:.2f} seconds from: {path}")
    return tokenizer


def semantic_codec(device: torch.device) -> RepCodec:
    with Timer() as t:
        path = "checkpoints/semantic_codec.safetensors"

        with torch.device("meta"):
            model = RepCodec()
        data = safetensors.torch.load_file(path, device=str(device))
        model.load_state_dict(data, assign=True)
    print(f">> semantic_codec weights restored from: {path} in {t:.2f} seconds")
    return model.eval()


def semantic_model(device: torch.device) -> transformers.Wav2Vec2BertModel:
    with Timer() as t:
        model = transformers.Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
        model = model.eval().to(device)
    print(f">> semantic_model weights restored in {t:.2f} seconds")
    return model


def semantic_stats(device: torch.device) -> tuple[Tensor, Tensor]:
    with Timer() as t:
        path = hf.hf_hub_download("amphion/dualcodec", "w2vbert2_mean_var_stats_emilia.pt")
        data = cast(dict[str, Tensor], torch.load(path))
        mean = data["mean"].to(device)
        std = data["var"].sqrt().to(device)

    print(f">> semantic_mean and semantic_var weights restored in {t:.2f} seconds from: {path}")
    return mean, std


def cfm(device: torch.device, dim: int = 512, in_channels: int = 80) -> CFM:
    with Timer() as t:
        path = "checkpoints/cfm.safetensors"
        data = safetensors.torch.load_file(path, device=str(device))
        with torch.device("meta"):
            model = CFM(dim=dim, in_channels=in_channels)
        model.load_state_dict(data, assign=True)

    print(f">> CFM weights restored in {t:.2f} seconds from: {path}")
    return model.eval()


def length_regulator(device: torch.device, dim: int = 512) -> InterpolateRegulator:
    with Timer() as t:
        path = "checkpoints/length_regulator.safetensors"
        data = safetensors.torch.load_file(path, device=str(device))
        with torch.device("meta"):
            model = InterpolateRegulator(dim)
        model.load_state_dict(data, assign=True)

    print(f">> Length Regulator weights restored in {t:.2f} seconds from: {path}")
    return model.eval()


def bigvgan(device: torch.device, use_cuda_kernel: bool) -> BigVGAN:
    with Timer() as t:
        model = BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel=use_cuda_kernel)
        model.remove_weight_norm()
        model = model.eval().to(device)

    print(f">> bigvgan weights restored in {t:.2f} seconds.")
    return model
