import logging
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import huggingface_hub as hf
import safetensors.torch
import torch
import torch._inductor.codecache  # noqa: F401 # pyright: ignore[reportUnusedImport]
import transformers
from torch import Tensor

from bigvgan import BigVGANInference as BigVGAN
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.s2mel.campplus.DTDNN import CAMPPlus
from indextts.s2mel.flow_matching import CFM
from indextts.s2mel.length_regulator import InterpolateRegulator
from indextts.util import Timer
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.repcodec_model import RepCodec

CHECKPOINT_DIR = Path("./checkpoints")
GPT_PATH = CHECKPOINT_DIR / "gpt_compiled.pt2"


def extract_features() -> transformers.SeamlessM4TFeatureExtractor:
    with Timer() as t:
        model = transformers.SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    print(f">> feature extractor restored in {t:.2f} seconds")
    return model


def gpt(device: torch.device, dim: int, use_accel: bool, use_compiled: bool = True) -> UnifiedVoice:
    import torch

    with Timer() as t:
        aoti_load_package = GPT_PATH
        assert aoti_load_package.exists(), (
            f"Compiled GPT model not found at {aoti_load_package}. Please run the load script to compile the model first."
        )
        if aoti_load_package.exists() and use_compiled:
            # Torch 2.10+ does not always attach this submodule eagerly, but the
            # PT2 archive loader references torch._inductor.codecache directly.
            # Importing it once ensures the attribute is available for AOTI load.
            from torch import _inductor as inductor

            print(f"Loading compiled GPT model from {aoti_load_package}")
            model = inductor.aoti_load_package(aoti_load_package)
            print(f">> Compiled GPT model loaded in {t:.2f} seconds from: {aoti_load_package}")
            return model
        path = CHECKPOINT_DIR / "gpt.safetensors"

        if not path.exists():
            pt_path = hf.hf_hub_download("IndexTeam/IndexTTS-2", filename="gpt.pth")
            data = cast(dict[str, Tensor], torch.load(pt_path, map_location="cpu"))
            safetensors.torch.save_file(data, path)

        data = safetensors.torch.load_file(path, device=str(device))

        with torch.device("meta"):
            model = UnifiedVoice(dim, use_accel=use_accel)
        model.load_state_dict(data, assign=True)

    print(f">> GPT weights restored in {t:.2f} seconds from: {path}")
    return model.eval()


def campplus_model(device: torch.device) -> CAMPPlus:
    with Timer() as t:
        path = hf.hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
        data = cast(Mapping[str, object], torch.load(path, map_location=device))

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
        path = CHECKPOINT_DIR / "semantic_codec.safetensors"
        if not path.exists():
            cache_path = hf.hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
            cache_data = safetensors.torch.load_file(cache_path)
            for k in list(cache_data.keys()):
                if k.startswith("decoder."):
                    del cache_data[k]
            safetensors.torch.save_file(cache_data, path)

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


def _get_s2mel_checkpoint() -> dict[str, dict[str, dict[str, Tensor]]]:
    path = hf.hf_hub_download("IndexTeam/IndexTTS-2", "s2mel.pth")
    return cast(dict[str, dict[str, dict[str, Tensor]]], torch.load(path, map_location="cpu", weights_only=False))


def cfm(device: torch.device, dim: int) -> CFM:
    with Timer() as t:
        path = CHECKPOINT_DIR / "cfm.safetensors"
        if not path.exists():
            data = _get_s2mel_checkpoint()
            data = data["net"]["cfm"]
            del data["estimator.x_embedder.bias"]
            del data["estimator.x_embedder.weight_g"]
            del data["estimator.x_embedder.weight_v"]
            del data["estimator.cond_embedder.weight"]
            del data["estimator.content_mask_embedder.weight"]
            safetensors.torch.save_file(data, path)
        else:
            data = safetensors.torch.load_file(path, device=str(device))
        with torch.device("meta"):
            model = CFM(dim)
        model.load_state_dict(data, assign=True)

    print(f">> CFM weights restored in {t:.2f} seconds from: {path}")
    return model.eval()


def length_regulator(device: torch.device, dim: int) -> InterpolateRegulator:
    with Timer() as t:
        path = CHECKPOINT_DIR / "length_regulator.safetensors"
        if not path.exists():
            data = _get_s2mel_checkpoint()
            data = data["net"]["length_regulator"]
            del data["embedding.weight"]
            del data["mask_token"]
            safetensors.torch.save_file(data, path)
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


if __name__ == "__main__":
    from torch import _inductor as inductor

    # Reduce matmul precision warning noise while improving TensorCore throughput.
    torch.set_float32_matmul_precision("high")
    if hasattr(torch, "_inductor") and hasattr(inductor, "config"):
        inductor.config.max_autotune = False
        inductor.config.max_autotune_gemm = False
    logging.getLogger("_inductor.utils").setLevel(logging.ERROR)

    device = torch.device("cuda")
    gpt_model = gpt(device, 512, use_accel=False, use_compiled=False)
    warnings.filterwarnings("ignore", module=r".*copyreg", lineno=104, category=FutureWarning)
    with Timer() as t, device:
        export_module = torch.export.export(
            gpt_model,
            args=(
                torch.zeros(1, 32, UnifiedVoice.voice_dim),
                torch.zeros(1, 16, dtype=torch.long),
                torch.zeros(1, 32, dtype=torch.long),
                torch.zeros(1, UnifiedVoice.voice_dim),
            ),
        )
    print(f">> GPT model exported in {t:.2f} seconds.")

    with Timer() as t:
        output_path = inductor.aoti_compile_and_package(
            export_module, package_path=GPT_PATH, inductor_configs={"max_autotune": False, "max_autotune_gemm": False}
        )
    print(f"Model compiled to: {output_path} in {t:.2f} seconds.")
