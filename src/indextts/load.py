import logging
from pathlib import Path
from typing import cast

import huggingface_hub as hf
import safetensors.torch
import torch
import transformers
from beartype import beartype
from jaxtyping import Float
from torch import Tensor, nn

from bigvgan import BigVGANInference as BigVGAN
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.s2mel.campplus.DTDNN import CAMPPlus
from indextts.s2mel.flow_matching import CFM
from indextts.s2mel.length_regulator import InterpolateRegulator
from indextts.util import Timer
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.repcodec_model import RepCodec

logger = logging.getLogger(__name__)

type TensorDict = dict[str, Tensor]

CHECKPOINT_DIR = Path("./checkpoints")
CAMPPLUS_FILE = "campplus_cn_common.safetensors"
SEMANTIC_CODEC_FILE = "semantic_codec.safetensors"
GPT_FILE = "gpt.safetensors"
CFM_FILE = "cfm.safetensors"
LENGTH_REGULATOR_FILE = "length_regulator.safetensors"


def _restore_model_weights[T: nn.Module](module: type[T], filename: str) -> T:
    """Instantiate *module* on the meta device, load safetensors weights from *filename*, and return an eval model.

    Args:
        module: The ``nn.Module`` subclass to instantiate (called with no arguments).
        filename: Checkpoint filename relative to :data:`CHECKPOINT_DIR`.

    Returns:
        The module in eval mode with weights assigned from the safetensors file.
    """
    path = CHECKPOINT_DIR / filename
    logger.info(">> Restoring %s weights from: %s...", filename, path)
    data = safetensors.torch.load_file(path, device=str(torch.get_default_device()))
    with torch.device("meta"):
        model = module()
    model.load_state_dict(data, assign=True)
    return model.eval()


def load_feature_extractor() -> transformers.SeamlessM4TFeatureExtractor:
    """Load the SeamlessM4T feature extractor for semantic embedding extraction.

    Downloads (or uses a cached copy of) the ``facebook/w2v-bert-2.0`` feature extractor
    configuration from HuggingFace Hub.
    """
    return transformers.SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")


def load_unified_voice() -> UnifiedVoice:
    """Load the UnifiedVoice GPT model from the local safetensors checkpoint."""
    return _restore_model_weights(UnifiedVoice, GPT_FILE)


def load_campplus() -> CAMPPlus:
    """Load the CAMPPlus speaker-embedding model from the local safetensors checkpoint."""
    return _restore_model_weights(CAMPPlus, CAMPPLUS_FILE)


def load_tokenizer(normalizer: TextNormalizer) -> TextTokenizer:
    """Download the BPE vocabulary model from HuggingFace Hub and return a :class:`TextTokenizer`.

    Args:
        normalizer: Pre-initialized :class:`TextNormalizer` to attach to the tokenizer.
    """
    with Timer() as t:
        path = Path(hf.hf_hub_download("IndexTeam/IndexTTS-2", "bpe.model"))
        tokenizer = TextTokenizer(path, normalizer)

    logger.info(">> bpe model restored in %.2f seconds from: %s", t.elapsed, path)
    return tokenizer


def load_semantic_codec() -> RepCodec:
    """Load the semantic codec (RepCodec) from the local safetensors checkpoint."""
    return _restore_model_weights(RepCodec, SEMANTIC_CODEC_FILE)


def load_semantic_model() -> transformers.Wav2Vec2BertModel:
    """Download (or use cached) the Wav2Vec2Bert semantic model and move it to *device*."""
    with Timer() as t:
        model = transformers.Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
        model = model.eval().to(torch.get_default_device())
    logger.info(">> semantic_model weights restored in %.2f seconds", t.elapsed)
    return model


@beartype
def load_semantic_stats() -> tuple[Float[Tensor, "dim"], Float[Tensor, "dim"]]:
    """Download the w2v-bert2 mean/variance statistics and return ``(mean, std)`` tensors.

    These are used to normalise the semantic model's hidden-state features
    before feeding them into the downstream IndexTTS2 components.
    """
    with Timer() as t:
        path = hf.hf_hub_download("amphion/dualcodec", "w2vbert2_mean_var_stats_emilia.pt")
        raw = torch.load(path, map_location=torch.get_default_device())
        assert isinstance(raw, dict), f"Expected dict from {path}, got {type(raw).__name__}"
        data = cast(TensorDict, raw)
        mean = data["mean"]
        std = data["var"].sqrt()

    logger.info(">> semantic_mean and semantic_std weights restored in %.2f seconds from: %s", t.elapsed, path)
    return mean, std


def load_cfm() -> CFM:
    """Load the Conditional Flow Matching (CFM) mel-spectrogram decoder from the local safetensors checkpoint."""
    return _restore_model_weights(CFM, CFM_FILE)


def load_length_regulator() -> InterpolateRegulator:
    """Load the InterpolateRegulator from the local safetensors checkpoint."""
    return _restore_model_weights(InterpolateRegulator, LENGTH_REGULATOR_FILE)


def load_bigvgan(use_cuda_kernel: bool) -> BigVGAN:
    """Download (or use cached) the BigVGAN vocoder and move it to *device*.

    Args:
        use_cuda_kernel: Whether to enable the fused CUDA activation kernel.
    """
    with Timer() as t:
        model = BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel=use_cuda_kernel)
        model.remove_weight_norm()
        model = model.eval().to(torch.get_default_device())

    logger.info(">> bigvgan weights restored in %.2f seconds", t.elapsed)
    return model


if __name__ == "__main__":
    path = CHECKPOINT_DIR / GPT_FILE
    if not path.exists():
        pt_path = hf.hf_hub_download("IndexTeam/IndexTTS-2", filename="gpt.pth")
        data = torch.load(pt_path, map_location="cpu")
        data = cast(TensorDict, data)
        safetensors.torch.save_file(data, path)

    path = CHECKPOINT_DIR / SEMANTIC_CODEC_FILE
    if not path.exists():
        pt_path = hf.hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        data = safetensors.torch.load_file(pt_path)
        for k in list(data.keys()):
            if k.startswith("decoder."):
                del data[k]
        safetensors.torch.save_file(data, path)

    lr_path = CHECKPOINT_DIR / LENGTH_REGULATOR_FILE
    cfm_path = CHECKPOINT_DIR / CFM_FILE
    if not lr_path.exists() or not cfm_path.exists():
        path = hf.hf_hub_download("IndexTeam/IndexTTS-2", "s2mel.pth")
        s2mel_data = torch.load(path, map_location="cpu", weights_only=False)
        s2mel_data = cast(dict[str, dict[str, TensorDict]], s2mel_data)
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

    path = CHECKPOINT_DIR / CAMPPLUS_FILE
    if not path.exists():
        pt_path = hf.hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin", local_dir=CHECKPOINT_DIR)
        data = torch.load(pt_path, map_location="cpu")
        data = cast(TensorDict, data)
        safetensors.torch.save_file(data, path)
