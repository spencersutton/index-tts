import logging
from pathlib import Path

import safetensors.torch
from huggingface_hub import hf_hub_download

from indextts.config import CheckpointsConfig
from indextts.s2mel.modules.bigvgan import BigVGAN
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.model import MyModel
from indextts.utils.maskgct.models.codec.kmeans.repcodec_model import RepCodec

logger = logging.getLogger(__name__)


def load_bigvgan(name: str, use_cuda_kernel: bool) -> BigVGAN:
    model = BigVGAN.from_pretrained(name, use_cuda_kernel=use_cuda_kernel)
    model.remove_weight_norm()
    logger.info(f"bigvgan weights restored from: {name}")
    return model.eval()


def load_campplus() -> CAMPPlus:
    model = CAMPPlus()
    path = "checkpoints/campplus_cn_common.safetensors"
    safetensors.torch.load_model(model, path, strict=False)

    logger.info(f"campplus_model weights restored from: {path}")
    return model.eval()


def load_semantic_codec_model() -> RepCodec:
    checkpoint = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    model = RepCodec()
    safetensors.torch.load_model(model, checkpoint, strict=False)
    logger.info(f"semantic_codec weights restored from: {checkpoint}")
    return model.eval()


def load_s2mel_model(cfg: CheckpointsConfig, model_dir: Path) -> MyModel:
    model = MyModel(cfg.s2mel)

    safetensors.torch.load_model(model.cfm, model_dir / cfg.cfm_checkpoint, strict=False)
    model.cfm.eval()

    safetensors.torch.load_model(model.gpt_layer, model_dir / cfg.gpt_layer_checkpoint, strict=False)
    model.gpt_layer.eval()

    safetensors.torch.load_model(model.length_regulator, model_dir / cfg.len_reg_checkpoint, strict=False)
    model.length_regulator.eval()

    return model.eval()
