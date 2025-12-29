"""IndexTTS v2 inference module for text-to-speech synthesis."""

from __future__ import annotations

import contextlib
import logging
import random
import typing
from collections.abc import Collection, Generator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, no_type_check

import safetensors.torch
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch import Tensor, nn
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
from transformers import BatchFeature, SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

from indextts.audio_utils import generate_silence_interval, insert_interval_silence
from indextts.config import CheckpointsConfig
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.qwen_emotion import QwenEmotion
from indextts.s2mel.modules.audio import mel_spectrogram
from indextts.s2mel.modules.bigvgan import BigVGAN
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.flow_matching import CFM
from indextts.s2mel.modules.length_regulator import InterpolateRegulator
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.maskgct.models.codec.kmeans.repcodec_model import RepCodec

if typing.TYPE_CHECKING:
    import numpy as np
    from gradio import Progress

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@no_type_check
def _strip_dead_functorch_predispatch_calls(gm: torch.fx.GraphModule) -> int:
    """Remove dead functorch predispatch calls that break torch.export.save().

    Certain PyTorch versions can leave dead (num_users=0) call_function nodes
    targeting internal functorch predispatch helpers, e.g.:

    - torch._functorch.predispatch.lazy_load_decompositions
    - torch._functorch.predispatch._vmap_increment_nesting

    The .pt2 serializer cannot serialize these Python functions, even when
    they're dead. We remove them if they have zero users.

    Returns:
        Number of nodes removed.
    """
    # Best-effort: trigger lazy decompositions load outside the graph so the
    # graph does not need to encode it as a side-effecting call.
    with contextlib.suppress(Exception):
        pred = getattr(getattr(torch, "_functorch", None), "predispatch", None)
        lazy_fn = getattr(pred, "lazy_load_decompositions", None)
        if callable(lazy_fn):
            lazy_fn()

    removed = 0
    graph = gm.graph
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if len(node.users) != 0:
            continue

        target = node.target
        mod = str(getattr(target, "__module__", ""))
        if mod.startswith(("torch._functorch.predispatch", "torch._functorch")):
            graph.erase_node(node)
            removed += 1

    if removed:
        graph.lint()
        gm.recompile()
    return removed


@no_type_check
def _safe_torch_export_save(program: Any, path: str | Path) -> None:
    """Save an ExportedProgram, retrying after sanitizing known bad nodes."""
    gm = getattr(program, "graph_module", None)
    # Retry a few times: stripping one dead node may reveal another.
    for _ in range(5):
        try:
            torch.export.save(program, path)
        except Exception as e:
            msg = str(e)
            if gm is None:
                raise
            if "Serializing <function" not in msg and "torch._functorch" not in msg:
                raise

            removed = _strip_dead_functorch_predispatch_calls(gm)
            if removed == 0:
                raise

    # If we fall out of the loop, re-raise with a final attempt for a clearer error.
    torch.export.save(program, path)


# =============================================================================
# Constants
# =============================================================================

OUTPUT_SR = 22050
SEMANTIC_SR = 16000
MAX_LEN = 15

# Emotion bias factors: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
_EMO_BIAS = (0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625)
_MAX_EMO_SUM = 0.8


# =============================================================================
# Device Detection
# =============================================================================


@dataclass
class DeviceConfig:
    """Configuration for compute device and precision."""

    device: str
    use_fp16: bool
    use_cuda_kernel: bool

    @classmethod
    def auto_detect(
        cls,
        device: str | None = None,
        use_fp16: bool = False,
        use_cuda_kernel: bool | None = None,
    ) -> DeviceConfig:
        """Auto-detect optimal device configuration.

        Args:
            device: Explicit device string or None for auto-detection
            use_fp16: Whether to use FP16 precision
            use_cuda_kernel: Whether to use custom CUDA kernels for BigVGAN

        Returns:
            DeviceConfig with optimal settings for the available hardware
        """
        if device is not None:
            return cls(
                device=device,
                use_fp16=False if device == "cpu" else use_fp16,
                use_cuda_kernel=(use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")),
            )

        # Auto-detect device
        if torch.cuda.is_available():
            return cls(
                device="cuda:0",
                use_fp16=use_fp16,
                use_cuda_kernel=use_cuda_kernel is None or use_cuda_kernel,
            )
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return cls(device="xpu", use_fp16=use_fp16, use_cuda_kernel=False)
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            # FP16 on MPS has overhead vs FP32
            return cls(device="mps", use_fp16=False, use_cuda_kernel=False)

        logger.info("Running on CPU - inference will be slow")
        return cls(device="cpu", use_fp16=False, use_cuda_kernel=False)


def _load_model[T: nn.Module](model: T, path: Path | str, device: str = "cpu") -> T:
    """Load model weights from safetensors file."""
    safetensors.torch.load_model(model, path, device=device, strict=False)
    logger.info(f"{model.__class__.__name__} weights restored from: {path}")
    return model.eval().to(device)


# =============================================================================
# Emotion Processing
# =============================================================================


def normalize_emo_vec(emo_vector: Sequence[float], apply_bias: bool = True) -> list[float]:
    """Normalize an emotion vector with optional bias and scaling.

    Args:
        emo_vector: Emotion intensity values in order:
            [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
        apply_bias: Whether to apply predefined bias factors

    Returns:
        Normalized emotion vector with sum capped at 0.8
    """
    result = list(emo_vector)

    # Apply bias to de-emphasize problematic emotions
    if apply_bias:
        result = [v * b for v, b in zip(result, _EMO_BIAS)]

    # Cap total sum at 0.8
    emo_sum = sum(result)
    if emo_sum > _MAX_EMO_SUM:
        scale = _MAX_EMO_SUM / emo_sum
        result = [v * scale for v in result]

    return result


def _find_most_similar_cosine(query_vector: Tensor, matrix: Tensor) -> Tensor:
    """Find the index of the most similar vector in matrix using cosine similarity."""
    similarities = torch.cosine_similarity(query_vector.float(), matrix.float(), dim=1)
    return torch.argmax(similarities)


# =============================================================================
# IndexTTS2 Main Class
# =============================================================================


class IndexTTS2:
    """IndexTTS v2 text-to-speech synthesis engine."""

    # Type annotations
    device: str
    use_fp16: bool
    use_cuda_kernel: bool
    use_accel: bool
    stop_mel_token: int
    cfg: CheckpointsConfig

    # Models
    qwen_emo: QwenEmotion
    gpt: UnifiedVoice
    extract_features: SeamlessM4TFeatureExtractor
    semantic_model: Wav2Vec2BertModel
    semantic_codec: RepCodec
    campplus_model: CAMPPlus
    bigvgan: BigVGAN
    tokenizer: TextTokenizer
    gpt_layer: nn.Sequential[nn.Module]
    cfm: CFM
    length_regulator: InterpolateRegulator

    # Tensors
    semantic_mean: Tensor
    semantic_std: Tensor
    emo_matrix: tuple[Tensor, ...]
    emo_num: tuple[int, ...]
    spk_matrix: tuple[Tensor, ...]

    if typing.TYPE_CHECKING:
        gr_progress: Progress | None
    model_version: float

    def __init__(
        self,
        cfg_path: Path = Path("checkpoints/config.yaml"),
        model_dir: Path = Path("checkpoints"),
        use_fp16: bool = False,
        device: str | None = None,
        use_cuda_kernel: bool | None = None,
        use_accel: bool = False,
    ) -> None:
        """Initialize IndexTTS2 synthesis engine.

        Args:
            cfg_path: Path to configuration YAML file
            model_dir: Directory containing model checkpoints
            use_fp16: Enable FP16 precision (not supported on CPU/MPS)
            device: Compute device (auto-detected if None)
            use_cuda_kernel: Use custom CUDA kernels for BigVGAN
            use_accel: Enable flash attention acceleration
        """
        # Configure device
        dev_cfg = DeviceConfig.auto_detect(device, use_fp16, use_cuda_kernel)
        self.device = dev_cfg.device
        self.use_fp16 = dev_cfg.use_fp16
        self.use_cuda_kernel = dev_cfg.use_cuda_kernel
        self.use_accel = use_accel
        self.gr_progress = None

        if self.device.startswith("cuda"):
            with contextlib.suppress(AttributeError):
                torch.set_float32_matmul_precision("high")

        # Load configuration
        self.cfg = CheckpointsConfig(**cast(Mapping[str, Any], OmegaConf.load(cfg_path)))  # pyright: ignore[reportAny]
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        self.model_version = self.cfg.version

        # Load models
        cfg = self.cfg

        # Emotion model
        self.qwen_emo = QwenEmotion(model_dir / cfg.qwen_emo_path)

        # GPT model
        self.gpt = _load_model(UnifiedVoice(use_accel=self.use_accel), model_dir / cfg.gpt_checkpoint, self.device)
        if self.use_fp16:
            self.gpt.half()

        self.gpt.post_init_gpt2_config(half=self.use_fp16)

        # Semantic models
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").eval().to(self.device)

        stat_mean_var = safetensors.safe_open(model_dir / cfg.w2v_stat, framework="pt", device=self.device)
        self.semantic_mean = stat_mean_var.get_tensor("mean")
        self.semantic_std = torch.sqrt(stat_mean_var.get_tensor("var"))

        # Semantic codec model
        checkpoint = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        self.semantic_codec = _load_model(RepCodec(), checkpoint, self.device)

        # S2Mel model
        self.cfm = _load_model(CFM(cfg.s2mel), model_dir / cfg.cfm_checkpoint, self.device)
        self.gpt_layer = _load_model(
            nn.Sequential(
                nn.Linear(1280, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 1024),
            ),
            model_dir / cfg.gpt_layer_checkpoint,
            self.device,
        )
        self.length_regulator = _load_model(
            InterpolateRegulator(
                channels=cfg.s2mel.length_regulator.channels,
                sampling_ratios=cfg.s2mel.length_regulator.sampling_ratios,
                in_channels=cfg.s2mel.length_regulator.in_channels,
            ),
            model_dir / cfg.len_reg_checkpoint,
            self.device,
        )
        if self.use_fp16:
            self.cfm.half()
            self.gpt_layer.half()
            self.length_regulator.half()

        # CAMPPlus model
        self.campplus_model = _load_model(CAMPPlus(), "checkpoints/campplus_cn_common.safetensors")

        # BigVGAN vocoder
        self.bigvgan = BigVGAN.from_pretrained(cfg.vocoder.name, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan.remove_weight_norm()
        self.bigvgan = self.bigvgan.eval().to(self.device)
        if self.use_fp16:
            self.bigvgan.half()
        logger.info(f"bigvgan weights restored from: {cfg.vocoder.name}")

        # Text processing
        normalizer = TextNormalizer()
        normalizer.load()
        self.tokenizer = TextTokenizer(model_dir / cfg.dataset.bpe_model, normalizer)
        logger.info("TextTokenizer loaded")

        # Emotion matrices
        emo_matrix = cast(Tensor, torch.load(model_dir / cfg.emo_matrix))
        spk_matrix = cast(Tensor, torch.load(model_dir / cfg.spk_matrix))
        self.emo_num = tuple(cfg.emo_num)
        self.emo_matrix = torch.split(emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(spk_matrix, self.emo_num)

    # -------------------------------------------------------------------------
    # Semantic Embedding
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def get_emb(self, features: BatchFeature[Tensor]) -> Tensor:
        """Extract semantic embedding from audio features."""
        vq_emb = self.semantic_model(
            input_features=features["input_features"],
            attention_mask=features["attention_mask"],
            output_hidden_states=True,
        )
        assert not isinstance(vq_emb, tuple) and vq_emb.hidden_states is not None
        feat = vq_emb.hidden_states[17]
        return (feat - self.semantic_mean) / self.semantic_std

    def _set_gr_progress(self, value: float, desc: str) -> None:
        """Update Gradio progress bar if available."""
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    # -------------------------------------------------------------------------
    # Inference Methods
    # -------------------------------------------------------------------------

    def infer(
        self,
        spk_audio_prompt: Path,
        text: str,
        output_path: Path | None,
        emo_audio_prompt: Path | None = None,
        emo_alpha: float = 1.0,
        emo_vector: Collection[float] | None = None,
        use_emo_text: bool = False,
        emo_text: str | None = None,
        use_random: bool = False,
        interval_silence: int = 200,
        max_text_tokens_per_segment: int = 120,
        stream_return: bool = False,
        verbose: bool = False,
        **generation_kwargs: object,
    ) -> Tensor | Generator[Tensor | Path | tuple[int, np.ndarray] | None] | Path | tuple[int, np.ndarray] | None:
        """Synthesize speech from text.

        Args:
            spk_audio_prompt: Path to speaker reference audio
            text: Text to synthesize
            output_path: Path to save output audio (or None)
            emo_audio_prompt: Optional emotion reference audio
            emo_alpha: Emotion blending factor (0-1)
            emo_vector: Optional explicit emotion vector
            use_emo_text: Derive emotion from text content
            emo_text: Text for emotion detection (defaults to main text)
            use_random: Use random emotion style selection
            interval_silence: Silence duration between segments (ms)
            max_text_tokens_per_segment: Max tokens per synthesis segment
            stream_return: If True, return generator for streaming
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated audio as Tensor, Path, or streaming generator
        """
        if verbose:
            logging.basicConfig(level=logging.DEBUG)

        gen = self.infer_generator(
            spk_audio_prompt=spk_audio_prompt,
            text=text,
            output_path=output_path,
            emo_audio_prompt=emo_audio_prompt,
            emo_alpha=emo_alpha,
            emo_vector=emo_vector,
            use_emo_text=use_emo_text,
            emo_text=emo_text,
            use_random=use_random,
            interval_silence=interval_silence,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            stream_return=stream_return,
            **generation_kwargs,  # type: ignore
        )

        if stream_return:
            return gen

        return next(iter(gen))

    def get_emo_vec(self, emb: Tensor) -> Tensor:
        encoded, mask = self.gpt.emo_conditioning_encoder(emb)
        conditioning = self.gpt.emo_perceiver_encoder(
            encoded,
            self.gpt.emo_cond_mask_pad(mask.squeeze(1)),
        ).squeeze(1)
        return self.gpt.emo_layer(self.gpt.emovec_layer(conditioning))

    @torch.inference_mode()
    def infer_generator(
        self,
        output_path: Path | None,
        spk_audio_prompt: Path,
        text: str,
        cfm_steps: int = 25,
        emo_alpha: float = 1.0,
        emo_audio_prompt: Path | None = None,
        emo_text: str | None = None,
        emo_vector: Collection[float] | None = None,
        interval_silence: int = 200,
        max_text_tokens_per_segment: int = 120,
        quick_streaming_tokens: int = 0,
        stream_return: bool = False,
        use_emo_text: bool = False,
        use_random: bool = False,
        **generation_kwargs: Any,  # pyright: ignore[reportAny]
    ) -> Generator[Tensor | Path | tuple[int, np.ndarray] | None]:
        """Generator-based inference for streaming synthesis.

        Args:
            cfm_steps: Number of CFM diffusion steps (default 25). Lower values are faster
                but may reduce quality. Values 15-25 are recommended.
        """
        logger.info("Starting inference...")
        self._set_gr_progress(0, "starting inference...")
        device = self.device

        # Process emotion configuration
        # Clear emo_audio_prompt if using text/vector guidance
        if use_emo_text or emo_vector is not None:
            emo_audio_prompt = None

        # Generate emotion vectors from text
        if use_emo_text:
            emo_text = emo_text or text
            emo_dict = self.qwen_emo.inference(emo_text)
            logger.info(f"Detected emotion from text: {emo_dict}")
            emo_vector = list(emo_dict.values())

        # Scale emotion vectors by alpha
        if emo_vector is not None:
            scale = max(0.0, min(1.0, emo_alpha))
            if scale != 1.0:
                emo_vector = [int(x * scale * 10000) / 10000 for x in emo_vector]
                logger.info(f"Scaled emotion vectors to {scale:.2f}x: {emo_vector}")

        # Use speaker audio as emotion reference if not specified
        if emo_audio_prompt is None:
            emo_audio_prompt = spk_audio_prompt
            emo_alpha = 1.0

        # Load audio at both sample rates
        decoder_22k = AudioDecoder(spk_audio_prompt, num_channels=1, sample_rate=OUTPUT_SR)
        audio_22k = decoder_22k.get_samples_played_in_range(0, MAX_LEN)

        decoder_16k = AudioDecoder(spk_audio_prompt, num_channels=1, sample_rate=SEMANTIC_SR)
        audio_16k = decoder_16k.get_samples_played_in_range(0, MAX_LEN)

        # Extract speaker conditioning embedding
        inputs = self.extract_features(audio_16k.data, sampling_rate=audio_16k.sample_rate, return_tensors="pt")
        spk_cond_emb = self.get_emb(inputs.to(self.device))
        _, S_ref = self.semantic_codec.quantize(spk_cond_emb)

        # Extract mel spectrogram
        ref_mel = mel_spectrogram(audio_22k.data.float())

        # Extract speaker style
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.data, num_mel_bins=80, dither=0, sample_frequency=SEMANTIC_SR
        )
        feat -= feat.mean(dim=0, keepdim=True)
        style = self.campplus_model(feat.unsqueeze(0)).to(self.device)

        # Generate prompt condition
        prompt_condition = self.length_regulator(S_ref, ylens=torch.tensor([ref_mel.size(2)], device=self.device))

        # Compute emotion matrix if using explicit vectors
        if emo_vector is None:
            emovec_mat, weight_vector = None, None
        else:
            weight_vector = torch.tensor(list(emo_vector))

            # Select emotion indices
            if use_random:
                indices = [random.randint(0, n - 1) for n in self.emo_num]  # noqa: S311
            else:
                indices = [_find_most_similar_cosine(style, mat) for mat in self.spk_matrix]

            # Build weighted emotion matrix
            emovec_mat = (
                weight_vector.unsqueeze(1)
                * torch.cat([self.emo_matrix[i][idx].unsqueeze(0) for i, idx in enumerate(indices)], dim=0)
            ).sum(dim=0, keepdim=True)

        # Tokenize and segment text
        self._set_gr_progress(0.1, "text processing...")

        tokens = self.tokenizer.tokenize(text)

        # Check for unknown tokens
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if self.tokenizer.unk_token_id in token_ids:
            unk_tokens = [t for t, tid in zip(tokens, token_ids) if tid == self.tokenizer.unk_token_id]
            logger.warning(f"Text contains {len(unk_tokens)} unknown tokens: {unk_tokens}")

        # Get emotion conditioning embedding
        audio = AudioDecoder(emo_audio_prompt, num_channels=1, sample_rate=SEMANTIC_SR).get_samples_played_in_range(
            0, MAX_LEN
        )
        emo_cond_emb = self.get_emb(
            self.extract_features(audio.data, sampling_rate=audio.sample_rate, return_tensors="pt").to(device)
        )

        # Pre-calculate emotion vector
        base_vec = self.get_emo_vec(spk_cond_emb)
        emovec = base_vec + emo_alpha * (self.get_emo_vec(emo_cond_emb) - base_vec)

        if emovec_mat is not None and weight_vector is not None:
            emovec = emovec_mat + (1 - weight_vector.sum()) * emovec

        # Run batch inference
        max_mel_tokens = cast(int, generation_kwargs.pop("max_mel_tokens", 1500))

        # Convert segments to tensors
        batch_text_tokens = [
            torch.tensor(self.tokenizer.convert_tokens_to_ids(seg), dtype=torch.int32, device=device)
            for seg in self.tokenizer.split_segments(
                tokens, max_text_tokens_per_segment, quick_streaming_tokens=quick_streaming_tokens
            )
        ]
        # Pad batch
        text_tokens_batch = nn.utils.rnn.pad_sequence(
            batch_text_tokens, batch_first=True, padding_value=self.gpt.config.stop_text_token
        )
        batch_size = text_tokens_batch.size(0)

        # Generate mel codes
        codes_batch, speech_conditioning_latent = self.gpt.inference_speech(
            speech_condition=spk_cond_emb.expand(batch_size, -1, -1),
            text_inputs=text_tokens_batch,
            emo_speech_condition=emo_cond_emb.expand(batch_size, -1, -1),
            emo_vec=emovec,
            do_sample=generation_kwargs.pop("do_sample", True),
            top_p=generation_kwargs.pop("top_p", 0.8),
            top_k=generation_kwargs.pop("top_k", 30),
            temperature=generation_kwargs.pop("temperature", 0.8),  # pyright: ignore[reportAny]
            num_return_sequences=1,
            length_penalty=generation_kwargs.pop("length_penalty", 0.0),
            num_beams=generation_kwargs.pop("num_beams", 3),  # pyright: ignore[reportAny]
            repetition_penalty=generation_kwargs.pop("repetition_penalty", 10.0),
            max_generate_length=max_mel_tokens,
            **generation_kwargs,  # pyright: ignore[reportAny]
        )

        # Warn if generation was truncated
        if (codes_batch[:, -1] != self.stop_mel_token).any():
            logger.warning(
                "Generation exceeded max_mel_tokens (%d). Consider adjusting parameters.",
                max_mel_tokens,
            )

        # Process each segment
        wavs: list[Tensor] = []
        silence: Tensor | None = None
        for seg_idx, code in enumerate(codes_batch):
            self._set_gr_progress(
                0.2 + 0.7 * seg_idx / len(codes_batch),
                f"Synthesizing segment {seg_idx + 1}/{len(codes_batch)}...",
            )

            # Trim code at stop token
            if self.stop_mel_token in code:
                stop_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                code_len = stop_idx[0, 0].item() if stop_idx.numel() > 0 else len(code)
            else:
                code_len = len(code)

            code = code[:code_len].unsqueeze(0)  # noqa: PLW2901

            # GPT forward pass
            latent = self.gpt(
                speech_conditioning_latent=speech_conditioning_latent[seg_idx : seg_idx + 1],
                text_inputs=batch_text_tokens[seg_idx].unsqueeze(0),
                mel_codes=code,
                emo_vec=emovec,
                use_speed=torch.zeros(spk_cond_emb.size(0), device=device).long(),
            )

            # S2Mel conversion
            cat_condition = torch.cat(
                [
                    prompt_condition,
                    self.length_regulator(
                        self.semantic_codec.quantizer.vq2emb(code.unsqueeze(1)).transpose(1, 2)
                        + self.gpt_layer(latent),
                        ylens=(torch.tensor([code_len], device=device) * 1.72).long(),
                    ),
                ],
                dim=1,
            )

            vc_target = self.cfm(
                cat_condition,
                torch.tensor([cat_condition.size(1)], device=device),
                ref_mel,
                style,
                cfm_steps,
                inference_cfg_rate=0.7,
            )
            vc_target = vc_target[:, :, ref_mel.size(-1) :]

            # BigVGAN vocoder
            wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0).squeeze(1)

            if stream_return:
                yield wav.cpu()
                if silence is None:
                    silence = generate_silence_interval(wavs, interval_silence, sample_rate=OUTPUT_SR).cpu()
                yield silence
            else:
                wavs.append(wav.cpu())

        self._set_gr_progress(0.9, "saving audio...")

        if stream_return:
            return

        # Save or return audio
        wav = torch.cat(
            [w.detach().cpu() for w in insert_interval_silence(wavs, interval_silence, sample_rate=OUTPUT_SR)], dim=1
        )

        if output_path:
            output_path.unlink(missing_ok=True)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            AudioEncoder(wav, sample_rate=OUTPUT_SR).to_file(output_path)
            logger.info(f"Audio saved to: {output_path}")
            yield output_path
        else:
            wav_data = (wav * torch.iinfo(torch.int16).max).type(torch.int16)
            yield (OUTPUT_SR, wav_data.numpy().T)
