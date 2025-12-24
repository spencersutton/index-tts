"""IndexTTS v2 inference module for text-to-speech synthesis."""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import logging
import random
import time
import typing
from collections.abc import Collection, Generator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any, cast

import safetensors.torch
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
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
from indextts.s2mel.modules.model import MyModel
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.maskgct.models.codec.kmeans.repcodec_model import RepCodec

if typing.TYPE_CHECKING:
    import numpy as np
    from gradio import Progress

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    use_torch_compile: bool
    use_accel: bool
    dtype: torch.dtype | None
    stop_mel_token: int
    cfg: CheckpointsConfig

    # Models
    qwen_emo: QwenEmotion
    gpt: UnifiedVoice
    extract_features: SeamlessM4TFeatureExtractor
    semantic_model: Wav2Vec2BertModel
    semantic_codec: RepCodec
    s2mel: MyModel
    campplus_model: CAMPPlus
    bigvgan: BigVGAN
    tokenizer: TextTokenizer

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
        use_deepspeed: bool = False,
        use_accel: bool = False,
        use_torch_compile: bool = False,
    ) -> None:
        """Initialize IndexTTS2 synthesis engine.

        Args:
            cfg_path: Path to configuration YAML file
            model_dir: Directory containing model checkpoints
            use_fp16: Enable FP16 precision (not supported on CPU/MPS)
            device: Compute device (auto-detected if None)
            use_cuda_kernel: Use custom CUDA kernels for BigVGAN
            use_deepspeed: Enable DeepSpeed inference optimization
            use_accel: Enable flash attention acceleration
            use_torch_compile: Enable torch.compile optimization
        """
        # Configure device
        dev_cfg = DeviceConfig.auto_detect(device, use_fp16, use_cuda_kernel)
        self.device = dev_cfg.device
        self.use_fp16 = dev_cfg.use_fp16
        self.use_cuda_kernel = dev_cfg.use_cuda_kernel
        self.use_accel = use_accel
        self.use_torch_compile = use_torch_compile
        self.dtype = torch.float16 if self.use_fp16 else None
        self.gr_progress = None

        if self.device.startswith("cuda"):
            with contextlib.suppress(AttributeError):
                torch.set_float32_matmul_precision("high")

        # Load configuration
        self.cfg = CheckpointsConfig(**cast(Mapping[str, Any], OmegaConf.load(cfg_path)))
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        self.model_version = self.cfg.version

        # Load models
        self._load_models(model_dir, use_deepspeed)

        # Apply torch.compile if requested
        if use_torch_compile:
            self.s2mel.enable_torch_compile()

    # -------------------------------------------------------------------------
    # Initialization Helpers
    # -------------------------------------------------------------------------

    def _load_models(self, model_dir: Path, use_deepspeed: bool) -> None:
        """Load all model components."""
        cfg = self.cfg

        # Emotion model
        self.qwen_emo = QwenEmotion(model_dir / cfg.qwen_emo_path)

        # GPT model
        self.gpt = UnifiedVoice(use_accel=self.use_accel).to(self.device)
        gpt_path = model_dir / cfg.gpt_checkpoint
        safetensors.torch.load_model(self.gpt, gpt_path)
        self.gpt = self.gpt.eval()
        if self.use_fp16:
            self.gpt.half()
        logger.info(f"GPT weights restored from: {gpt_path}")

        # Initialize GPT inference mode
        use_deepspeed = self._check_deepspeed(use_deepspeed)
        self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=self.use_fp16)

        # Preload CUDA kernel if needed
        if self.use_cuda_kernel:
            self._preload_cuda_kernel()

        # Semantic models
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").eval().to(self.device)

        stat_mean_var = safetensors.safe_open(model_dir / cfg.w2v_stat, framework="pt", device=self.device)
        self.semantic_mean = stat_mean_var.get_tensor("mean")
        self.semantic_std = torch.sqrt(stat_mean_var.get_tensor("var"))

        # Semantic codec model
        checkpoint = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        self.semantic_codec = RepCodec()
        safetensors.torch.load_model(self.semantic_codec, checkpoint, strict=False)
        self.semantic_codec = self.semantic_codec.eval().to(self.device)
        logger.info(f"semantic_codec weights restored from: {checkpoint}")

        # S2Mel model
        self.s2mel = MyModel(cfg.s2mel)
        safetensors.torch.load_model(self.s2mel.cfm, model_dir / cfg.cfm_checkpoint)
        self.s2mel.cfm.eval()
        safetensors.torch.load_model(self.s2mel.gpt_layer, model_dir / cfg.gpt_layer_checkpoint)
        self.s2mel.gpt_layer.eval()
        safetensors.torch.load_model(self.s2mel.length_regulator, model_dir / cfg.len_reg_checkpoint)
        self.s2mel.length_regulator.eval()
        self.s2mel = self.s2mel.eval().to(self.device)
        if self.use_fp16:
            self.s2mel.half()

        # CAMPPlus model
        self.campplus_model = CAMPPlus()
        path = "checkpoints/campplus_cn_common.safetensors"
        safetensors.torch.load_model(self.campplus_model, path)
        # CAMPPlus is relatively small and only run once per prompt; keeping it on CPU
        # can save VRAM without materially affecting throughput.
        self.campplus_model = self.campplus_model.eval().cpu()
        logger.info(f"campplus_model weights restored from: {path}")

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

    def _check_deepspeed(self, use_deepspeed: bool) -> bool:
        """Check if DeepSpeed is available and requested."""
        if not use_deepspeed:
            return False

        try:
            if importlib.util.find_spec("deepspeed") is None:
                logger.info("DeepSpeed not found, falling back to normal inference")
                return False
        except (ImportError, OSError, CalledProcessError) as e:
            logger.info(f"Failed to load DeepSpeed: {e}")
            return False

        return True

    def _preload_cuda_kernel(self) -> None:
        """Preload custom CUDA kernel for BigVGAN."""
        try:
            from indextts.s2mel.modules.bigvgan.alias_free_activation.cuda import (  # noqa: PLC0415
                activation1d,
            )

            logger.info(
                f"Preloaded custom CUDA kernel: {activation1d.anti_alias_activation_cuda}",
            )
        except Exception as e:  # noqa: BLE001
            logger.info(f"Failed to load CUDA kernel, falling back to torch: {e}")
            self.use_cuda_kernel = False

    # -------------------------------------------------------------------------
    # Audio Prompt Processing
    # -------------------------------------------------------------------------

    def _get_emo_embedding(self, path: Path) -> Tensor:
        """Extract emotion embedding from audio file."""
        decoder = AudioDecoder(path, num_channels=1, sample_rate=SEMANTIC_SR)
        audio = decoder.get_samples_played_in_range(0, MAX_LEN)
        inputs = self.extract_features(audio.data, sampling_rate=audio.sample_rate, return_tensors="pt")
        return self.get_emb(inputs.to(self.device))

    @functools.lru_cache
    def process_audio_prompt(self, prompt: Path) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Process audio prompt to extract conditioning features.

        Args:
            prompt: Path to audio file

        Returns:
            Tuple of (spk_cond_emb, style, prompt_condition, ref_mel)
        """
        # Load audio at both sample rates
        decoder_22k = AudioDecoder(prompt, num_channels=1, sample_rate=OUTPUT_SR)
        audio_22k = decoder_22k.get_samples_played_in_range(0, MAX_LEN)

        decoder_16k = AudioDecoder(prompt, num_channels=1, sample_rate=SEMANTIC_SR)
        audio_16k = decoder_16k.get_samples_played_in_range(0, MAX_LEN)

        # Extract speaker conditioning embedding
        inputs = self.extract_features(audio_16k.data.cpu(), sampling_rate=audio_16k.sample_rate, return_tensors="pt")
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
        prompt_condition = self.s2mel.length_regulator(
            S_ref, ylens=torch.tensor([ref_mel.size(2)], device=self.device), n_quantizers=3, f0=None
        )[0]

        return spk_cond_emb, style, prompt_condition, ref_mel

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
            **generation_kwargs,
        )

        if stream_return:
            return gen

        try:
            return next(iter(gen))
        except (StopIteration, IndexError):
            return None

    @torch.inference_mode()
    def infer_generator(
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
        quick_streaming_tokens: int = 0,
        cfm_steps: int = 25,
        **generation_kwargs: Any,
    ) -> Generator[Tensor | Path | tuple[int, np.ndarray] | None]:
        """Generator-based inference for streaming synthesis.

        Args:
            cfm_steps: Number of CFM diffusion steps (default 25). Lower values are faster
                but may reduce quality. Values 15-25 are recommended.
        """
        logger.info("Starting inference...")
        self._set_gr_progress(0, "starting inference...")
        start_time = time.perf_counter()

        # Process emotion configuration
        emo_audio_prompt, emo_alpha, emo_vector = self._process_emotion_config(
            spk_audio_prompt, emo_audio_prompt, emo_alpha, emo_vector, use_emo_text, emo_text, text
        )

        # Load speaker conditioning
        spk_cond_emb, style, prompt_condition, ref_mel = self.process_audio_prompt(spk_audio_prompt)

        # Compute emotion matrix if using explicit vectors
        emovec_mat, weight_vector = self._compute_emo_matrix(emo_vector, style, use_random)

        # Get emotion conditioning embedding
        emo_cond_emb = self._get_emo_embedding(emo_audio_prompt)

        # Tokenize and segment text
        self._set_gr_progress(0.1, "text processing...")
        segments, batch_text_tokens = self._tokenize_and_segment(
            text, max_text_tokens_per_segment, quick_streaming_tokens
        )

        # Pre-calculate emotion vector
        emovec = self._compute_emovec(spk_cond_emb, emo_cond_emb, emo_alpha, emovec_mat, weight_vector)

        # Run batch inference
        max_mel_tokens = cast(int, generation_kwargs.pop("max_mel_tokens", 1500))

        yield from self._run_batch_inference(
            segments=segments,
            batch_text_tokens=batch_text_tokens,
            spk_cond_emb=spk_cond_emb,
            emo_cond_emb=emo_cond_emb,
            emovec=emovec,
            prompt_condition=prompt_condition,
            ref_mel=ref_mel,
            style=style,
            max_mel_tokens=max_mel_tokens,
            interval_silence=interval_silence,
            stream_return=stream_return,
            output_path=output_path,
            start_time=start_time,
            generation_kwargs=generation_kwargs,
            cfm_steps=cfm_steps,
        )

    # -------------------------------------------------------------------------
    # Inference Helpers
    # -------------------------------------------------------------------------

    def _process_emotion_config(
        self,
        spk_audio_prompt: Path,
        emo_audio_prompt: Path | None,
        emo_alpha: float,
        emo_vector: Collection[float] | None,
        use_emo_text: bool,
        emo_text: str | None,
        text: str,
    ) -> tuple[Path, float, Collection[float] | None]:
        """Process and normalize emotion configuration."""
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

        return emo_audio_prompt, emo_alpha, emo_vector

    def _compute_emo_matrix(
        self,
        emo_vector: Collection[float] | None,
        style: Tensor,
        use_random: bool,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Compute emotion matrix from explicit emotion vector."""
        if emo_vector is None:
            return None, None

        weight_vector = torch.tensor(list(emo_vector))

        # Select emotion indices
        if use_random:
            indices: list[int | Tensor] = [random.randint(0, n - 1) for n in self.emo_num]  # noqa: S311
        else:
            indices = [_find_most_similar_cosine(style, mat) for mat in self.spk_matrix]

        # Build weighted emotion matrix
        emo_vecs = [self.emo_matrix[i][idx].unsqueeze(0) for i, idx in enumerate(indices)]
        emo_mat = torch.cat(emo_vecs, dim=0)
        emovec_mat = (weight_vector.unsqueeze(1) * emo_mat).sum(dim=0, keepdim=True)

        return emovec_mat, weight_vector

    def _tokenize_and_segment(
        self,
        text: str,
        max_text_tokens_per_segment: int,
        quick_streaming_tokens: int,
    ) -> tuple[list[list[str]], list[Tensor]]:
        """Tokenize text and split into segments."""
        tokens = self.tokenizer.tokenize(text)
        segments = self.tokenizer.split_segments(
            tokens, max_text_tokens_per_segment, quick_streaming_tokens=quick_streaming_tokens
        )

        # Check for unknown tokens
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if self.tokenizer.unk_token_id in token_ids:
            unk_tokens = [t for t, tid in zip(tokens, token_ids) if tid == self.tokenizer.unk_token_id]
            logger.warning(f"Text contains {len(unk_tokens)} unknown tokens: {unk_tokens}")

        # Convert segments to tensors
        batch_tokens = [
            torch.tensor(self.tokenizer.convert_tokens_to_ids(seg), dtype=torch.int32, device=self.device)
            for seg in segments
        ]

        return segments, batch_tokens

    def _compute_emovec(
        self,
        spk_cond_emb: Tensor,
        emo_cond_emb: Tensor,
        emo_alpha: float,
        emovec_mat: Tensor | None,
        weight_vector: Tensor | None,
    ) -> Tensor:
        """Compute final emotion vector with optional matrix blending."""
        device = spk_cond_emb.device
        emovec = self.gpt.merge_emovec(
            spk_cond_emb,
            emo_cond_emb,
            torch.tensor([spk_cond_emb.shape[-1]], device=device),
            torch.tensor([emo_cond_emb.shape[-1]], device=device),
            alpha=emo_alpha,
        )

        if emovec_mat is not None and weight_vector is not None:
            emovec = emovec_mat + (1 - weight_vector.sum()) * emovec

        return emovec

    def _run_batch_inference(
        self,
        *,
        segments: list[list[str]],
        batch_text_tokens: list[Tensor],
        spk_cond_emb: Tensor,
        emo_cond_emb: Tensor,
        emovec: Tensor,
        prompt_condition: Tensor,
        ref_mel: Tensor,
        style: Tensor,
        max_mel_tokens: int,
        interval_silence: int,
        stream_return: bool,
        output_path: Path | None,
        start_time: float,
        generation_kwargs: dict[str, Any],
        cfm_steps: int = 25,
    ) -> Generator[Tensor | Path | tuple[int, np.ndarray] | None]:
        """Run batched inference over text segments.

        Args:
            cfm_steps: Number of CFM diffusion steps (default 25).
        """
        if not batch_text_tokens:
            return

        # Timing accumulators
        gpt_gen_time = 0.0
        gpt_forward_time = 0.0
        s2mel_time = 0.0
        bigvgan_time = 0.0
        wavs: list[Tensor] = []
        silence: Tensor | None = None
        has_warned = False

        # Pad batch
        text_tokens_batch = pad_sequence(
            batch_text_tokens, batch_first=True, padding_value=self.gpt.cfg.stop_text_token
        )
        batch_size = text_tokens_batch.size(0)

        # Generate mel codes
        t0 = time.perf_counter()
        codes_batch, speech_conditioning_latent = self.gpt.inference_speech(
            spk_cond_emb.expand(batch_size, -1, -1),
            text_tokens_batch,
            emo_cond_emb.expand(batch_size, -1, -1),
            cond_lengths=torch.tensor([spk_cond_emb.shape[-1]] * batch_size, device=spk_cond_emb.device),
            emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]] * batch_size, device=emo_cond_emb.device),
            emo_vec=emovec,
            do_sample=generation_kwargs.pop("do_sample", True),
            top_p=generation_kwargs.pop("top_p", 0.8),
            top_k=generation_kwargs.pop("top_k", 30),
            temperature=generation_kwargs.pop("temperature", 0.8),
            num_return_sequences=1,
            length_penalty=generation_kwargs.pop("length_penalty", 0.0),
            num_beams=generation_kwargs.pop("num_beams", 3),
            repetition_penalty=generation_kwargs.pop("repetition_penalty", 10.0),
            max_generate_length=max_mel_tokens,
            **generation_kwargs,
        )
        gpt_gen_time = time.perf_counter() - t0

        # Warn if generation was truncated
        if (codes_batch[:, -1] != self.stop_mel_token).any() and not has_warned:
            logger.warning(
                "Generation exceeded max_mel_tokens (%d). Consider adjusting parameters.",
                max_mel_tokens,
            )
            has_warned = True

        # Process each segment
        for seg_idx, code in enumerate(codes_batch):
            self._set_gr_progress(
                0.2 + 0.7 * seg_idx / len(segments),
                f"Synthesizing segment {seg_idx + 1}/{len(segments)}...",
            )

            wav, seg_gpt_time, seg_s2mel_time, seg_bigvgan_time = self._process_segment(
                code=code,
                text_tokens=batch_text_tokens[seg_idx],
                speech_conditioning_latent=speech_conditioning_latent[seg_idx : seg_idx + 1],
                emo_cond_emb=emo_cond_emb,
                emovec=emovec,
                spk_cond_emb=spk_cond_emb,
                prompt_condition=prompt_condition,
                ref_mel=ref_mel,
                style=style,
                cfm_steps=cfm_steps,
            )

            gpt_forward_time += seg_gpt_time
            s2mel_time += seg_s2mel_time
            bigvgan_time += seg_bigvgan_time
            wavs.append(wav.cpu())

            if stream_return:
                yield wav.cpu()
                if silence is None:
                    silence = generate_silence_interval(wavs, interval_silence, sample_rate=OUTPUT_SR).cpu()
                yield silence

        end_time = time.perf_counter()

        # Log timing stats
        self._log_inference_stats(
            gpt_gen_time, gpt_forward_time, s2mel_time, bigvgan_time, start_time, end_time, wavs, interval_silence
        )

        if stream_return:
            return

        # Save or return audio
        yield from self._finalize_audio(wavs, interval_silence, output_path)

    def _process_segment(
        self,
        *,
        code: Tensor,
        text_tokens: Tensor,
        speech_conditioning_latent: Tensor,
        emo_cond_emb: Tensor,
        emovec: Tensor,
        spk_cond_emb: Tensor,
        prompt_condition: Tensor,
        ref_mel: Tensor,
        style: Tensor,
        cfm_steps: int = 25,
    ) -> tuple[Tensor, float, float, float]:
        """Process a single segment to generate audio.

        Args:
            cfm_steps: Number of CFM diffusion steps.
        """
        # Trim code at stop token
        if self.stop_mel_token in code:
            stop_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
            code_len = stop_idx[0, 0].item() if stop_idx.numel() > 0 else len(code)
        else:
            code_len = len(code)

        code = code[:code_len].unsqueeze(0)
        text_tokens = text_tokens.unsqueeze(0)

        code_lens = torch.tensor([code_len], device=code.device)

        # GPT forward pass
        t0 = time.perf_counter()
        latent = self.gpt(
            speech_conditioning_latent,
            text_tokens,
            torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
            code,
            torch.tensor([code.shape[-1]], device=code.device),
            emo_cond_emb,
            cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=spk_cond_emb.device),
            emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=emo_cond_emb.device),
            emo_vec=emovec,
            use_speed=torch.zeros(spk_cond_emb.size(0), device=spk_cond_emb.device).long(),
        )
        gpt_time = time.perf_counter() - t0

        # S2Mel conversion
        t0 = time.perf_counter()
        latent = self.s2mel.gpt_layer(latent)

        S_infer = self.semantic_codec.quantizer.vq2emb(code.unsqueeze(1)).transpose(1, 2)
        S_infer += latent

        target_lengths = (code_lens * 1.72).long()

        cond = self.s2mel.length_regulator(S_infer, ylens=target_lengths, n_quantizers=3, f0=None)[0]
        cat_condition = torch.cat([prompt_condition, cond], dim=1)

        vc_target = self.s2mel.cfm.inference(
            cat_condition,
            torch.tensor([cat_condition.size(1)], device=cat_condition.device),
            ref_mel,
            style,
            None,
            cfm_steps,
            inference_cfg_rate=0.7,
        )
        vc_target = vc_target[:, :, ref_mel.size(-1) :]
        s2mel_time = time.perf_counter() - t0

        # BigVGAN vocoder
        t0 = time.perf_counter()
        wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0).squeeze(1)
        bigvgan_time = time.perf_counter() - t0

        return wav, gpt_time, s2mel_time, bigvgan_time

    def _log_inference_stats(
        self,
        gpt_gen_time: float,
        gpt_forward_time: float,
        s2mel_time: float,
        bigvgan_time: float,
        start_time: float,
        end_time: float,
        wavs: list[Tensor],
        interval_silence: int,
    ) -> None:
        """Log inference timing statistics."""
        self._set_gr_progress(0.9, "saving audio...")

        all_wavs = insert_interval_silence(wavs, interval_silence, sample_rate=OUTPUT_SR)
        # `insert_interval_silence()` and other helpers may create tensors on the
        # current default device (e.g., CUDA) even if `wavs` are on CPU.
        # Normalize to CPU before concatenation to avoid device-mismatch errors.
        all_wavs = [w.detach().cpu() for w in all_wavs]
        wav = torch.cat(all_wavs, dim=1)
        wav_length = wav.shape[-1] / OUTPUT_SR
        total_time = end_time - start_time

        logger.info(f"gpt_gen_time: {gpt_gen_time:.2f}s")
        logger.info(f"gpt_forward_time: {gpt_forward_time:.2f}s")
        logger.info(f"s2mel_time: {s2mel_time:.2f}s")
        logger.info(f"bigvgan_time: {bigvgan_time:.2f}s")
        logger.info(f"Total inference time: {total_time:.2f}s")
        logger.info(f"Generated audio length: {wav_length:.2f}s")
        logger.info(f"RTF: {total_time / wav_length if wav_length > 0 else 0:.4f}")

    def _finalize_audio(
        self,
        wavs: list[Tensor],
        interval_silence: int,
        output_path: Path | None,
    ) -> Generator[Path | tuple[int, np.ndarray]]:
        """Finalize and output generated audio."""
        all_wavs = insert_interval_silence(wavs, interval_silence, sample_rate=OUTPUT_SR)
        all_wavs = [w.detach().cpu() for w in all_wavs]
        wav = torch.cat(all_wavs, dim=1)

        if output_path:
            output_path.unlink(missing_ok=True)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            AudioEncoder(wav, sample_rate=OUTPUT_SR).to_file(output_path)
            logger.info(f"Audio saved to: {output_path}")
            yield output_path
        else:
            wav_data = (wav * torch.iinfo(torch.int16).max).type(torch.int16)
            yield (OUTPUT_SR, wav_data.numpy().T)
