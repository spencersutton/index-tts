import contextlib
import functools
import logging
import os
import random
import time
import typing
from collections.abc import Callable, Generator, Mapping
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any, cast

import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

from indextts.config import CheckpointsConfig
from indextts.gpt.model_v2 import GPT2InferenceModel, UnifiedVoice
from indextts.qwen_emotion import QwenEmotion
from indextts.s2mel.modules.audio import mel_spectrogram
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.commons import MyModel, load_checkpoint
from indextts.s2mel.modules.flow_matching import CFM
from indextts.s2mel.modules.length_regulator import InterpolateRegulator
from indextts.utils.checkpoint import load_checkpoint2
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.maskgct.models.codec.kmeans.repcodec_model import RepCodec

if typing.TYPE_CHECKING:
    from gradio import Progress

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SAMPLING_RATE = 22050


def _load_bigvgan(cfg: CheckpointsConfig, device: str, use_cuda_kernel: bool) -> "bigvgan.BigVGAN":
    name = cfg.vocoder.name
    model = bigvgan.BigVGAN.from_pretrained(name, use_cuda_kernel=use_cuda_kernel)
    model = model.to(device)
    model.remove_weight_norm()
    model.eval()
    logger.info(">> bigvgan weights restored from: %s", name)
    return model


def _load_s2mel(cfg: CheckpointsConfig, device: str, model_dir: Path) -> MyModel:
    s2mel_path = model_dir / cfg.s2mel_checkpoint
    model = MyModel(cfg.s2mel, use_gpt_latent=True)
    model = load_checkpoint(model, s2mel_path).to(device)
    assert model.cfm.estimator is not None
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    logger.info(">> s2mel weights restored from: %s", s2mel_path)
    return model.eval()


def _load_semantic_model(device: str) -> Wav2Vec2BertModel:
    model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    assert isinstance(model, Wav2Vec2BertModel)
    return model.to(device).eval()


def _load_semantic_codec(device: str) -> RepCodec:
    model = RepCodec().eval()
    path = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(model, path, strict=False)
    model = model.to(device).eval()
    logger.info(">> semantic_codec weights restored from: %s", path)
    return model


def generate_silence_interval(
    wavs: list[Tensor],
    interval_silence: int = 200,
) -> Tensor:
    """Silences to be insert between generated segments."""
    assert interval_silence > 0, "interval_silence must be greater than 0"
    assert len(wavs) > 0, "wavs list must not be empty"

    # get channel_size
    channel_size = wavs[0].size(0)
    # get silence tensor
    sil_dur = int(SAMPLING_RATE * interval_silence / 1000.0)
    return torch.zeros(channel_size, sil_dur)


def insert_interval_silence(
    wavs: list[Tensor],
    interval_silence: int = 200,
) -> list[Tensor]:
    """Insert silences between generated segments.
    wavs: List[torch.tensor]
    """
    if not wavs or interval_silence <= 0:
        return wavs

    # get channel_size
    channel_size = wavs[0].size(0)
    # get silence tensor
    sil_dur = int(SAMPLING_RATE * interval_silence / 1000.0)
    sil_tensor = torch.zeros(channel_size, sil_dur)

    wavs_list: list[Tensor] = []
    for i, wav in enumerate(wavs):
        wavs_list.append(wav)
        if i < len(wavs) - 1:
            wavs_list.append(sil_tensor)

    return wavs_list


def _load_and_cut_audio(
    audio_path: Path,
    max_audio_length_seconds: float,
    sr: int | None = None,
) -> tuple[Tensor, int]:
    samples = AudioDecoder(audio_path).get_all_samples()
    orig_sr = samples.sample_rate
    audio = samples.data

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr is not None and orig_sr != sr:
        audio = torchaudio.functional.resample(audio, orig_sr, sr)
    else:
        sr = orig_sr
    max_audio_samples = int(max_audio_length_seconds * sr)

    if audio.shape[1] > max_audio_samples:
        logger.debug("Audio too long (%d samples), truncating to %d samples", audio.shape[1], max_audio_samples)
        audio = audio[:, :max_audio_samples]
    return audio, sr


def _load_campplus_weights(device: str) -> CAMPPlus:
    path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model = model.to(device).eval()
    logger.info(">> campplus_model weights restored from: %s", path)
    return model


def normalize_emo_vec(emo_vector: list[float], apply_bias: bool = True) -> list[float]:
    """
    Normalizes an emotion vector by applying optional bias factors and scaling the sum.

    Args:
        emo_vector (list[float]): A list of emotion intensity values, typically in the order:
            [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm].
        apply_bias (bool, optional): Whether to apply predefined bias factors to de-emphasize
            certain emotions. Defaults to True.

    Returns:
        list[float]: The normalized emotion vector, possibly biased and scaled so that the sum
            does not exceed 0.8.
    """
    # apply biased emotion factors for better user experience,
    # by de-emphasizing emotions that can cause strange results
    if apply_bias:
        # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]  # noqa: ERA001
        emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
        emo_vector = [vec * bias for vec, bias in zip(emo_vector, emo_bias)]

    # the total emotion sum must be 0.8 or less
    emo_sum = sum(emo_vector)
    if emo_sum > 0.8:
        scale_factor = 0.8 / emo_sum
        emo_vector = [vec * scale_factor for vec in emo_vector]

    return emo_vector


class IndexTTS2:
    device: str
    use_fp16: bool
    cfg: CheckpointsConfig
    model_dir: Path
    dtype: torch.dtype | None
    stop_mel_token: int
    qwen_emo: QwenEmotion
    gpt: "UnifiedVoice"
    extract_features: SeamlessM4TFeatureExtractor
    semantic_model: Wav2Vec2BertModel
    semantic_mean: Tensor
    semantic_std: Tensor
    semantic_codec: RepCodec
    s2mel: MyModel
    campplus_model: CAMPPlus
    bigvgan: "bigvgan.BigVGAN"
    tokenizer: TextTokenizer
    emo_matrix: tuple[Tensor, ...]
    emo_num: list[int]
    spk_matrix: tuple[Tensor, ...]
    mel_fn: Callable[[Tensor], Tensor]

    if typing.TYPE_CHECKING:
        gr_progress: Progress | None
    model_version: float

    @functools.lru_cache  # noqa: B019
    def process_audio(self, prompt: Path) -> Tensor:
        audio, _ = _load_and_cut_audio(prompt, 15, sr=16000)
        inputs = self.extract_features(audio, sampling_rate=16000, return_tensors="pt")  # ty:ignore[invalid-argument-type]
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        return self.get_emb(input_features, attention_mask)

    @functools.lru_cache  # noqa: B019
    def process_audio_prompt(self, prompt: Path) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        audio, sr = _load_and_cut_audio(prompt, 15)
        audio_22k = torchaudio.transforms.Resample(sr, 22050).forward(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000).forward(audio)

        inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")  # ty:ignore[invalid-argument-type]
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        spk_cond_emb = self.get_emb(input_features, attention_mask)

        _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
        ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
        ref_target_lengths = Tensor([ref_mel.size(2)]).to(ref_mel.device)
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(ref_mel.device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat -= feat.mean(dim=0, keepdim=True)  # feat2另外一个滤波器能量组特征[922, 80]
        style = self.campplus_model(feat.unsqueeze(0))  # 参考音频的全局style2[1,192]

        length_regulator = cast(InterpolateRegulator, self.s2mel.models["length_regulator"])
        prompt_condition = length_regulator(S_ref, ylens=ref_target_lengths, n_quantizers=3, f0=None)[0]

        return (spk_cond_emb, style, prompt_condition, ref_mel)

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
        """Args:
        cfg_path (str): path to the config file.
        model_dir (str): path to the model directory.
        use_fp16 (bool): whether to use fp16.
        device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
        use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
        use_deepspeed (bool): whether to use DeepSpeed or not.
        use_accel (bool): whether to use acceleration engine for GPT2 or not.
        use_torch_compile (bool): whether to use torch.compile for optimization or not.

        """
        if device is not None:
            self.device = device
            self.use_fp16 = False if device == "cpu" else use_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = "xpu"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = False
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.use_fp16 = False  # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.use_fp16 = False
            self.use_cuda_kernel = False
            logger.info(">> Be patient, it may take a while to run in CPU mode.")

        if self.device.startswith("cuda"):
            with contextlib.suppress(AttributeError):
                torch.set_float32_matmul_precision("high")

        # Configure torch.compile inductor optimizations
        if use_torch_compile:
            # Enable persistent cache to avoid recompilation between runs
            import torch._dynamo.config as dynamo_config  # noqa: PLC0415, PLC2701

            cache_dir = model_dir / ".torch_compile_cache"
            cache_dir.mkdir(exist_ok=True, parents=True)

            # Set environment variables for caching (must be set before any compilation)
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))
            os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")  # Enable FX graph caching

            # Suppress verbose compilation logs (set to True for debugging)
            dynamo_config.verbose = False
            # Cache compiled graphs to disk
            dynamo_config.cache_size_limit = 256  # Increase cache size for large models

            if self.device.startswith("cuda"):
                try:
                    import torch._inductor.config as inductor_config  # noqa: PLC0415, PLC2701

                    # Enable CUDA graphs for reduced kernel launch overhead
                    inductor_config.triton.cudagraphs = True
                    # Fuse more operations for better performance
                    inductor_config.coordinate_descent_tuning = True
                    # Enable frozen weights optimization (inference-only)
                    inductor_config.freezing = True
                    # Enable FX graph caching (persists compiled graphs to disk)
                    inductor_config.fx_graph_cache = True
                    # Remote cache for distributed setups (local file-based)
                    inductor_config.fx_graph_remote_cache = False
                except (ImportError, AttributeError):
                    pass  # Older PyTorch versions may not have these options

            logger.info(">> torch.compile cache directory: %s", cache_dir)

        cfg = cast(Mapping[str, Any], OmegaConf.load(cfg_path))
        self.cfg = cast(CheckpointsConfig, cfg)  # pyright: ignore[reportInvalidCast]
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.use_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        self.use_accel = use_accel
        self.use_torch_compile = use_torch_compile

        self.qwen_emo = QwenEmotion(self.model_dir / self.cfg.qwen_emo_path)

        self.gpt = UnifiedVoice(use_accel=self.use_accel)
        self.gpt_path = self.model_dir / self.cfg.gpt_checkpoint
        load_checkpoint2(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.use_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        logger.info(">> GPT weights restored from: %s", self.gpt_path)

        if use_deepspeed:
            try:
                import importlib.util  # noqa: PLC0415

                if importlib.util.find_spec("deepspeed") is None:
                    use_deepspeed = False
                    logger.info(">> DeepSpeed not found. Falling back to normal inference.")
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                logger.info(">> Failed to load DeepSpeed. Falling back to normal inference. Error: %s", e)

        self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=self.use_fp16)

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.s2mel.modules.bigvgan.alias_free_activation.cuda import activation1d  # noqa: PLC0415

                logger.info(
                    ">> Preload custom CUDA kernel for BigVGAN: %s",
                    activation1d.anti_alias_activation_cuda,
                )
            except Exception as e:  # noqa: BLE001
                logger.info(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch. %r", e)
                self.use_cuda_kernel = False

        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

        self.semantic_model = _load_semantic_model(self.device)

        stat_mean_var = torch.load(self.model_dir / self.cfg.w2v_stat)
        self.semantic_mean = stat_mean_var["mean"].to(self.device)
        self.semantic_std = torch.sqrt(stat_mean_var["var"]).to(self.device)

        self.semantic_codec = _load_semantic_codec(self.device)

        self.s2mel = _load_s2mel(self.cfg, self.device, self.model_dir)

        # load campplus_model
        self.campplus_model = _load_campplus_weights(self.device)

        self.bigvgan = _load_bigvgan(self.cfg, self.device, self.use_cuda_kernel)

        normalizer = TextNormalizer()
        normalizer.load()
        logger.info(">> TextNormalizer loaded")

        bpe_path = self.model_dir / self.cfg.dataset.bpe_model
        self.tokenizer = TextTokenizer(bpe_path, normalizer)
        logger.info(">> bpe model loaded from: %s", bpe_path)

        emo_matrix: Tensor = torch.load(self.model_dir / self.cfg.emo_matrix)
        emo_matrix = emo_matrix.to(self.device)

        spk_matrix: Tensor = torch.load(self.model_dir / self.cfg.spk_matrix)
        spk_matrix = spk_matrix.to(self.device)

        self.emo_num = list(self.cfg.emo_num)
        self.emo_matrix = torch.split(emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(spk_matrix, self.emo_num)

        spect_params = self.cfg.s2mel.preprocess_params.spect_params
        mel_fn_args = {
            "n_fft": spect_params.n_fft,
            "win_size": spect_params.win_length,
            "hop_size": spect_params.hop_length,
            "num_mels": spect_params.n_mels,
            "sampling_rate": self.cfg.s2mel.preprocess_params.sr,
            "fmin": spect_params.fmin or 0,
            "fmax": None if spect_params.fmax == "None" else 8000,
            "center": False,
        }
        self.mel_fn = functools.partial(mel_spectrogram, **mel_fn_args)

        # Enable torch.compile optimization if requested
        if use_torch_compile:
            logger.info(">> Enabling torch.compile optimization")
            self.s2mel.enable_torch_compile()

            # Compile the inner inference model used for AR generation
            # This is critical because inference_speech() bypasses self.gpt()
            self.gpt.inference_model = cast(
                GPT2InferenceModel,
                torch.compile(self.gpt.inference_model, dynamic=True),
            )

            self.gpt = cast(UnifiedVoice, torch.compile(self.gpt))

            # Compile BigVGAN only when not using custom CUDA kernels
            # Custom CUDA kernels conflict with torch.compile tracing
            if not self.use_cuda_kernel:
                self.bigvgan = cast(
                    bigvgan.BigVGAN,
                    torch.compile(self.bigvgan, dynamic=True),
                )

            self.semantic_model = cast(
                Wav2Vec2BertModel,
                torch.compile(self.semantic_model, dynamic=True),
            )

            # Compile semantic codec (RepCodec) for quantization operations
            self.semantic_codec = cast(RepCodec, torch.compile(self.semantic_codec, dynamic=True))

            # CAMPPlus is a small model - use reduce-overhead mode for lower kernel launch latency
            self.campplus_model = cast(
                CAMPPlus,
                torch.compile(self.campplus_model, dynamic=True, mode="reduce-overhead"),
            )

            logger.info(">> torch.compile optimization enabled successfully")

        # 进度引用显示（可选）
        # Progress reference display (optional)
        self.gr_progress = None
        self.model_version = self.cfg.version

    @torch.inference_mode()
    def get_emb(self, input_features: Tensor, attention_mask: Tensor) -> Tensor:
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        return (feat - self.semantic_mean) / self.semantic_std

    def _set_gr_progress(self, value: float, desc: str) -> None:
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    # 原始推理模式
    # Original inference mode
    def infer(
        self,
        spk_audio_prompt: Path,
        text: str,
        output_path: Path | None,
        emo_audio_prompt: Path | None = None,
        emo_alpha: float = 1.0,
        emo_vector: list[float] | None = None,
        use_emo_text: bool = False,
        emo_text: str | None = None,
        use_random: bool = False,
        interval_silence: int = 200,
        max_text_tokens_per_segment: int = 120,
        stream_return: bool = False,
        more_segment_before: int = 0,
        **generation_kwargs: Any,  # noqa: ANN401
    ) -> Tensor | Generator[Tensor | Path | tuple[int, np.ndarray] | None] | Path | tuple[int, np.ndarray] | None:
        gen = self.infer_generator(
            spk_audio_prompt,
            text,
            output_path,
            emo_audio_prompt,
            emo_alpha,
            emo_vector,
            use_emo_text,
            emo_text,
            use_random,
            interval_silence,
            max_text_tokens_per_segment,
            stream_return,
            more_segment_before,
            **generation_kwargs,
        )
        if stream_return:
            return gen

        try:
            return next(iter(gen))
        except IndexError:
            return None

    def combine_weighted_styles(self, vector: list[float], style: Tensor, use_random: bool = False) -> Tensor:
        weight_vector = torch.tensor(vector, device=self.device)
        if use_random:
            random_index = [random.randint(0, x - 1) for x in self.emo_num]  # noqa: S311
        else:
            random_index = [_find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]

        matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
        matrix = torch.cat(matrix, 0)
        vector_matrix = weight_vector.unsqueeze(1) * matrix
        return torch.sum(vector_matrix, 0).unsqueeze(0)

    @torch.inference_mode()
    def infer_generator(
        self,
        spk_audio_prompt: Path,
        text: str,
        output_path: Path | None,
        emo_audio_prompt: Path | None = None,
        emo_alpha: float = 1.0,
        emo_vector: list[float] | None = None,
        use_emo_text: bool = False,
        emo_text: str | None = None,
        use_random: bool = False,
        interval_silence: int = 200,
        max_text_tokens_per_segment: int = 120,
        stream_return: bool = False,
        quick_streaming_tokens: int = 0,
        **generation_kwargs: Any,  # noqa: ANN401
    ) -> Generator[Tensor | Path | tuple[int, np.ndarray] | None]:
        # Mark CUDA graph step begin at the start of each inference
        # This tells PyTorch it's safe to reuse CUDA graph buffers
        # Must be called in the worker thread before any compiled models run
        if self.use_torch_compile and torch.cuda.is_available():
            torch.compiler.cudagraph_mark_step_begin()

        logger.info(">> starting inference...")
        self._set_gr_progress(0, "starting inference...")
        logger.debug(
            "origin text:%s, spk_audio_prompt:%s, emo_audio_prompt:%s, emo_alpha:%s, emo_vector:%s, use_emo_text:%s, emo_text:%s",
            text,
            spk_audio_prompt,
            emo_audio_prompt,
            emo_alpha,
            emo_vector,
            use_emo_text,
            emo_text,
        )
        start_time = time.perf_counter()

        if use_emo_text or emo_vector is not None:
            # we're using a text or emotion vector guidance; so we must remove
            # "emotion reference voice", to ensure we use correct emotion mixing!
            emo_audio_prompt = None

        if use_emo_text:
            # automatically generate emotion vectors from text prompt
            if emo_text is None:
                emo_text = text  # use main text prompt
            emo_dict = self.qwen_emo.inference(emo_text)
            logger.info("detected emotion vectors from text: %s", emo_dict)
            # convert ordered dict to list of vectors; the order is VERY important!
            emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            # we have emotion vectors; they can't be blended via alpha mixing
            # in the main inference process later, so we must pre-calculate
            # their new strengths here based on the alpha instead!
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:
                # scale each vector and truncate to 4 decimals (for nicer printing)
                emo_vector = [int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector]
                logger.info("scaled emotion vectors to %sx: %s", emo_vector_scale, emo_vector)

        if emo_audio_prompt is None:
            # we are not using any external "emotion reference voice"; use
            # speaker's voice as the main emotion reference audio.
            emo_audio_prompt = spk_audio_prompt
            # must always use alpha=1.0 when we don't have an external reference voice
            emo_alpha = 1.0

        spk_cond_emb, style, prompt_condition, ref_mel = self.process_audio_prompt(spk_audio_prompt)

        if emo_vector is not None:
            emovec_mat = self.combine_weighted_styles(emo_vector, style, use_random)

        emo_cond_emb = self.process_audio(emo_audio_prompt)

        self._set_gr_progress(0.1, "text processing...")
        text_tokens_list = self.tokenizer.tokenize(text)
        segments = self.tokenizer.split_segments(
            text_tokens_list,
            max_text_tokens_per_segment,
            quick_streaming_tokens=quick_streaming_tokens,
        )
        segments_count = len(segments)

        text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        if self.tokenizer.unk_token_id in text_token_ids:
            logger.warning(
                "input text contains %d unknown tokens (id=%d): %s\n"
                "     Tokens which can't be encoded: %s\n"
                "     Consider updating the BPE model or modifying the text to avoid unknown tokens.",
                text_token_ids.count(self.tokenizer.unk_token_id),
                self.tokenizer.unk_token_id,
                text_tokens_list,
                [
                    t
                    for t, id in zip(text_tokens_list, text_token_ids, strict=False)
                    if id == self.tokenizer.unk_token_id
                ],
            )

        logger.debug("text_tokens_list: %s", text_tokens_list)
        logger.debug("segments count: %d", segments_count)
        logger.debug("max_text_tokens_per_segment: %d", max_text_tokens_per_segment)
        logger.debug(*segments)
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        emovec_mat = None
        weight_vector = None

        # [OPTIMIZATION] Pre-calculate emovec once before the loop
        with (
            torch.inference_mode(),
            torch.autocast(
                torch.device(self.device).type,
                enabled=self.dtype is not None,
                dtype=self.dtype,
            ),
        ):
            emovec = self.gpt.merge_emovec(
                spk_cond_emb,
                emo_cond_emb,
                torch.tensor([spk_cond_emb.shape[-1]], device=self.device),
                torch.tensor([emo_cond_emb.shape[-1]], device=self.device),
                alpha=emo_alpha,
            )

            if emo_vector is not None and weight_vector is not None:
                emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec

        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        has_warned = False
        silence = None  # for stream_return

        # [OPTIMIZATION] Batch processing for inference_speech
        batch_text_tokens = []
        for sent in segments:
            tt = self.tokenizer.convert_tokens_to_ids(sent)
            batch_text_tokens.append(torch.tensor(tt, dtype=torch.int32, device=self.device))

        if not batch_text_tokens:
            # Handle empty segments if necessary
            pass
        else:
            # Pad with stop_text_token (which is ignored by the model)
            text_tokens_batch = pad_sequence(
                batch_text_tokens,
                batch_first=True,
                padding_value=self.gpt.stop_text_token,
            )

            logger.debug("Batch text tokens shape: %s", text_tokens_batch.shape)

            m_start_time = time.perf_counter()
            with (
                torch.inference_mode(),
                torch.autocast(
                    text_tokens_batch.device.type,
                    enabled=self.dtype is not None,
                    dtype=self.dtype,
                ),
            ):
                # Expand conditions to match batch size
                batch_size = text_tokens_batch.size(0)
                spk_cond_emb_batch = spk_cond_emb.expand(batch_size, -1, -1)
                emo_cond_emb_batch = emo_cond_emb.expand(batch_size, -1, -1)

                codes_batch, speech_conditioning_latent = self.gpt.inference_speech(
                    spk_cond_emb_batch,
                    text_tokens_batch,
                    emo_cond_emb_batch,
                    cond_lengths=torch.tensor(
                        [spk_cond_emb.shape[-1]] * batch_size,
                        device=self.device,
                    ),
                    emo_cond_lengths=torch.tensor(
                        [emo_cond_emb.shape[-1]] * batch_size,
                        device=self.device,
                    ),
                    emo_vec=emovec,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=autoregressive_batch_size,
                    length_penalty=length_penalty,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    max_generate_length=max_mel_tokens,
                    **generation_kwargs,
                )
            gpt_gen_time += time.perf_counter() - m_start_time

            if not has_warned and (codes_batch[:, -1] != self.stop_mel_token).any():
                logger.warning(
                    "WARN: generation stopped due to exceeding `max_mel_tokens` (%s). "
                    "Consider reducing `max_text_tokens_per_segment`(%s) or increasing `max_mel_tokens`.",
                    max_mel_tokens,
                    max_text_tokens_per_segment,
                )
                has_warned = True

            # Process each segment result
            for seg_idx, code in enumerate(codes_batch):
                self._set_gr_progress(
                    0.2 + 0.7 * seg_idx / segments_count,
                    f"speech synthesis {seg_idx + 1}/{segments_count}...",
                )

                # Trim code
                if self.stop_mel_token not in code:
                    code_len = len(code)
                else:
                    len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0]
                    code_len = len_[0].item() if len_.numel() > 0 else len(code)

                code = code[:code_len].unsqueeze(0)  # (1, S)  # noqa: PLW2901
                code_lens = torch.LongTensor([code_len]).to(self.device)

                # Get corresponding text tokens for this segment (unpadded)
                text_tokens = batch_text_tokens[seg_idx].unsqueeze(0)  # (1, L)

                logger.debug("Segment %s: code len %s", seg_idx, code_len)

                m_start_time = time.perf_counter()
                with (
                    torch.inference_mode(),
                    torch.autocast(
                        text_tokens.device.type,
                        enabled=self.dtype is not None,
                        dtype=self.dtype,
                    ),
                ):
                    use_speed = torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
                    latent = self.gpt(
                        speech_conditioning_latent[seg_idx : seg_idx + 1],
                        text_tokens,
                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                        code,
                        torch.tensor([code.shape[-1]], device=text_tokens.device),
                        emo_cond_emb,
                        cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_vec=emovec,
                        use_speed=use_speed,
                    )
                    gpt_forward_time += time.perf_counter() - m_start_time

                dtype = None
                with torch.autocast(
                    text_tokens.device.type,
                    enabled=dtype is not None,
                    dtype=dtype,
                ):
                    m_start_time = time.perf_counter()
                    diffusion_steps = 25
                    inference_cfg_rate = 0.7
                    latent = self.s2mel.models["gpt_layer"](latent)
                    S_infer = self.semantic_codec.quantizer.vq2emb(code.unsqueeze(1))
                    S_infer = S_infer.transpose(1, 2)
                    S_infer += latent
                    target_lengths = (code_lens * 1.72).long()

                    length_regulator = cast(
                        InterpolateRegulator,
                        self.s2mel.models["length_regulator"],
                    )
                    cond = length_regulator(S_infer, ylens=target_lengths, n_quantizers=3, f0=None)[0]
                    cat_condition = torch.cat([prompt_condition, cond], dim=1)
                    cfm = cast(CFM, self.s2mel.models["cfm"])
                    assert ref_mel is not None
                    assert style is not None
                    vc_target = cfm.inference(
                        cat_condition,
                        torch.LongTensor([cat_condition.size(1)]).to(cond.device),
                        ref_mel,
                        style,
                        None,
                        diffusion_steps,
                        inference_cfg_rate=inference_cfg_rate,
                    )
                    assert ref_mel is not None
                    vc_target = vc_target[:, :, ref_mel.size(-1) :]
                    s2mel_time += time.perf_counter() - m_start_time

                    m_start_time = time.perf_counter()
                    wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                    logger.debug(wav.shape)
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                logger.debug(
                    "wav shape: %s min: %s max: %s",
                    wav.shape,
                    wav.min(),
                    wav.max(),
                )
                wavs.append(wav.cpu())  # to cpu before saving
                if stream_return:
                    yield wav.cpu()
                    if silence is None:
                        silence = generate_silence_interval(
                            wavs,
                            interval_silence=interval_silence,
                        )
                    yield silence
        end_time = time.perf_counter()

        self._set_gr_progress(0.9, "saving audio...")
        wavs = insert_interval_silence(wavs, interval_silence=interval_silence)
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / SAMPLING_RATE
        logger.info(">> gpt_gen_time: %.2f seconds", gpt_gen_time)
        logger.info(">> gpt_forward_time: %.2f seconds", gpt_forward_time)
        logger.info(">> s2mel_time: %.2f seconds", s2mel_time)
        logger.info(">> bigvgan_time: %.2f seconds", bigvgan_time)
        logger.info(">> Total inference time: %.2f seconds", end_time - start_time)
        logger.info(">> Generated audio length: %.2f seconds", wav_length)
        logger.info(">> RTF: %.4f", (end_time - start_time) / wav_length)

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            save_to_file(output_path, wav, SAMPLING_RATE)

            if stream_return:
                return None
            yield output_path
        else:
            if stream_return:
                return None
            # 返回以符合Gradio的格式要求
            # Scale to int16 range for Gradio compatibility
            wav_data = (wav * torch.iinfo(torch.int16).max).type(torch.int16)
            wav_data = wav_data.numpy().T
            yield (SAMPLING_RATE, wav_data)


def save_to_file(output_path: Path, wav: Tensor, sampling_rate: int) -> None:
    # 直接保存音频到指定路径中
    # Directly save audio to the specified path
    if output_path.is_file():
        output_path.unlink()
        logger.info(">> remove old wav file: %s", output_path)
    if not output_path.parent.exists():
        output_path.parent.mkdir(exist_ok=True, parents=True)

    assert wav.dtype == torch.float32
    assert wav.ndim == 2

    encoder = AudioEncoder(wav, sample_rate=sampling_rate)
    encoder.to_file(output_path)
    logger.info(">> wav file saved to: %s", output_path)


def _find_most_similar_cosine(query_vector: Tensor, matrix: Tensor) -> Tensor:
    query_vector = query_vector.float()
    matrix = matrix.float()

    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    return torch.argmax(similarities)
