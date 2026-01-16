import os
import random
import time
import warnings
from collections.abc import Callable, Sequence
from functools import cache, cached_property
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any, cast

import librosa
import safetensors.torch
import torch
import torch.nn.functional as F
import torchaudio
from bigvganinference import bigvgan
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch import Tensor
from transformers import SeamlessM4TFeatureExtractor

from indextts.config import IndexTTSConfig
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.qwen import QwenEmotion
from indextts.s2mel.modules.audio import mel_spectrogram
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.maskgct.models.codec.kmeans.repcodec_model import RepCodec
from indextts.utils.maskgct_utils import build_semantic_model

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"

CHECKPOINT_DIR = Path("checkpoints")
SAMPLING_RATE = 22050
MAX_AUDIO_LENGTH_SECONDS = 15
TARGET_SAMPLING_RATE = 16000


def normalize_emo_vec(vector: Sequence[float]) -> list[float]:
    # apply biased emotion factors for better user experience,
    # by de-emphasizing emotions that can cause strange results

    # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    biases = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
    vector = [vec * bias for vec, bias in zip(vector, biases)]

    # the total emotion sum must be 0.8 or less
    total = sum(vector)
    if total > 0.8:
        scale_factor = 0.8 / total
        vector = [vec * scale_factor for vec in vector]

    return list(vector)


def mel_fn(x: Tensor) -> Tensor:
    return mel_spectrogram(
        x,
        n_fft=1024,
        win_size=1024,
        hop_size=256,
        num_mels=80,
        sampling_rate=SAMPLING_RATE,
        fmin=0,
        fmax=None,
        center=False,
    )


@cache
def get_silence_interval(size: int, interval_silence: int = 200) -> Tensor:
    """Silences to be insert between generated segments."""

    return torch.zeros(size, (SAMPLING_RATE * interval_silence) // 1000)


def find_most_similar_cosine(query_vector: Tensor, matrix: Tensor) -> Tensor:
    query_vector = query_vector.float()
    matrix = matrix.float()

    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    return torch.argmax(similarities)


class IndexTTS2:
    @cached_property[RepCodec]
    def semantic_codec(self) -> RepCodec:
        model = RepCodec().eval()
        path = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        safetensors.torch.load_model(model, path, strict=False)
        model = model.to(self.device).eval()
        print(f">> semantic_codec weights restored from: {path}")
        return model

    def __init__(
        self,
        cfg_path: Path = CHECKPOINT_DIR / "config.yaml",
        model_dir: Path = CHECKPOINT_DIR,
        use_fp16: bool = False,
        device: str | None = None,
        use_cuda_kernel: bool | None = None,
        use_deepspeed: bool = False,
        use_accel: bool = False,
        use_torch_compile: bool = False,
    ) -> None:
        """
        Args:
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
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = cast(IndexTTSConfig, OmegaConf.load(cfg_path))
        self.dtype = torch.float16 if self.use_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        self.qwen_emo = QwenEmotion(self.cfg.qwen_emo_path)

        self.gpt = UnifiedVoice(**self.cfg.gpt, use_accel=use_accel)
        gpt_path = model_dir / self.cfg.gpt_checkpoint
        load_checkpoint(self.gpt, gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.use_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", gpt_path)

        if use_deepspeed:
            try:
                import deepspeed  # type: ignore  # noqa: F401
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                print(f">> Failed to load DeepSpeed. Falling back to normal inference. Error: {e}")

        self.gpt.post_init_gpt2_config(
            use_deepspeed=use_deepspeed, kv_cache=True, half=self.use_fp16, model_dim=self.cfg.gpt.model_dim
        )

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from bigvganinference.alias_free_activation.cuda import activation1d

                print(">> Preload custom CUDA kernel for BigVGAN", activation1d.anti_alias_activation_cuda)
            except Exception as e:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                print(f"{e!r}")
                self.use_cuda_kernel = False

        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(self.cfg.w2v_stat)
        self.semantic_model = self.semantic_model.to(self.device)
        self.semantic_model.eval()
        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)

        # Enable torch.compile optimization if requested
        if use_torch_compile:
            print(">> Enabling torch.compile optimization")
            self.s2mel.enable_torch_compile()
            print(">> torch.compile optimization enabled successfully")

        # load campplus_model
        campplus_ckpt_path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
        campplus_model = CAMPPlus()
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = campplus_model.to(self.device)
        self.campplus_model.eval()
        print(">> campplus_model weights restored from:", campplus_ckpt_path)

        bigvgan_name = self.cfg.vocoder.name
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", bigvgan_name)

        self.bpe_path = Path(hf_hub_download(**self.cfg.dataset.bpe_model))
        self.normalizer = TextNormalizer(enable_glossary=True)
        self.normalizer.load()
        print(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> bpe model loaded from:", self.bpe_path)

        # 加载术语词汇表（如果存在）
        self.glossary_path = os.path.join(self.model_dir, "glossary.yaml")
        if Path(self.glossary_path).exists():
            self.normalizer.load_glossary_from_yaml(self.glossary_path)
            print(">> Glossary loaded from:", self.glossary_path)

        emo_matrix = torch.load(hf_hub_download(repo_id="IndexTeam/IndexTTS-2", filename=self.cfg.emo_matrix))
        self.emo_matrix = emo_matrix.to(self.device)
        self.emo_num = list(self.cfg.emo_num)

        spk_matrix = torch.load(hf_hub_download(repo_id="IndexTeam/IndexTTS-2", filename=self.cfg.spk_matrix))
        self.spk_matrix = spk_matrix.to(self.device)

        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)

        # 缓存参考音频：
        self.cache_spk_cond: Tensor | None = None
        self.cache_s2mel_style: Tensor | None = None
        self.cache_s2mel_prompt: Tensor | None = None
        self.cache_spk_audio_prompt: Path | None = None
        self.cache_emo_cond: Tensor | None = None
        self.cache_emo_audio_prompt: Path | None = None
        self.cache_mel: Tensor | None = None

        # 进度引用显示（可选）
        self.gr_progress: Callable[..., None] | None = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

    @cached_property[MyModel]
    def s2mel(self) -> MyModel:
        assert isinstance(self.cfg.s2mel_checkpoint, str)
        path = Path(self.cfg.s2mel_checkpoint)
        model = load_checkpoint2(MyModel(), path).to(self.device)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        model.eval()
        print(">> s2mel weights restored from:", path)
        return model

    @torch.inference_mode()
    def get_emb(self, input_features: Tensor, attention_mask: Tensor) -> Tensor:
        vq_emb = self.semantic_model(
            input_features=input_features, attention_mask=attention_mask, output_hidden_states=True
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        return (feat - self.semantic_mean) / self.semantic_std

    def _set_gr_progress(self, value: float, desc: str) -> None:
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    def _load_and_cut_audio(
        self, audio_path: Path, verbose: bool = False, sample_rate: float | None = None
    ) -> tuple[Tensor, int]:
        if not sample_rate:
            audio, sample_rate = librosa.load(audio_path)
        else:
            audio, _ = librosa.load(audio_path, sr=sample_rate)
        audio = torch.tensor(audio).unsqueeze(0)
        max_audio_samples = int(MAX_AUDIO_LENGTH_SECONDS * sample_rate)

        if audio.shape[1] > max_audio_samples:
            if verbose:
                print(f"Audio too long ({audio.shape[1]} samples), truncating to {max_audio_samples} samples")
            audio = audio[:, :max_audio_samples]
        return audio, int(sample_rate)

    # 原始推理模式
    def infer(
        self,
        spk_audio_prompt: Path,
        text: str,
        output_path: Path,
        emo_audio_prompt: Path | None = None,
        emo_alpha: float = 1.0,
        emo_vector: Sequence[float] | None = None,
        use_emo_text: bool = False,
        emo_text: str | None = None,
        use_random: bool = False,
        interval_silence: int = 200,
        verbose: bool = False,
        max_text_tokens_per_segment: int = 120,
        stream_return: bool = False,
        more_segment_before: int = 0,
        **generation_kwargs: Any,
    ) -> Tensor | None:
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
            verbose,
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

    def infer_generator(
        self,
        spk_audio_prompt: Path,
        text: str,
        output_path: Path,
        emo_audio_prompt: Path | None = None,
        emo_alpha: float = 1.0,
        emo_vector: Sequence[float] | None = None,
        use_emo_text: bool = False,
        emo_text: str | None = None,
        use_random: bool = False,
        interval_silence: int = 200,
        verbose: bool = False,
        max_text_tokens_per_segment: int = 120,
        stream_return: bool = False,
        quick_streaming_tokens: int = 0,
        **generation_kwargs: Any,
    ) -> Tensor | None:
        print(">> starting inference...")
        self._set_gr_progress(0, "starting inference...")
        if verbose:
            print(
                f"origin text:{text}, spk_audio_prompt:{spk_audio_prompt}, "
                f"emo_audio_prompt:{emo_audio_prompt}, emo_alpha:{emo_alpha}, "
                f"emo_vector:{emo_vector}, use_emo_text:{use_emo_text}, "
                f"emo_text:{emo_text}"
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
            print(f"detected emotion vectors from text: {emo_dict}")
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
                print(f"scaled emotion vectors to {emo_vector_scale}x: {emo_vector}")

        if emo_audio_prompt is None:
            # we are not using any external "emotion reference voice"; use
            # speaker's voice as the main emotion reference audio.
            emo_audio_prompt = spk_audio_prompt
            # must always use alpha=1.0 when we don't have an external reference voice
            emo_alpha = 1.0

        # 如果参考音频改变了，才需要重新生成, 提升速度
        if (
            self.cache_spk_cond is None
            or self.cache_s2mel_prompt is None
            or self.cache_spk_audio_prompt != spk_audio_prompt
        ):
            if self.cache_spk_cond is not None:
                self.cache_spk_cond = None
                self.cache_s2mel_style = None
                self.cache_s2mel_prompt = None
                self.cache_mel = None
                torch.cuda.empty_cache()
            audio, sr = self._load_and_cut_audio(spk_audio_prompt, verbose)
            audio_22k: Tensor = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(audio)
            audio_16k: Tensor = torchaudio.transforms.Resample(sr, TARGET_SAMPLING_RATE)(audio)

            inputs = self.extract_features(audio_16k.tolist(), sampling_rate=TARGET_SAMPLING_RATE, return_tensors="pt")
            input_features = inputs["input_features"]
            attention_mask = inputs["attention_mask"]
            input_features = input_features.to(self.device)
            attention_mask = attention_mask.to(self.device)
            spk_cond_emb = self.get_emb(input_features, attention_mask)

            s_ref = self.semantic_codec.quantize(spk_cond_emb)
            ref_mel = mel_fn(audio_22k.to(spk_cond_emb.device).float())
            ref_target_lengths = torch.tensor([ref_mel.size(2)], dtype=torch.long).to(ref_mel.device)
            feat = torchaudio.compliance.kaldi.fbank(
                audio_16k.to(ref_mel.device), num_mel_bins=80, dither=0, sample_frequency=TARGET_SAMPLING_RATE
            )
            feat -= feat.mean(dim=0, keepdim=True)  # feat2另外一个滤波器能量组特征[922, 80]
            style = self.campplus_model(feat.unsqueeze(0))  # 参考音频的全局style2[1,192]

            prompt_condition = self.s2mel.length_regulator(s_ref, ylens=ref_target_lengths)[0]

            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_mel = ref_mel
        else:
            style = self.cache_s2mel_style
            prompt_condition = self.cache_s2mel_prompt
            spk_cond_emb = self.cache_spk_cond
            ref_mel = self.cache_mel

        weight_vector = None
        emovec_mat = None
        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector, device=self.device)
            if use_random:
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                assert style is not None
                random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]

            emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
            emo_matrix = torch.cat(emo_matrix, 0)
            emovec_mat = weight_vector.unsqueeze(1) * emo_matrix
            emovec_mat = torch.sum(emovec_mat, 0)
            emovec_mat = emovec_mat.unsqueeze(0)

        if self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_audio_prompt:
            if self.cache_emo_cond is not None:
                self.cache_emo_cond = None
                torch.cuda.empty_cache()
            emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt, verbose, sample_rate=TARGET_SAMPLING_RATE)
            emo_inputs = self.extract_features(
                emo_audio.tolist(), sampling_rate=TARGET_SAMPLING_RATE, return_tensors="pt"
            )
            emo_input_features = emo_inputs["input_features"]
            emo_attention_mask = emo_inputs["attention_mask"]
            emo_input_features = emo_input_features.to(self.device)
            emo_attention_mask = emo_attention_mask.to(self.device)
            emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)

            self.cache_emo_cond = emo_cond_emb
            self.cache_emo_audio_prompt = emo_audio_prompt
        else:
            emo_cond_emb = self.cache_emo_cond

        self._set_gr_progress(0.1, "text processing...")
        text_tokens_list = self.tokenizer.tokenize(text)
        segments = self.tokenizer.split_segments(
            text_tokens_list, max_text_tokens_per_segment, quick_streaming_tokens=quick_streaming_tokens
        )
        segments_count = len(segments)

        text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        if self.tokenizer.unk_token_id in text_token_ids:
            print(
                f"  >> Warning: input text contains {text_token_ids.count(self.tokenizer.unk_token_id)} unknown tokens (id={self.tokenizer.unk_token_id}):"
            )
            print(
                "     Tokens which can't be encoded: ",
                [t for t, id in zip(text_tokens_list, text_token_ids) if id == self.tokenizer.unk_token_id],
            )
            print("     Consider updating the BPE model or modifying the text to avoid unknown tokens.")

        if verbose:
            print("text_tokens_list:", text_tokens_list)
            print("segments count:", segments_count)
            print("max_text_tokens_per_segment:", max_text_tokens_per_segment)
            print(*segments, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)

        wavs: list[Tensor] = []
        gpt_gen_time: float = 0
        gpt_forward_time: float = 0
        s2mel_time: float = 0
        bigvgan_time: float = 0
        has_warned = False
        for seg_idx, sent in enumerate(segments):
            self._set_gr_progress(
                0.2 + 0.7 * seg_idx / segments_count, f"speech synthesis {seg_idx + 1}/{segments_count}..."
            )

            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
            if verbose:
                print(text_tokens)
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                # debug tokenizer
                text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                print("text_token_syms is same as segment tokens", text_token_syms == sent)

            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    emovec = self.gpt.merge_emo_vec(
                        spk_cond_emb,
                        emo_cond_emb,
                        torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        alpha=emo_alpha,
                    )

                    if weight_vector is not None and emovec_mat is not None:
                        emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec

                    codes, speech_conditioning_latent = self.gpt.inference_speech(
                        spk_cond_emb,
                        text_tokens,
                        emo_cond_emb,
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
                    assert isinstance(codes, Tensor)

                gpt_gen_time += time.perf_counter() - m_start_time
                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_segment`({max_text_tokens_per_segment}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning,
                    )
                    has_warned = True

                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)

                code_lens = []
                max_code_len = 0
                for code in codes:
                    if self.stop_mel_token not in code:
                        code_len = len(code)
                    else:
                        len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0]
                        code_len = len_[0].item() if len_.numel() > 0 else len(code)
                    code_lens.append(code_len)
                    max_code_len = max(int(max_code_len), int(code_len))
                codes = codes[:, :max_code_len]
                code_lens = torch.tensor(code_lens, dtype=torch.long)
                code_lens = code_lens.to(self.device)
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                m_start_time = time.perf_counter()
                use_speed = torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
                with torch.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = self.gpt(
                        speech_conditioning_latent,
                        text_tokens,
                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                        codes,
                        torch.tensor([codes.shape[-1]], device=text_tokens.device),
                        emo_cond_emb,
                        emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_vec=emovec,
                        use_speed=use_speed,
                    )
                    gpt_forward_time += time.perf_counter() - m_start_time

                dtype = None
                with torch.autocast(text_tokens.device.type, enabled=dtype is not None, dtype=dtype):
                    m_start_time = time.perf_counter()
                    latent = self.s2mel.gpt_layer(latent)
                    s_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
                    s_infer = s_infer.transpose(1, 2)
                    s_infer += latent
                    target_lengths = (code_lens * 1.72).long()

                    cond = self.s2mel.length_regulator(s_infer, ylens=target_lengths)[0]
                    cat_condition = torch.cat([prompt_condition, cond], dim=1)
                    assert ref_mel is not None and style is not None
                    vc_target = self.s2mel.cfm.inference(cat_condition, ref_mel, style)
                    vc_target = vc_target[:, :, ref_mel.size(-1) :]
                    s2mel_time += time.perf_counter() - m_start_time

                    m_start_time = time.perf_counter()
                    wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                if verbose:
                    print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                wavs.append(wav.cpu())  # to cpu before saving
                if stream_return:
                    yield wav.cpu()
                    yield get_silence_interval(wavs[0].size(0), interval_silence)
        end_time = time.perf_counter()

        self._set_gr_progress(0.9, "saving audio...")
        silence_tensor = get_silence_interval(wavs[0].size(0), interval_silence)
        # Insert silences between segments
        wavs = [item for x in wavs for item in (x, silence_tensor)][:-1]
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / SAMPLING_RATE
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> s2mel_time: {s2mel_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            if output_path.is_file():
                output_path.unlink()
                print(">> remove old wav file:", output_path)
            if output_path.parent != Path():
                output_path.parent.mkdir(exist_ok=True, parents=True)
            torchaudio.save(output_path, wav.type(torch.int16), SAMPLING_RATE)
            print(">> wav file saved to:", output_path)
            if stream_return:
                return None
            yield output_path
        else:
            if stream_return:
                return None
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            yield (SAMPLING_RATE, wav_data)
