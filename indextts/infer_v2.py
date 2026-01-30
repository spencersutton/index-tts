import os
import random
import warnings
from collections.abc import Callable, Generator, Mapping, Sequence
from functools import cache, cached_property, lru_cache
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any, cast

import huggingface_hub as hf
import librosa
import safetensors.torch
import torch
import torch.nn.functional as F
import torchaudio
import transformers
from bigvganinference import bigvgan
from jaxtyping import Float, Int
from torch import Tensor
from torchcodec.encoders import AudioEncoder

from indextts.config import IndexTTSConfig
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.qwen import QwenEmotion
from indextts.s2mel.modules.audio import mel_spectrogram
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.commons import MyModel
from indextts.util import Timer
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.repcodec_model import RepCodec

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"

MAX_AUDIO_LENGTH_SECONDS = 15
EMO_NUM = [3, 17, 2, 8, 4, 5, 10, 24]


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


@cache
def get_silence_interval(size: int, interval_silence: int = 200, sampling_rate: int = 22050) -> Tensor:
    """Silences to be insert between generated segments."""

    return torch.zeros(size, (sampling_rate * interval_silence) // 1000)


def find_most_similar_cosine(query_vector: Float[Tensor, "1 C"], matrix: Float[Tensor, "N C"]) -> int:
    query_vector = query_vector.float()
    matrix = matrix.float()

    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    return int(torch.argmax(similarities))


def _load_and_cut_audio(audio_path: Path, sample_rate: float | None = None) -> tuple[Tensor, int]:
    if not sample_rate:
        audio, sample_rate = librosa.load(audio_path)
    else:
        audio, _ = librosa.load(audio_path, sr=sample_rate)
    audio = torch.tensor(audio).unsqueeze(0)
    assert audio.dim() == 2 and audio.size(0) == 1, "Only mono audio is supported."
    max_audio_samples = int(MAX_AUDIO_LENGTH_SECONDS * sample_rate)

    if audio.shape[1] > max_audio_samples:
        audio = audio[:, :max_audio_samples]
    return audio, int(sample_rate)


class IndexTTS2:
    cfg: IndexTTSConfig
    dtype: torch.dtype
    device: torch.device
    use_fp16: bool
    use_cuda_kernel: bool
    use_accel: bool
    stop_mel_token: int

    emo_matrix: tuple[Tensor, ...]
    spk_matrix: tuple[Tensor, ...]

    glossary_path: Path

    # 进度引用显示（可选）
    gr_progress: Callable[..., None] | None = None
    model_version: int | None

    has_warned: bool = False

    def generate_voice_conversion(
        self,
        code_lens: list[int],
        prompt_condition: Float[Tensor, "B T C"],
        style: Float[Tensor, "B C"],
        ref_mel: Float[Tensor, "B N T"],
        codes: Int[Tensor, "B T"],
        latent: Float[Tensor, "B T C"],
    ) -> Tensor:
        semantic_inference = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
        semantic_inference = semantic_inference.mT + self.s2mel.gpt_layer(latent)
        target_lengths = (torch.tensor(code_lens, device=self.device) * 1.72).long()

        cond = self.s2mel.length_regulator.__call__(semantic_inference, ylens=target_lengths)
        cond = torch.cat([prompt_condition, cond], dim=1)
        target = self.s2mel.cfm.inference(cond, ref_mel, style)
        return target[:, :, ref_mel.size(-1) :]

    @lru_cache(5)  # noqa: B019
    def extract_emotion_features(self, prompt: Path) -> Tensor:
        print(">> extracting emotion features from prompt:", prompt)
        audio, _ = _load_and_cut_audio(prompt, sample_rate=16000)
        inputs = self.extract_features(audio.numpy(), sampling_rate=16000, return_tensors="pt")
        inputs = inputs.to(self.device)
        return self.get_emb(inputs["input_features"], inputs["attention_mask"])

    def generate_emotion_matrix(
        self, weight_vector: Float[Tensor, "emo"], style: Float[Tensor, "B C"], use_random: bool = False
    ) -> Tensor:
        if use_random:
            index = [random.randint(0, x - 1) for x in EMO_NUM]
        else:
            index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]

        matrix = [x[index].unsqueeze(0) for index, x in zip(index, self.emo_matrix)]
        matrix = torch.cat(matrix, 0)
        matrix = weight_vector.unsqueeze(1) * matrix
        matrix = torch.sum(matrix, 0)
        return matrix.unsqueeze(0)

    def get_matrix(self, filename: str) -> tuple[Tensor, ...]:
        path = hf.hf_hub_download(repo_id="IndexTeam/IndexTTS-2", filename=filename)
        data = torch.load(path, map_location=self.device)
        return torch.split(data, EMO_NUM)

    @cached_property[UnifiedVoice]
    def gpt(self) -> UnifiedVoice:
        with Timer() as T:
            path = hf.hf_hub_download("IndexTeam/IndexTTS-2", filename="gpt.pth")
            data = torch.load(path, map_location=self.device, mmap=True)

            with torch.device("meta"):
                model = UnifiedVoice(cfg=self.cfg.gpt, use_accel=self.use_accel)
            model.load_state_dict(data, assign=True)
            model = model.eval()

            if self.use_fp16:
                model = model.half()

        print(f">> GPT weights restored in {T:.2f} seconds from: {path}")
        return model

    @cached_property[QwenEmotion]
    def qwen_emo(self) -> QwenEmotion:
        return QwenEmotion(self.cfg.qwen_emo_path)

    @cached_property[TextNormalizer]
    def normalizer(self) -> TextNormalizer:  # noqa: PLR6301
        normalizer = TextNormalizer()
        normalizer.load()
        return normalizer

    @cached_property[TextTokenizer]
    def tokenizer(self) -> TextTokenizer:
        with Timer() as T:
            path = Path(hf.hf_hub_download(repo_id=self.cfg.dataset.repo_id, filename=self.cfg.dataset.filename))
            tokenizer = TextTokenizer(path, self.normalizer)

        print(f">> bpe model restored in {T:.2f} seconds from: {path}")
        return tokenizer

    @cached_property[CAMPPlus]
    def campplus_model(self) -> CAMPPlus:
        with Timer() as T:
            path = hf.hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
            data = torch.load(path, map_location=self.device)

            model = CAMPPlus()
            model.load_state_dict(data)
            model = model.eval().to(self.device)

        print(f">> campplus_model weights restored in {T:.2f} seconds from: {path}")
        return model

    @cached_property[bigvgan.BigVGAN]
    def bigvgan(self) -> bigvgan.BigVGAN:
        with Timer() as T:
            # Simpler but slower version
            if False:
                model = bigvgan.BigVGAN.from_pretrained(  # pyright: ignore[reportUnreachable]
                    "nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel=self.use_cuda_kernel
                )
                model.remove_weight_norm()
                model = model.eval().to(self.device)

            path = hf.hf_hub_download(repo_id=self.cfg.vocoder.repo_id, filename=self.cfg.vocoder.filename)
            data = torch.load(path, map_location=self.device, mmap=True)

            json_path = hf.hf_hub_download(self.cfg.vocoder.repo_id, filename="config.json")
            hparams = bigvgan.load_hparams_from_json(json_path)

            with torch.device("meta"):
                model = bigvgan.BigVGAN(h=hparams)
            model.load_state_dict(data["generator"], assign=True)

            model = model.eval()
            model.remove_weight_norm()

        print(f">> bigvgan weights restored in {T:.2f} seconds from: {path}")
        return model

    @cached_property[RepCodec]
    def semantic_codec(self) -> RepCodec:
        with Timer() as T:
            path = hf.hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")

            model = RepCodec()
            missing, unexpected = safetensors.torch.load_model(model, path, strict=False)
            if missing:
                print(f">> semantic_codec missing keys: {missing}")
            if unexpected:
                print(f">> semantic_codec unexpected keys: {unexpected}")
            model = model.eval().to(self.device)
        print(f">> semantic_codec weights restored from: {path} in {T:.2f} seconds")
        return model

    @cached_property[transformers.Wav2Vec2BertModel]
    def semantic_model(self) -> transformers.Wav2Vec2BertModel:
        with Timer() as T:
            model = transformers.Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
            model = model.eval().to(self.device)
        print(f">> semantic_model weights restored in {T:.2f} seconds")
        return model

    @cached_property[Tensor]
    def semantic_mean(self) -> Tensor:
        with Timer() as T:
            path = hf.hf_hub_download(repo_id=self.cfg.w2v_stat.repo_id, filename=self.cfg.w2v_stat.filename)
            data = torch.load(path)
            data = data["mean"].to(self.device)
        print(f">> semantic_mean weights restored in {T:.2f} seconds from: {path}")
        return data

    @cached_property[Tensor]
    def semantic_std(self) -> Tensor:
        with Timer() as T:
            path = hf.hf_hub_download(repo_id=self.cfg.w2v_stat.repo_id, filename=self.cfg.w2v_stat.filename)
            data = torch.load(path)
            data = torch.sqrt(data["var"]).to(self.device)
        print(f">> semantic_std weights restored in {T:.2f} seconds from: {path}")
        return data

    @cached_property[MyModel]
    def s2mel(self) -> MyModel:
        with Timer() as T:
            path = hf.hf_hub_download("IndexTeam/IndexTTS-2", filename="s2mel.pth")
            data = torch.load(path, map_location=self.device, mmap=True)
            params = data["net"]

            with torch.device("meta"):
                model = MyModel()
            model.cfm.load_state_dict(params["cfm"], strict=False, assign=True)
            model.length_regulator.load_state_dict(params["length_regulator"], strict=False, assign=True)
            model.gpt_layer.load_state_dict(params["gpt_layer"], assign=True)
            model = model.eval()

        print(f">> s2mel weights restored in {T:.2f} seconds: {path}")
        return model

    @cached_property[transformers.SeamlessM4TFeatureExtractor]
    def extract_features(self) -> transformers.SeamlessM4TFeatureExtractor:  # noqa: PLR6301
        return transformers.SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    def __init__(
        self,
        model_dir: Path = Path("checkpoints"),
        use_fp16: bool = False,
        device: str | None = None,
        use_cuda_kernel: bool = False,
        use_deepspeed: bool = False,
        use_accel: bool = False,
        use_torch_compile: bool = False,
    ) -> None:
        """
        Args:
            model_dir (str): path to the model directory.
            use_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
            use_deepspeed (bool): whether to use DeepSpeed or not.
            use_accel (bool): whether to use acceleration engine for GPT2 or not.
            use_torch_compile (bool): whether to use torch.compile for optimization or not.
        """

        self.device = (
            torch.device(device) if device else torch.accelerator.current_accelerator() or torch.get_default_device()
        )
        self.use_cuda_kernel = use_cuda_kernel and str(self.device).startswith("cuda")
        self.use_fp16 = use_fp16 and self.device not in ["cpu", "mps"]
        self.cfg = IndexTTSConfig()
        self.dtype = torch.float16 if self.use_fp16 else torch.get_default_dtype()
        self.use_accel = use_accel

        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        if use_deepspeed:
            try:
                import deepspeed  # type: ignore  # noqa: F401
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                print(f">> Failed to load DeepSpeed. Falling back to normal inference. Error: {e}")

        self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, half=self.use_fp16)

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from bigvganinference.alias_free_activation.cuda import activation1d

                print(">> Preload custom CUDA kernel for BigVGAN", activation1d.anti_alias_activation_cuda)
            except Exception as e:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                print(f"{e!r}")
                self.use_cuda_kernel = False

        # Enable torch.compile optimization if requested
        if use_torch_compile:
            print(">> Enabling torch.compile optimization")
            self.s2mel.enable_torch_compile()
            print(">> torch.compile optimization enabled successfully")

        self.emo_matrix = self.get_matrix(self.cfg.emo_matrix)
        self.spk_matrix = self.get_matrix(self.cfg.spk_matrix)

        # 加载术语词汇表（如果存在）
        self.glossary_path = model_dir / "glossary.yaml"
        if self.glossary_path.exists():
            self.normalizer.load_glossary_from_yaml(self.glossary_path)
            print(">> Glossary loaded from:", self.glossary_path)

        self.model_version = int(self.cfg.version)

    @torch.inference_mode()
    def get_emb(self, input_features: Float[Tensor, "B T f"], attention_mask: Int[Tensor, "B T"]) -> Tensor:
        vq_emb = self.semantic_model(
            input_features=input_features, attention_mask=attention_mask, output_hidden_states=True
        )
        assert not isinstance(vq_emb, tuple) and vq_emb.hidden_states is not None
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        return (feat - self.semantic_mean) / self.semantic_std

    def _set_gr_progress(self, value: float, desc: str) -> None:
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    # 原始推理模式
    def infer(
        self,
        output_path: Path,
        spk_audio_prompt: Path,
        text: str,
        emo_alpha: float = 1.0,
        emo_audio_prompt: Path | None = None,
        emo_text: str | None = None,
        emo_vector: Sequence[float] | None = None,
        interval_silence: int = 200,
        max_text_tokens_per_segment: int = 120,
        more_segment_before: int = 0,
        stream_return: bool = False,
        use_emo_text: bool = False,
        use_random: bool = False,
        **generation_kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ) -> Path | Generator[Tensor] | None:
        if use_emo_text or emo_vector is not None:
            # we're using a text or emotion vector guidance; so we must remove
            # "emotion reference voice", to ensure we use correct emotion mixing!
            emo_audio_prompt = None

        if use_emo_text:
            # automatically generate emotion vectors from text prompt
            emo_text = emo_text or text  # use main text prompt
            emo_dict = self.qwen_emo.inference(emo_text)
            print(f"detected emotion vectors from text: {emo_dict}")
            # convert ordered dict to list of vectors; the order is VERY important!
            emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            # we have emotion vectors; they can'T be blended via alpha mixing
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
            # must always use alpha=1.0 when we don'T have an external reference voice
            emo_alpha = 1.0

        gen = self.infer_generator(
            spk_audio_prompt,
            text,
            output_path,
            emo_audio_prompt,
            emo_alpha,
            emo_vector,
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
            return next(iter(gen))  # pyright: ignore[reportReturnType]
        except IndexError:
            return None

    @lru_cache  # noqa: B019
    def extract_audio_features(self, prompt: Path) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        print(">> extracting audio features from prompt:", prompt)
        audio, sr = _load_and_cut_audio(prompt)
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        audio_22k = torchaudio.functional.resample(audio, sr, 22050)

        mel = mel_spectrogram(audio_22k, sample_rate=22050)
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(self.device), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat -= feat.mean(dim=0, keepdim=True)  # feat2另外一个滤波器能量组特征[922, 80]
        style = self.campplus_model(feat.unsqueeze(0))  # 参考音频的全局style2[1,192]

        inputs = cast(
            Mapping[str, Tensor],
            self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt").to(self.device),
        )

        embedding = self.get_emb(inputs["input_features"], inputs["attention_mask"])
        prompt_condition = self.s2mel.length_regulator.__call__(
            self.semantic_codec.quantize(embedding), ylens=torch.tensor([mel.size(2)], device=self.device)
        )
        return prompt_condition, style, mel, embedding

    @torch.inference_mode()
    def infer_generator(
        self,
        spk_audio_prompt: Path,
        text: str,
        output_path: Path,
        emo_audio_prompt: Path | None = None,
        emo_alpha: float = 1.0,
        emo_vector: Sequence[float] | None = None,
        use_random: bool = False,
        interval_silence: int = 200,
        max_text_tokens_per_segment: int = 120,
        stream_return: bool = False,
        quick_streaming_tokens: int = 0,
        **generation_kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ) -> Generator[Tensor]:
        print(">> starting inference...")
        self._set_gr_progress(0, "starting inference...")
        inference_timer = Timer()
        inference_timer.start()

        prompt_condition, style, ref_mel, speaker_conditioning_embedding = self.extract_audio_features(spk_audio_prompt)

        weight_vector = None
        emotion_matrix = None
        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector, device=self.device)
            emotion_matrix = self.generate_emotion_matrix(weight_vector, style, use_random=use_random)

        emotion_conditioning_embedding = self.extract_emotion_features(emo_audio_prompt)

        self._set_gr_progress(0.1, "text processing...")
        text_tokens_list = self.tokenizer.tokenize(text)
        segments = self.tokenizer.split_segments(
            text_tokens_list, max_text_tokens_per_segment, quick_streaming_tokens=quick_streaming_tokens
        )

        text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        if self.tokenizer.unk_token_id in text_token_ids:
            print(
                f"  >> Warning: input text contains {text_token_ids.count(self.tokenizer.unk_token_id)} unknown tokens (id={self.tokenizer.unk_token_id}):"
            )
            print(
                "     Tokens which can'T be encoded: ",
                [T for T, id in zip(text_tokens_list, text_token_ids) if id == self.tokenizer.unk_token_id],
            )
            print("     Consider updating the BPE model or modifying the text to avoid unknown tokens.")

        autoregressive_batch_size = 1
        do_sample = generation_kwargs.pop("do_sample", True)
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        temperature = generation_kwargs.pop("temperature", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        top_p = generation_kwargs.pop("top_p", 0.8)

        wavs: list[Tensor] = []
        gpt_gen_time = Timer()
        gpt_forward_time = Timer()
        s2mel_time = Timer()
        bigvgan_time = Timer()
        has_warned = False
        for seg_idx, sent in enumerate(segments):
            self._set_gr_progress(
                0.2 + 0.7 * seg_idx / len(segments), f"speech synthesis {seg_idx + 1}/{len(segments)}..."
            )

            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)

            with torch.inference_mode():
                with torch.autocast(self.device.type, dtype=self.dtype), gpt_gen_time:
                    emotion_vector = self.gpt.get_emo_vec(emotion_conditioning_embedding)
                    base_vector = self.gpt.get_emo_vec(speaker_conditioning_embedding)

                    emotion_vector = base_vector + emo_alpha * (emotion_vector - base_vector)

                    if weight_vector is not None and emotion_matrix is not None:
                        emotion_vector = torch.as_tensor(
                            emotion_matrix + (1 - torch.sum(weight_vector)) * emotion_vector
                        )

                    speech_conditioning_latent = self.gpt.process_speech_condition(speaker_conditioning_embedding)
                    codes = self.gpt.inference_speech(
                        speech_conditioning_latent,
                        text_tokens,
                        emo_vec=emotion_vector,
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

                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_segment`({max_text_tokens_per_segment}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning,
                    )
                    has_warned = True

                code_lens = [
                    x.tolist().index(self.stop_mel_token) if self.stop_mel_token in x else len(x) for x in codes
                ]
                codes = codes[:, : max(code_lens)]

                with torch.autocast(self.device.type, dtype=self.dtype), gpt_forward_time:
                    assert speech_conditioning_latent.shape == torch.Size([1, 32, 1280])
                    assert text_tokens.shape[0] == 1
                    assert codes.shape[0] == 1
                    assert emotion_conditioning_embedding.shape == torch.Size([1, 749, 1024])
                    assert emotion_vector.shape == torch.Size([1, 1280])
                    latent = self.gpt(
                        speech_conditioning_latent,
                        text_tokens,
                        codes,
                        emotion_conditioning_embedding,
                        emo_vec=emotion_vector,
                        use_speed=speaker_conditioning_embedding.size(0),
                        device=self.device,
                    )

                with s2mel_time:
                    voice_conversion_target = self.generate_voice_conversion(
                        code_lens, prompt_condition, style, ref_mel, codes, latent
                    )

                with bigvgan_time:
                    wav: Tensor = self.bigvgan(voice_conversion_target.float()).squeeze().unsqueeze(0).squeeze(1)

                wavs.append(wav.cpu())  # to cpu before saving
                if stream_return:
                    yield wav.cpu()
                    yield get_silence_interval(wavs[0].size(0), interval_silence, self.cfg.sample_rate)
        inference_timer.stop()

        self._set_gr_progress(0.9, "saving audio...")
        silence_tensor = get_silence_interval(wavs[0].size(0), interval_silence, self.cfg.sample_rate)
        # Insert silences between segments
        wavs = [item for x in wavs for item in (x, silence_tensor)][:-1]
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / self.cfg.sample_rate
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> s2mel_time: {s2mel_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {inference_timer:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {inference_timer.elapsed / wav_length:.4f}")

        wav = wav.cpu()
        if output_path:
            # Save audio directly to the specified path
            AudioEncoder(wav, sample_rate=self.cfg.sample_rate).to_file(output_path)
            print(">> wav file saved to:", output_path)
            yield output_path  # pyright: ignore[reportReturnType]
        else:
            # Return in a format compatible with Gradio
            wav_data = wav.type(torch.int16)  # pyright: ignore[reportUnreachable]
            wav_data = wav_data.numpy().T
            yield (self.cfg.sample_rate, wav_data)
