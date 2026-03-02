import logging
import os
import random
import warnings
from collections.abc import Callable, Generator, Mapping, Sequence
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Final, cast

import huggingface_hub as hf
import torch
import torch.nn.functional as F
import torchaudio
import transformers
from beartype import beartype
from jaxtyping import Float, Int
from torch import Tensor
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from bigvgan.inference import BigVGANInference
from indextts import load
from indextts.gpt.model_v2 import UnifiedVoice, post_init_gpt2_config
from indextts.qwen import QwenEmotion
from indextts.s2mel import CFM, CAMPPlus, InterpolateRegulator, mel_spectrogram
from indextts.s2mel.audio import N_MELS, SAMPLING_RATE
from indextts.util import Timer
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.repcodec_model import RepCodec

logger = logging.getLogger(__name__)

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"

EMO_NUM: Final[Sequence[int]] = (3, 17, 2, 8, 4, 5, 10, 24)
MAX_AUDIO_LENGTH_SECONDS: Final = 15
WIDEBAND_SR: Final = 16000

MAX_MEL_TOKENS: Final = 1815
START_MEL_TOKEN: Final = 8192
STOP_MEL_TOKEN: Final = START_MEL_TOKEN + 1

# Maximum total weight allowed across all emotion vector components.
# Emotion vectors whose components sum above this are scaled down proportionally.
EMO_MAX_WEIGHT_SUM: Final = 0.8

# Progress-bar checkpoints for the synthesis loop.
# Synthesis occupies the range [_PROGRESS_SYNTH_START, _PROGRESS_SYNTH_START + _PROGRESS_SYNTH_SPAN].
_PROGRESS_SYNTH_START: Final = 0.2
_PROGRESS_SYNTH_SPAN: Final = 0.7


def normalize_emo_vec(vector: Sequence[float]) -> list[float]:
    # apply biased emotion factors for better user experience,
    # by de-emphasizing emotions that can cause strange results

    # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    biases = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
    vector = [vec * bias for vec, bias in zip(vector, biases)]

    # the total emotion sum must be EMO_MAX_WEIGHT_SUM or less
    total = sum(vector)
    if total > EMO_MAX_WEIGHT_SUM:
        scale_factor = EMO_MAX_WEIGHT_SUM / total
        vector = [vec * scale_factor for vec in vector]

    return list(vector)


@beartype
def _get_silence_interval(size: int, interval_silence: int) -> Float[Tensor, "batch samples"]:
    return torch.zeros(size, (SAMPLING_RATE * interval_silence) // 1000)


@beartype
def _load_and_cut_audio(path: Path, sample_rate: int | None = None) -> tuple[Float[Tensor, "1 samples"], int]:
    samples = AudioDecoder(path, num_channels=1, sample_rate=sample_rate).get_samples_played_in_range(
        0, MAX_AUDIO_LENGTH_SECONDS
    )
    audio = samples.data
    sample_rate = samples.sample_rate

    if audio.dim() != 2 or audio.size(0) != 1:
        raise ValueError(f"Only mono audio is supported. Got shape: {audio.shape}")
    max_audio_samples = MAX_AUDIO_LENGTH_SECONDS * sample_rate

    if audio.shape[1] > max_audio_samples:
        audio = audio[:, :max_audio_samples]
    return audio, sample_rate


class IndexTTS2:
    device: torch.device
    dtype: torch.dtype
    use_accel: bool

    emo_matrix: tuple[Tensor, ...]
    spk_matrix: tuple[Tensor, ...]

    glossary_path: Path

    # Progress reference display (optional)
    gr_progress: Callable[..., None] | None = None
    model_version: float = 2.0

    bigvgan: BigVGANInference
    campplus_model: CAMPPlus
    cfm: CFM
    extract_features: Final = load.load_feature_extractor()
    gpt: UnifiedVoice
    length_regulator: InterpolateRegulator
    normalizer: Final = TextNormalizer()
    semantic_codec: RepCodec
    semantic_mean: Tensor
    semantic_model: transformers.Wav2Vec2BertModel
    semantic_std: Tensor
    tokenizer: TextTokenizer

    @cached_property[QwenEmotion]
    def qwen_emo(self) -> QwenEmotion:
        return QwenEmotion("dsinghvi/qwen0.6bemo4-merge")

    def __init__(
        self,
        device: str | None = None,
        use_cuda_kernel: bool = False,
        use_accel: bool = False,
        use_torch_compile: bool = False,
    ) -> None:
        """
        Args:
            device (str | None): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
            use_accel (bool): whether to use acceleration engine for GPT2 or not.
            use_torch_compile (bool): whether to use torch.compile for optimization or not.
        """

        self.device = (
            torch.device(device) if device else torch.accelerator.current_accelerator() or torch.get_default_device()
        )
        self.dtype = torch.get_default_dtype()
        self.use_accel = use_accel

        self.gpt = load.load_unified_voice(self.device)
        self.semantic_model = load.load_semantic_model(self.device)
        self.semantic_mean, self.semantic_std = load.load_semantic_stats(self.device)
        self.semantic_codec = load.load_semantic_codec(self.device)
        self.bigvgan = load.load_bigvgan(self.device, use_cuda_kernel)
        self.campplus_model = load.load_campplus(self.device)
        self.tokenizer = load.load_tokenizer(self.normalizer)
        self.cfm = load.load_cfm(self.device)
        self.length_regulator = load.load_length_regulator(self.device)

        post_init_gpt2_config(self.gpt)

        # Enable torch.compile optimization if requested
        if use_torch_compile:
            logger.info(">> Enabling torch.compile optimization")
            self.cfm.enable_torch_compile()
            logger.info(">> torch.compile optimization enabled successfully")

        self.spk_matrix = self._get_matrix("feat1.pt")
        self.emo_matrix = self._get_matrix("feat2.pt")

        # 加载术语词汇表（如果存在）
        self.glossary_path = Path("checkpoints") / "glossary.yaml"
        if self.glossary_path.exists():
            self.normalizer.load_glossary_from_yaml(self.glossary_path)
            logger.info(">> Glossary loaded from: %s", self.glossary_path)

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
        quick_streaming_tokens: int = 0,
        stream_return: bool = False,
        use_emo_text: bool = False,
        use_random: bool = False,
        do_sample: bool = True,
        length_penalty: float = 0.0,
        num_beams: int = 3,
        repetition_penalty: float = 10.0,
        temperature: float = 0.8,
        top_k: int = 30,
        top_p: float = 0.8,
    ) -> Path | Generator[Tensor] | None:
        # --- input validation ---
        if not (0.0 <= emo_alpha <= 1.0):
            raise ValueError(f"emo_alpha must be in [0.0, 1.0], got {emo_alpha}")
        if interval_silence < 0:
            raise ValueError(f"interval_silence must be >= 0 ms, got {interval_silence}")
        if num_beams < 1:
            raise ValueError(f"num_beams must be >= 1, got {num_beams}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0.0, got {temperature}")
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0.0, 1.0], got {top_p}")
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")

        if use_emo_text or emo_vector is not None:
            # we're using a text or emotion vector guidance; so we must remove
            # "emotion reference voice", to ensure we use correct emotion mixing!
            emo_audio_prompt = None

        if use_emo_text:
            # automatically generate emotion vectors from text prompt
            emo_text = emo_text or text  # use main text prompt
            emo_dict = self.qwen_emo.inference(emo_text)
            logger.info("detected emotion vectors from text: %s", emo_dict)
            # convert ordered dict to list of vectors; the order is VERY important!
            emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            # we have emotion vectors; they can't be blended via alpha mixing
            # in the main inference process later, so we must pre-calculate
            # their new strengths here based on the alpha instead!
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:  # noqa: RUF069
                # scale each vector and truncate to 4 decimals (for nicer printing)
                emo_vector = [int(x * emo_vector_scale * 10_000) / 10_000 for x in emo_vector]
                logger.info("scaled emotion vectors to %sx: %s", emo_vector_scale, emo_vector)

        if emo_audio_prompt is None:
            # we are not using any external "emotion reference voice"; use
            # speaker's voice as the main emotion reference audio.
            emo_audio_prompt = spk_audio_prompt
            # must always use alpha=1.0 when we don't have an external reference voice
            emo_alpha = 1.0

        gen = self._infer_generator(
            spk_audio_prompt,
            text,
            output_path,
            emo_audio_prompt,
            emo_alpha=emo_alpha,
            emo_vector=emo_vector,
            use_random=use_random,
            interval_silence=interval_silence,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            stream_return=stream_return,
            quick_streaming_tokens=quick_streaming_tokens,
            do_sample=do_sample,
            length_penalty=length_penalty,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if stream_return:
            return gen
        try:
            return next(gen)  # pyright: ignore[reportReturnType]
        except IndexError:
            return None

    @torch.inference_mode()
    def _infer_generator(
        self,
        spk_audio_prompt: Path,
        text: str,
        output_path: Path,
        emo_audio_prompt: Path,
        emo_alpha: float = 1.0,
        emo_vector: Sequence[float] | None = None,
        use_random: bool = False,
        interval_silence: int = 200,
        max_text_tokens_per_segment: int = 120,
        stream_return: bool = False,
        quick_streaming_tokens: int = 0,
        do_sample: bool = True,
        length_penalty: float = 0.0,
        num_beams: int = 3,
        repetition_penalty: float = 10.0,
        temperature: float = 0.8,
        top_k: int = 30,
        top_p: float = 0.8,
    ) -> Generator[Tensor]:
        logger.info(">> starting inference...")
        self._set_gr_progress(0.0, "starting inference...")
        inference_timer = Timer()
        inference_timer.start()

        prompt_condition, style, ref_mel, speaker_conditioning_embedding = self._extract_audio_features(
            spk_audio_prompt
        )

        weight_vector = None
        emotion_matrix = None
        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector, device=self.device)
            emotion_matrix = self._generate_emotion_matrix(weight_vector, style, use_random=use_random)

        emotion_conditioning_embedding = self._extract_emotion_features(emo_audio_prompt)

        self._set_gr_progress(0.1, "text processing...")
        text_tokens_list = self.tokenizer.tokenize(text)
        segments = self.tokenizer.split_segments(
            text_tokens_list, max_text_tokens_per_segment, quick_streaming_tokens=quick_streaming_tokens
        )

        emotion_vector = self.gpt.get_emo_vec(emotion_conditioning_embedding)
        base_vector = self.gpt.get_emo_vec(speaker_conditioning_embedding)

        emotion_vector = base_vector + emo_alpha * (emotion_vector - base_vector)

        if weight_vector is not None and emotion_matrix is not None:
            emotion_vector = torch.as_tensor(emotion_matrix + (1 - weight_vector.sum()) * emotion_vector)

        wavs: list[Tensor] = []
        gpt_gen_time = Timer()
        s2mel_time = Timer()
        bigvgan_time = Timer()
        has_warned = False
        for seg_idx, sent in enumerate(segments):
            self._set_gr_progress(
                _PROGRESS_SYNTH_START + _PROGRESS_SYNTH_SPAN * seg_idx / len(segments),
                f"speech synthesis {seg_idx + 1}/{len(segments)}...",
            )

            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)

            with torch.inference_mode():
                with torch.autocast(self.device.type, dtype=self.dtype), gpt_gen_time:
                    speech_conditioning_latent = self.gpt.process_speech_condition(speaker_conditioning_embedding)
                    codes = self.gpt.inference_speech(
                        speech_conditioning_latent,
                        text_tokens,
                        emo_vec=emotion_vector,
                        do_sample=do_sample,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        max_generate_length=MAX_MEL_TOKENS,
                    )

                if not has_warned and (codes[:, -1] != STOP_MEL_TOKEN).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `MAX_MEL_TOKENS` ({MAX_MEL_TOKENS}). "
                        + f"Input text tokens: {text_tokens.shape[1]}. "
                        + f"Consider reducing `max_text_tokens_per_segment`({max_text_tokens_per_segment}) or increasing `MAX_MEL_TOKENS`.",
                        stacklevel=2,
                        category=RuntimeWarning,
                    )
                    has_warned = True

                code_lens = [x.tolist().index(STOP_MEL_TOKEN) if STOP_MEL_TOKEN in x else len(x) for x in codes]
                codes = codes[:, : max(code_lens)]

                with s2mel_time:
                    generated_mel = self._generate_mel_from_codes(code_lens, prompt_condition, style, ref_mel, codes)

                with bigvgan_time:
                    wav = self.bigvgan(generated_mel.float()).squeeze().unsqueeze(0).squeeze(1)

                wavs.append(wav.cpu())  # to cpu before saving
                if stream_return:
                    yield wav.cpu()
                    yield _get_silence_interval(wavs[0].size(0), interval_silence)
        inference_timer.stop()

        self._set_gr_progress(0.9, "saving audio...")
        silence_tensor = _get_silence_interval(wavs[0].size(0), interval_silence)
        # Insert silences between segments
        wavs = [item for x in wavs for item in (x, silence_tensor)][:-1]
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / SAMPLING_RATE

        logger.info(">> gpt_gen_time:   %.2f seconds", gpt_gen_time.elapsed)
        logger.info(">> s2mel_time:     %.2f seconds", s2mel_time.elapsed)
        logger.info(">> bigvgan_time:   %.2f seconds", bigvgan_time.elapsed)
        logger.info(">> Total inference time: %.2f seconds", inference_timer.elapsed)
        logger.info(">> Generated audio length: %.2f seconds", wav_length)
        logger.info(">> RTF: %.4f", inference_timer.elapsed / wav_length)

        wav = wav.cpu()
        if output_path:
            # Save audio directly to the specified path
            AudioEncoder(wav, sample_rate=SAMPLING_RATE).to_file(output_path)
            logger.info(">> wav file saved to: %s", output_path)
            yield output_path  # pyright: ignore[reportReturnType]
        else:
            # Return in a format compatible with Gradio
            wav_data = wav.type(torch.int16)  # pyright: ignore[reportUnreachable]
            wav_data = wav_data.numpy().T
            yield (SAMPLING_RATE, wav_data)

    @beartype
    def _generate_mel_from_codes(
        self,
        code_lens: list[int],
        prompt_condition: Float[Tensor, "batch prompt_time cond_dim"],
        style: Float[Tensor, "batch style_dim"],
        ref_mel: Float[Tensor, "batch mel_bins ref_time"],
        codes: Int[Tensor, "batch time"],
    ) -> Float[Tensor, "batch mel_bins time"]:
        semantic_inference = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1)).mT
        target_lengths = (torch.tensor(code_lens, device=self.device) * 1.72).long().max().item()

        cond = self.length_regulator.__call__(semantic_inference, ylens=int(target_lengths))
        cond = torch.cat([prompt_condition, cond], dim=1)
        target = self.cfm.__call__(cond, ref_mel, style)
        return target[:, :, ref_mel.size(-1) :]

    @lru_cache(5)  # noqa: B019
    def _extract_emotion_features(self, prompt: Path) -> Tensor:
        logger.info(">> extracting emotion features from prompt: %s", prompt)
        audio, _ = _load_and_cut_audio(prompt, sample_rate=WIDEBAND_SR)
        inputs = self.extract_features(audio.numpy(), sampling_rate=WIDEBAND_SR, return_tensors="pt")
        inputs = cast(Mapping[str, Tensor], inputs.to(self.device))
        return self._get_emb(inputs["input_features"], inputs["attention_mask"])

    @beartype
    def _generate_emotion_matrix(
        self,
        weight_vector: Float[Tensor, "num_emotions"],
        style: Float[Tensor, "batch style_dim"],
        use_random: bool = False,
    ) -> Float[Tensor, "batch style_dim"]:
        if use_random:
            indices = [random.randint(0, x - 1) for x in EMO_NUM]
        else:
            indices = [int(F.cosine_similarity(style, x).argmax()) for x in self.spk_matrix]

        matrix = [x[i].unsqueeze(0) for i, x in zip(indices, self.emo_matrix)]
        matrix = torch.cat(matrix)
        matrix = weight_vector.unsqueeze(1) * matrix
        matrix = matrix.sum(dim=0)
        return matrix.unsqueeze(0)

    def _get_matrix(self, filename: str) -> tuple[Tensor, ...]:
        path = hf.hf_hub_download(repo_id="IndexTeam/IndexTTS-2", filename=filename)
        data = cast(Tensor, torch.load(path, map_location=self.device))
        return data.split(EMO_NUM)

    @torch.inference_mode()
    @beartype
    def _get_emb(
        self, input_features: Float[Tensor, "batch time n_mels"], attention_mask: Int[Tensor, "batch time"]
    ) -> Float[Tensor, "batch time dim"]:
        vq_emb = self.semantic_model(
            input_features=input_features, attention_mask=attention_mask, output_hidden_states=True
        )
        if isinstance(vq_emb, tuple) or vq_emb.hidden_states is None:
            raise RuntimeError("semantic_model did not return hidden states; ensure output_hidden_states=True is set")
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        return (feat - self.semantic_mean) / self.semantic_std

    def _set_gr_progress(self, value: float, desc: str) -> None:
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    @lru_cache  # noqa: B019
    def _extract_audio_features(self, prompt: Path) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        logger.info(">> extracting audio features from prompt: %s", prompt)
        audio, sr = _load_and_cut_audio(prompt)
        audio_16k = torchaudio.functional.resample(audio, sr, WIDEBAND_SR)
        audio_22k = torchaudio.functional.resample(audio, sr, SAMPLING_RATE)

        mel = mel_spectrogram(audio_22k)
        feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(self.device), num_mel_bins=N_MELS)
        feat -= feat.mean(dim=0, keepdim=True)  # feat2: Another filter energy group feature [922, 80]
        style = self.campplus_model(feat.unsqueeze(0))  # Global style of the reference audio [1, STYLE_DIM]

        inputs = cast(
            Mapping[str, Tensor],
            self.extract_features(audio_16k, sampling_rate=WIDEBAND_SR, return_tensors="pt").to(self.device),
        )

        embedding = self._get_emb(inputs["input_features"], inputs["attention_mask"])
        prompt_condition = self.length_regulator.__call__(self.semantic_codec.quantize(embedding), ylens=mel.size(2))
        return prompt_condition, style, mel, embedding
