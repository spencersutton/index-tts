import json
import os
import random
import re
import time
import warnings
from subprocess import CalledProcessError
from typing import Callable

import bigvgan
import librosa
import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from modelscope import AutoModelForCausalLM
from omegaconf import DictConfig, ListConfig, OmegaConf
from pyparsing import lru_cache
from safetensors.torch import load_model
from transformers import (
    AutoTokenizer,
    Qwen2Tokenizer,
    SeamlessM4TFeatureExtractor,
    Wav2Vec2BertModel,
)
from transformers.models.qwen3 import Qwen3ForCausalLM

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.s2mel.modules.audio import mel_spectrogram
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.maskgct.models.codec.kmeans.repcodec_model import RepCodec
from indextts.utils.maskgct_utils import build_semantic_codec, build_semantic_model

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"


class IndexTTS2:
    semantic_codec: RepCodec
    semantic_model: Wav2Vec2BertModel
    semantic_mean: torch.Tensor
    semantic_std: torch.Tensor
    normalizer: TextNormalizer
    extract_features: SeamlessM4TFeatureExtractor
    emo_matrix: list[torch.Tensor]
    spk_matrix: list[torch.Tensor]
    cache_spk_cond: torch.Tensor | None
    cache_s2mel_style: torch.Tensor | None
    cache_spk_audio_prompt: str | None
    cache_emo_cond: torch.Tensor | None
    cache_emo_audio_prompt: str | None
    cache_mel: torch.Tensor | None
    cache_s2mel_prompt: torch.Tensor | None
    dtype: torch.dtype | None
    device: str
    use_fp16: bool
    use_cuda_kernel: bool
    stop_mel_token: int
    cfg: DictConfig | ListConfig
    model_dir: str
    qwen_emo: "QwenEmotion"
    bpe_path: str
    emo_num: list[int]
    gr_progress: Callable[..., None] | None

    @property
    @lru_cache(maxsize=1)
    def campplus_model(self) -> CAMPPlus:
        print(">> loading campplus_model...")
        # load campplus_model
        path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
        model = CAMPPlus(feat_dim=80, embedding_size=192)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model = model.to(self.device)
        model.eval()
        print(">> campplus_model weights restored from:", path)
        return model

    @property
    @lru_cache(maxsize=1)
    def s2mel(self):
        print(">> loading s2mel...")
        path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        model = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        model, _, _, _ = load_checkpoint2(
            model,
            None,
            path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        assert isinstance(model, MyModel)
        model = model.to(self.device)
        estimator = model.models["cfm"].estimator
        assert isinstance(estimator, torch.nn.Module)
        assert callable(estimator.setup_caches)
        estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        model.eval()
        print(">> s2mel weights restored from:", path)
        return model

    @property
    @lru_cache(maxsize=1)
    def gpt(self) -> UnifiedVoice:
        print(">> loading gpt...")
        model = UnifiedVoice(**self.cfg.gpt)
        path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(model, path)
        model = model.to(self.device)
        if self.use_fp16:
            model.eval().half()
        else:
            model.eval()
        print(">> GPT weights restored from:", path)
        return model

    @property
    @lru_cache(maxsize=1)
    def tokenizer(self) -> TextTokenizer:
        print(">> loading tokenizer...")
        path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        normalizer = TextNormalizer()
        normalizer.load()
        print(">> TextNormalizer loaded")
        tokenizer = TextTokenizer(path, normalizer)
        print(">> bpe model loaded from:", path)
        return tokenizer

    @property
    @lru_cache(maxsize=1)
    def bigvgan(self) -> "bigvgan.BigVGAN":
        print(">> loading bigvgan...")
        name = self.cfg.vocoder.name
        model = bigvgan.BigVGAN.from_pretrained(
            name, use_cuda_kernel=self.use_cuda_kernel
        )
        model = model.to(self.device)
        model.remove_weight_norm()
        model.eval()
        print(">> bigvgan weights restored from:", name)
        return model

    def mel_fn(self, x: float) -> torch.Tensor:
        params = self.cfg.s2mel["preprocess_params"]
        spect_params = params["spect_params"]
        args = {
            "n_fft": spect_params["n_fft"],
            "win_size": spect_params["win_length"],
            "hop_size": spect_params["hop_length"],
            "num_mels": spect_params["n_mels"],
            "sampling_rate": params["sr"],
            "fmin": spect_params.get("fmin", 0),
            "fmax": None if spect_params.get("fmax", "None") == "None" else 8000,
            "center": False,
        }
        return mel_spectrogram(x, **args)

    def __init__(
        self,
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_fp16=False,
        device=None,
        use_cuda_kernel=None,
        use_deepspeed=False,
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            use_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
            use_deepspeed (bool): whether to use DeepSpeed or not.
        """
        if device is not None:
            self.device = device
            self.use_fp16 = False if device == "cpu" else use_fp16
            self.use_cuda_kernel = (
                use_cuda_kernel is not None
                and use_cuda_kernel
                and device.startswith("cuda")
            )
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

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.use_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        self.qwen_emo = QwenEmotion(
            os.path.join(self.model_dir, self.cfg.qwen_emo_path)
        )

        if use_deepspeed:
            try:
                import deepspeed  # noqa: F401
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                print(
                    f">> Failed to load DeepSpeed. Falling back to normal inference. Error: {e}"
                )

        self.gpt.post_init_gpt2_config(
            use_deepspeed=use_deepspeed, kv_cache=True, half=self.use_fp16
        )

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from bigvgan.alias_free_activation.cuda import activation1d

                print(
                    ">> Preload custom CUDA kernel for BigVGAN",
                    activation1d.anti_alias_activation_cuda,
                )
            except Exception as e:
                print(
                    ">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch."
                )
                print(f"{e!r}")
                self.use_cuda_kernel = False

        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        semantic_model, semantic_mean, semantic_std = build_semantic_model(
            os.path.join(self.model_dir, self.cfg.w2v_stat)
        )
        self.semantic_model = semantic_model.to(self.device)  # type: ignore[arg-type]
        self.semantic_model.eval()
        self.semantic_mean = semantic_mean.to(self.device)
        self.semantic_std = semantic_std.to(self.device)

        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = hf_hub_download(
            "amphion/MaskGCT", filename="semantic_codec/model.safetensors"
        )
        load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.device)
        self.semantic_codec.eval()
        print(">> semantic_codec weights restored from: {}".format(semantic_code_ckpt))

        emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix))
        emo_matrix = emo_matrix.to(self.device)
        self.emo_num = list(self.cfg.emo_num)

        spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix))
        spk_matrix = spk_matrix.to(self.device)

        self.emo_matrix = list(torch.split(emo_matrix, self.emo_num))
        self.spk_matrix = list(torch.split(spk_matrix, self.emo_num))

        # Cache reference audio:
        self.cache_spk_cond = None
        self.cache_s2mel_style = None
        self.cache_s2mel_prompt = None
        self.cache_spk_audio_prompt = None
        self.cache_emo_cond = None
        self.cache_emo_audio_prompt = None
        self.cache_mel = None

        # Progress callback reference (optional)
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

    @torch.no_grad()
    def get_emb(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    def interval_silence(
        self,
        wavs: list[torch.Tensor],
        sampling_rate: int = 22050,
        interval_silence: int = 200,
    ) -> torch.Tensor:
        """
        Silences to be insert between generated segments.
        """

        if not wavs or interval_silence <= 0:
            return torch.zeros(1, 0)  # Return empty tensor instead of list

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        return torch.zeros(channel_size, sil_dur)

    def insert_interval_silence(
        self,
        wavs: list[torch.Tensor],
        sampling_rate: int = 22050,
        interval_silence: int = 200,
    ) -> list[torch.Tensor]:
        """
        Insert silences between generated segments.
        wavs: List[torch.tensor]
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        sil_tensor = torch.zeros(channel_size, sil_dur)

        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav)
            if i < len(wavs) - 1:
                wavs_list.append(sil_tensor)

        return wavs_list

    def _set_gr_progress(self, value: float, desc: str) -> None:
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    def _load_and_cut_audio(
        self,
        audio_path: str,
        max_audio_length_seconds: int,
        verbose: bool = False,
        sr: int | float | None = None,
    ) -> tuple[torch.Tensor, int]:
        if sr is None:
            audio, sr = librosa.load(audio_path)
        else:
            audio, _ = librosa.load(audio_path, sr=sr)
        audio = torch.tensor(audio).unsqueeze(0)
        max_audio_samples = int(max_audio_length_seconds * sr)

        if audio.shape[1] > max_audio_samples:
            if verbose:
                print(
                    f"Audio too long ({audio.shape[1]} samples), truncating to {max_audio_samples} samples"
                )
            audio = audio[:, :max_audio_samples]
        return audio, int(sr)

    def normalize_emo_vec(
        self, emo_vector: list[float], apply_bias: bool = True
    ) -> list[float]:
        # apply biased emotion factors for better user experience,
        # by de-emphasizing emotions that can cause strange results
        if apply_bias:
            # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
            emo_vector = [vec * bias for vec, bias in zip(emo_vector, emo_bias)]

        # the total emotion sum must be 0.8 or less
        emo_sum = sum(emo_vector)
        if emo_sum > 0.8:
            scale_factor = 0.8 / emo_sum
            emo_vector = [vec * scale_factor for vec in emo_vector]

        return emo_vector

    # Original inference mode
    def infer(
        self,
        spk_audio_prompt: str,
        text: str,
        output_path: str,
        emo_audio_prompt: str | None = None,
        emo_alpha: float = 1.0,
        emo_vector: list[float] | None = None,
        use_emo_text: bool = False,
        emo_text: str | None = None,
        use_random: bool = False,
        interval_silence: int = 200,
        verbose: bool = False,
        max_text_tokens_per_segment: int = 120,
        stream_return: bool = False,
        more_segment_before: int = 0,
        **generation_kwargs,
    ):
        if stream_return:
            return self.infer_generator(
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
        else:
            try:
                return list(
                    self.infer_generator(
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
                )[0]
            except IndexError:
                return None

    cache_s2mel_prompt: torch.Tensor | None = None

    def infer_generator(
        self,
        spk_audio_prompt: str,
        text: str,
        output_path: str,
        emo_audio_prompt: str | None = None,
        emo_alpha: float = 1.0,
        emo_vector: list[float] | None = None,
        use_emo_text: bool = False,
        emo_text: str | None = None,
        use_random: bool = False,
        interval_silence: int = 200,
        verbose: bool = False,
        max_text_tokens_per_segment: int = 120,
        stream_return: bool = False,
        quick_streaming_tokens: int = 0,
        **generation_kwargs,
    ):
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
                emo_vector = [
                    int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector
                ]
                print(f"scaled emotion vectors to {emo_vector_scale}x: {emo_vector}")

        if emo_audio_prompt is None:
            # we are not using any external "emotion reference voice"; use
            # speaker's voice as the main emotion reference audio.
            emo_audio_prompt = spk_audio_prompt
            # must always use alpha=1.0 when we don't have an external reference voice
            emo_alpha = 1.0

        prompt_condition: torch.Tensor

        # Only regenerate when the reference audio has changed, to improve speed
        if (
            self.cache_spk_cond is None
            or self.cache_spk_audio_prompt != spk_audio_prompt
        ):
            if self.cache_spk_cond is not None:
                self.cache_spk_cond = None
                self.cache_s2mel_style = None
                self.cache_s2mel_prompt = None
                self.cache_mel = None
                torch.cuda.empty_cache()
            audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15, verbose)
            audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

            inputs = self.extract_features(
                audio_16k, sampling_rate=16000, return_tensors="pt"
            )
            input_features: torch.Tensor = inputs["input_features"]
            attention_mask: torch.Tensor = inputs["attention_mask"]
            input_features = input_features.to(self.device)
            attention_mask = attention_mask.to(self.device)
            spk_cond_emb = self.get_emb(input_features, attention_mask)

            _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
            ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
            feat = torchaudio.compliance.kaldi.fbank(
                audio_16k.to(ref_mel.device),
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000,
            )
            feat = feat - feat.mean(
                dim=0, keepdim=True
            )  # feat2: another filter-bank energy feature [922, 80]
            style = self.campplus_model(
                feat.unsqueeze(0)
            )  # Global style of the reference audio, shape [1, 192]

            prompt_condition = self.s2mel.models["length_regulator"](
                S_ref, ylens=ref_target_lengths, n_quantizers=3, f0=None
            )[0]

            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_mel = ref_mel
        else:
            style = self.cache_s2mel_style
            assert style is not None
            assert self.cache_s2mel_prompt is not None
            prompt_condition = self.cache_s2mel_prompt
            spk_cond_emb = self.cache_spk_cond
            ref_mel = self.cache_mel

        emovec_mat = None
        weight_vector = None
        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector).to(self.device)
            if use_random:
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                random_index = [
                    find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix
                ]

            emo_matrix = [
                tmp[index].unsqueeze(0)
                for index, tmp in zip(random_index, self.emo_matrix)
            ]
            emo_matrix = torch.cat(emo_matrix, 0)
            emovec_mat = weight_vector.unsqueeze(1) * emo_matrix
            emovec_mat = torch.sum(emovec_mat, 0)
            emovec_mat = emovec_mat.unsqueeze(0)

        if (
            self.cache_emo_cond is None
            or self.cache_emo_audio_prompt != emo_audio_prompt
        ):
            if self.cache_emo_cond is not None:
                self.cache_emo_cond = None
                torch.cuda.empty_cache()
            emo_audio, _ = self._load_and_cut_audio(
                emo_audio_prompt, 15, verbose, sr=16000
            )
            emo_inputs = self.extract_features(
                emo_audio.numpy(), sampling_rate=16000, return_tensors="pt"
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
            text_tokens_list,
            max_text_tokens_per_segment,
            quick_streaming_tokens=quick_streaming_tokens,
        )
        segments_count = len(segments)

        text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        if self.tokenizer.unk_token_id in text_token_ids:
            print(
                f"  >> Warning: input text contains {text_token_ids.count(self.tokenizer.unk_token_id)} unknown tokens (id={self.tokenizer.unk_token_id}):"
            )
            print(
                "     Tokens which can't be encoded: ",
                [
                    t
                    for t, id in zip(text_tokens_list, text_token_ids)
                    if id == self.tokenizer.unk_token_id
                ],
            )
            print(
                "     Consider updating the BPE model or modifying the text to avoid unknown tokens."
            )

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
        sampling_rate = 22050

        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        has_warned = False
        silence = None  # for stream_return
        for seg_idx, sent in enumerate(segments):
            self._set_gr_progress(
                0.2 + 0.7 * seg_idx / segments_count,
                f"speech synthesis {seg_idx + 1}/{segments_count}...",
            )

            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(
                text_tokens, dtype=torch.int32, device=self.device
            ).unsqueeze(0)
            if verbose:
                print(text_tokens)
                print(
                    f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}"
                )
                # debug tokenizer
                text_token_syms = self.tokenizer.convert_ids_to_tokens(
                    text_tokens[0].tolist()
                )
                print(
                    "text_token_syms is same as segment tokens", text_token_syms == sent
                )

            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.autocast(
                    text_tokens.device.type,
                    enabled=self.dtype is not None,
                    dtype=self.dtype,
                ):
                    emovec = self.gpt.merge_emovec(
                        spk_cond_emb,
                        emo_cond_emb,
                        torch.tensor(
                            [spk_cond_emb.shape[-1]], device=text_tokens.device
                        ),
                        torch.tensor(
                            [emo_cond_emb.shape[-1]], device=text_tokens.device
                        ),
                        alpha=emo_alpha,
                    )

                    if emo_vector is not None:
                        assert weight_vector is not None
                        emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec
                        # emovec = emovec_mat

                    codes, speech_conditioning_latent = self.gpt.inference_speech(
                        spk_cond_emb,
                        text_tokens,
                        emo_cond_emb,
                        cond_lengths=torch.tensor(
                            [spk_cond_emb.shape[-1]], device=text_tokens.device
                        ),
                        emo_cond_lengths=torch.tensor(
                            [emo_cond_emb.shape[-1]], device=text_tokens.device
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
                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_segment`({max_text_tokens_per_segment}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning,
                    )
                    has_warned = True

                assert isinstance(codes, torch.Tensor)
                code_lens = torch.tensor(
                    [codes.shape[-1]], device=codes.device, dtype=codes.dtype
                )

                code_len = 0
                code_lens = []
                for code in codes:
                    if self.stop_mel_token not in code:
                        code_lens.append(len(code))
                        code_len = len(code)
                    else:
                        len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[
                            0
                        ] + 1
                        code_len = len_ - 1
                    code_lens.append(code_len)
                codes = codes[:, :code_len]
                code_lens = torch.LongTensor(code_lens)
                code_lens = code_lens.to(self.device)
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                m_start_time = time.perf_counter()
                use_speed = (
                    torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
                )
                with torch.autocast(
                    text_tokens.device.type,
                    enabled=self.dtype is not None,
                    dtype=self.dtype,
                ):
                    latent = self.gpt(
                        speech_conditioning_latent,
                        text_tokens,
                        torch.tensor(
                            [text_tokens.shape[-1]], device=text_tokens.device
                        ),
                        codes,
                        torch.tensor([codes.shape[-1]], device=text_tokens.device),
                        emo_cond_emb,
                        cond_mel_lengths=torch.tensor(
                            [spk_cond_emb.shape[-1]], device=text_tokens.device
                        ),
                        emo_cond_mel_lengths=torch.tensor(
                            [emo_cond_emb.shape[-1]], device=text_tokens.device
                        ),
                        emo_vec=emovec,
                        use_speed=use_speed,
                    )
                    gpt_forward_time += time.perf_counter() - m_start_time

                dtype = None
                with torch.autocast(
                    text_tokens.device.type, enabled=dtype is not None, dtype=dtype
                ):
                    m_start_time = time.perf_counter()
                    diffusion_steps = 25
                    inference_cfg_rate = 0.7
                    latent = self.s2mel.models["gpt_layer"](latent)
                    S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
                    assert isinstance(S_infer, torch.Tensor)
                    S_infer = S_infer.transpose(1, 2)
                    S_infer = S_infer + latent
                    target_lengths = (code_lens * 1.72).long()

                    cond = self.s2mel.models["length_regulator"](
                        S_infer, ylens=target_lengths, n_quantizers=3, f0=None
                    )[0]
                    cat_condition = torch.cat([prompt_condition, cond], dim=1)
                    inference_output = self.s2mel.models["cfm"].inference
                    assert callable(inference_output)
                    vc_target: torch.Tensor = inference_output(
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
                    audio_waveform: torch.Tensor = self.bigvgan(vc_target.float())
                    wav = audio_waveform.squeeze().unsqueeze(0)
                    print(wav.shape)
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                if verbose:
                    print(
                        f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max()
                    )
                wavs.append(wav.cpu())  # to cpu before saving
                if stream_return:
                    yield wav.cpu()
                    if silence is None:
                        silence = self.interval_silence(
                            wavs,
                            sampling_rate=sampling_rate,
                            interval_silence=interval_silence,
                        )
                    yield silence
        end_time = time.perf_counter()

        self._set_gr_progress(0.9, "saving audio...")
        wavs = self.insert_interval_silence(
            wavs, sampling_rate=sampling_rate, interval_silence=interval_silence
        )
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
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
            # Directly save the audio to the specified path
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            if stream_return:
                return None
            yield output_path
        else:
            if stream_return:
                return None
            # Return in the format expected by Gradio
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            yield (sampling_rate, wav_data)


def find_most_similar_cosine(
    query_vector: torch.Tensor, matrix: torch.Tensor
) -> torch.Tensor:
    query_vector = query_vector.float()
    matrix = matrix.float()

    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    most_similar_index = torch.argmax(similarities)
    return most_similar_index


class QwenEmotion:
    model_dir: str
    tokenizer: Qwen2Tokenizer
    model: Qwen3ForCausalLM
    prompt: str
    cn_key_to_en: dict[str, str]
    desired_vector_order: list[str]
    melancholic_words: set[str]
    max_score: float
    min_score: float

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype="float16",  # "auto"
            device_map="auto",
        )
        self.prompt = "文本情感分类"
        self.cn_key_to_en = {
            "高兴": "happy",
            "愤怒": "angry",
            "悲伤": "sad",
            "恐惧": "afraid",
            "反感": "disgusted",
            # TODO: the "低落" (melancholic) emotion will always be mapped to
            # "悲伤" (sad) by QwenEmotion's text analysis. It doesn't know the
            # difference between those emotions even if the user writes exact words.
            # SEE: `self.melancholic_words` for the current workaround.
            "低落": "melancholic",
            "惊讶": "surprised",
            "自然": "calm",
        }
        self.desired_vector_order = [
            "高兴",
            "愤怒",
            "悲伤",
            "恐惧",
            "反感",
            "低落",
            "惊讶",
            "自然",
        ]
        self.melancholic_words = {
            # Emotion text phrases that will force QwenEmotion's "悲伤" (sad) detection
            # to become "低落" (melancholic) instead, to work around the limitation mentioned above.
            "低落",
            "melancholy",
            "melancholic",
            "depression",
            "depressed",
            "gloomy",
        }
        self.max_score = 1.2
        self.min_score = 0.0

    def clamp_score(self, value: float) -> float:
        return max(self.min_score, min(self.max_score, value))

    def convert(self, content: dict[str, float]) -> dict[str, float]:
        # generate emotion vector dictionary:
        # - insert values in desired order (Python 3.7+ `dict` remembers insertion order)
        # - convert Chinese keys to English
        # - clamp all values to the allowed min/max range
        # - use 0.0 for any values that were missing in `content`
        emotion_dict = {
            self.cn_key_to_en[cn_key]: self.clamp_score(content.get(cn_key, 0.0))
            for cn_key in self.desired_vector_order
        }

        # default to a calm/neutral voice if all emotion vectors were empty
        if all(val <= 0.0 for val in emotion_dict.values()):
            print(">> no emotions detected; using default calm/neutral voice")
            emotion_dict["calm"] = 1.0

        return emotion_dict

    def inference(self, text_input: str) -> dict[str, float]:
        messages = [
            {"role": "system", "content": f"{self.prompt}"},
            {"role": "user", "content": f"{text_input}"},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([str(text)], return_tensors="pt").to(
            self.model.device
        )

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,  # pyright: ignore[reportArgumentType]
            max_new_tokens=32768,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)

        # decode the JSON emotion detections as a dictionary
        try:
            content = json.loads(content)
        except json.decoder.JSONDecodeError:
            # invalid JSON; fallback to manual string parsing
            # print(">> parsing QwenEmotion response", content)
            content = {
                m.group(1): float(m.group(2))
                for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)
            }
            # print(">> dict result", content)

        # Workaround for QwenEmotion's inability to distinguish "悲伤" (sad) vs "低落" (melancholic).
        # If we detect any of the IndexTTS "melancholic" words, we swap those vectors
        # to represent the "sad" emotion as "melancholic" (instead of sadness).
        text_input_lower = text_input.lower()
        if any(word in text_input_lower for word in self.melancholic_words):
            # print(">> before vec swap", content)
            content["悲伤"], content["低落"] = (
                content.get("低落", 0.0),
                content.get("悲伤", 0.0),
            )
            # print(">>  after vec swap", content)

        return self.convert(content)


if __name__ == "__main__":
    prompt_wav = "examples/voice_01.wav"
    text = "欢迎大家来体验indextts2，并给予我们意见与反馈，谢谢大家。"

    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_cuda_kernel=False,
    )
    tts.infer(
        spk_audio_prompt=prompt_wav, text=text, output_path="gen.wav", verbose=True
    )
