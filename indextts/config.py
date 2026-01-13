from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any


class ConfigMapping(Mapping[str, Any]):
    """MixIn to allow dot access (via dataclass) and bracket access/** unpacking."""

    # These stubs appease the type checker for **unpacking
    def __getitem__(self, key: str) -> Any:
        # OmegaConf objects support this at runtime,
        # but we provide this for the type checker.
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        # This makes it compatible with dict() and **
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)


@dataclass
class MelConfig(ConfigMapping):
    sample_rate: int
    n_fft: int
    hop_length: int
    win_length: int
    n_mels: int
    mel_fmin: int
    normalize: bool


@dataclass
class HFModelReference(ConfigMapping):
    repo_id: str
    filename: str


@dataclass
class DatasetConfig(ConfigMapping):
    bpe_model: HFModelReference
    sample_rate: int
    squeeze: bool
    mel: MelConfig


@dataclass
class GptConfig(ConfigMapping):
    model_dim: int
    max_mel_tokens: int
    max_text_tokens: int
    heads: int
    use_mel_codes_as_input: bool
    mel_length_compression: int
    layers: int
    number_text_tokens: int
    number_mel_codes: int
    start_mel_token: int
    stop_mel_token: int
    start_text_token: int
    stop_text_token: int
    train_solo_embeddings: bool


@dataclass
class SemanticCodecConfig(ConfigMapping):
    codebook_size: int
    hidden_size: int
    codebook_dim: int
    vocos_dim: int
    vocos_intermediate_dim: int
    vocos_num_layers: int


@dataclass
class SpectParams(ConfigMapping):
    n_fft: int
    win_length: int
    hop_length: int
    n_mels: int
    fmin: int
    fmax: str | None


@dataclass
class PreprocessParams(ConfigMapping):
    sr: int
    spect_params: SpectParams


@dataclass
class StyleEncoderConfig(ConfigMapping):
    dim: int


@dataclass
class LengthRegulatorConfig(ConfigMapping):
    channels: int
    is_discrete: bool
    in_channels: int
    content_codebook_size: int
    sampling_ratios: list[int]
    vector_quantize: bool
    n_codebooks: int
    quantizer_dropout: float
    f0_condition: bool
    n_f0_bins: int


@dataclass
class DiTConfig(ConfigMapping):
    hidden_dim: int
    num_heads: int
    depth: int
    class_dropout_prob: float
    block_size: int
    in_channels: int
    style_condition: bool
    final_layer_type: str
    target: str
    content_dim: int
    content_codebook_size: int
    content_type: str
    f0_condition: bool
    n_f0_bins: int
    content_codebooks: int
    is_causal: bool
    long_skip_connection: bool
    zero_prompt_speech_token: bool
    time_as_token: bool
    style_as_token: bool
    uvit_skip_connection: bool
    add_resblock_in_transformer: bool


@dataclass
class WavenetConfig(ConfigMapping):
    hidden_dim: int
    num_layers: int
    kernel_size: int
    dilation_rate: int
    p_dropout: float
    style_condition: bool


@dataclass
class S2MelConfig(ConfigMapping):
    preprocess_params: PreprocessParams
    dit_type: str
    reg_loss_type: str
    style_encoder: StyleEncoderConfig
    length_regulator: LengthRegulatorConfig
    DiT: DiTConfig
    wavenet: WavenetConfig


@dataclass
class VocoderConfig(ConfigMapping):
    type: str
    name: str


@dataclass
class IndexTTSConfig(ConfigMapping):
    dataset: DatasetConfig
    gpt: GptConfig
    semantic_codec: SemanticCodecConfig
    s2mel: S2MelConfig
    gpt_checkpoint: str
    w2v_stat: HFModelReference
    s2mel_checkpoint: str
    emo_matrix: str
    spk_matrix: str
    emo_num: list[int]
    qwen_emo_path: str
    vocoder: VocoderConfig
    version: float
