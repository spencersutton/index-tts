from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

SAMPLING_RATE = 22050


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


START_MEL_TOKEN = 2**13
STOP_MEL_TOKEN = START_MEL_TOKEN + 1
NUMBER_MEL_CODES = STOP_MEL_TOKEN + 1

NUMBER_TEXT_TOKENS = 12000
START_TEXT_TOKEN = 0
STOP_TEXT_TOKEN = START_TEXT_TOKEN + 1


@dataclass
class GptConfig(ConfigMapping):
    model_dim: int
    max_mel_tokens: int
    max_text_tokens: int
    heads: int
    layers: int


@dataclass
class SemanticCodecConfig(ConfigMapping):
    hidden_size: int


@dataclass
class SpectParams(ConfigMapping):
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
    qwen_emo_path: str
    vocoder: VocoderConfig
    version: float
