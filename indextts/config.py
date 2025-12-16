"""Typed dataclasses for checkpoints/config.yaml."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MelConfig:
    sample_rate: int
    n_fft: int
    hop_length: int
    win_length: int
    n_mels: int
    mel_fmin: float
    normalize: bool


@dataclass
class DatasetConfig:
    bpe_model: str
    sample_rate: int
    squeeze: bool
    mel: MelConfig


@dataclass
class ConditionModule:
    output_size: int
    linear_units: int
    attention_heads: int
    num_blocks: int
    input_layer: str
    perceiver_mult: int


@dataclass
class GPTConfig:
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
    condition_type: str
    condition_module: ConditionModule
    emo_condition_module: ConditionModule


@dataclass
class SemanticCodecConfig:
    codebook_size: int
    hidden_size: int
    codebook_dim: int
    vocos_dim: int
    vocos_intermediate_dim: int
    vocos_num_layers: int


@dataclass
class SpectParams:
    n_fft: int
    win_length: int
    hop_length: int
    n_mels: int
    fmin: int | None
    fmax: str | None


@dataclass
class PreprocessParams:
    sr: int
    spect_params: SpectParams


@dataclass
class LengthRegulator:
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
class DiTConfig:
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
class WavenetConfig:
    hidden_dim: int
    num_layers: int
    kernel_size: int
    dilation_rate: int
    p_dropout: float
    style_condition: bool


@dataclass
class StyleEncoderConfig:
    dim: int


@dataclass
class S2MelConfig:
    preprocess_params: PreprocessParams
    dit_type: str
    reg_loss_type: str
    style_encoder: StyleEncoderConfig
    length_regulator: LengthRegulator
    DiT: DiTConfig
    wavenet: WavenetConfig


@dataclass
class VocoderConfig:
    type: str
    name: str


@dataclass
class CheckpointsConfig:
    dataset: DatasetConfig
    gpt: GPTConfig
    semantic_codec: SemanticCodecConfig
    s2mel: S2MelConfig

    # simple file pointers / other runtime fields
    gpt_checkpoint: str
    w2v_stat: str
    gpt_layer_checkpoint: str
    cfm_checkpoint: str
    len_reg_checkpoint: str
    emo_matrix: str
    spk_matrix: str
    emo_num: list[int]
    qwen_emo_path: str
    vocoder: VocoderConfig
    version: float
