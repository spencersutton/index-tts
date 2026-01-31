from dataclasses import dataclass, field


@dataclass
class _HFModelReference:
    repo_id: str
    filename: str


@dataclass
class UnifiedVoiceConfig:
    heads: int = 20
    layers: int = 24

    max_mel_tokens: int = 1815
    number_mel_codes: int = 8194
    start_mel_token: int = 8192
    stop_mel_token: int = 8193

    max_text_tokens: int = 600
    number_text_tokens: int = 12000
    start_text_token: int = 0
    stop_text_token: int = 1


@dataclass
class IndexTTSConfig:
    sample_rate: int = 22050
    spk_matrix: str = "feat1.pt"
    emo_matrix: str = "feat2.pt"
    dataset: _HFModelReference = field(default_factory=lambda: _HFModelReference("IndexTeam/IndexTTS-2", "bpe.model"))
    gpt: UnifiedVoiceConfig = field(default_factory=UnifiedVoiceConfig)
    gpt_checkpoint: str = "checkpoints/gpt.pth"
    qwen_emo_path: str = "dsinghvi/qwen0.6bemo4-merge"
    version: float = 2.0
    vocoder: _HFModelReference = field(
        default_factory=lambda: _HFModelReference("nvidia/bigvgan_v2_22khz_80band_256x", "bigvgan_generator.pt")
    )
    w2v_stat: _HFModelReference = field(
        default_factory=lambda: _HFModelReference("amphion/dualcodec", "w2vbert2_mean_var_stats_emilia.pt")
    )
