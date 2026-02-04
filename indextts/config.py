from dataclasses import dataclass, field


@dataclass
class UnifiedVoiceConfig:
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
    gpt: UnifiedVoiceConfig = field(default_factory=UnifiedVoiceConfig)
    version: float = 2.0
