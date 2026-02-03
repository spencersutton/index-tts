"""Project-wide architectural constants.

These values define the *shapes* used throughout IndexTTS2.
"""


# ---------------------------------------------------------------------------
# Core hidden sizes / embedding dimensions
# ---------------------------------------------------------------------------

# Shared base model width used by:
# - the Conformer encoder output (conditioning features)
# - the DiT/CFM diffusion transformer internal width
# - the length regulator output width
from typing import Final

S2MEL_MODEL_DIM: Final = 512

# Common derived dimension used for:
# - GLU splits (2x channels)
# - AdaLN modulation projections (shift+scale)
# - per-layer WaveNet conditioning chunks
S2MEL_DOUBLE_DIM: Final = S2MEL_MODEL_DIM * 2  # 1024

# The semantic stream often uses a 1024-wide representation that is projected down
# to `S2MEL_MODEL_DIM` by the length regulator (`content_in_proj: Linear(S2MEL_MODEL_DIM*2 -> S2MEL_MODEL_DIM)`).
SEMANTIC_STREAM_DIM: Final = S2MEL_MODEL_DIM * 2

# ---------------------------------------------------------------------------
# Audio feature dimensions
# ---------------------------------------------------------------------------

# Number of mel filterbank / mel-spectrogram bins.
MEL_BINS: Final = 80


# ---------------------------------------------------------------------------
# Style / speaker embedding dimensions
# ---------------------------------------------------------------------------

# CAMPPlus (x-vector) style embedding dimension used as global conditioning.
STYLE_EMBED_DIM: Final = 192


# ---------------------------------------------------------------------------
# GPT model dimensions
# ---------------------------------------------------------------------------

# GPT-2 hidden size / token embedding width (`n_embd`) for the UnifiedVoice model.
GPT_HIDDEN_SIZE: Final = 1280


# ---------------------------------------------------------------------------
# Token / vocabulary sizes (mel-code + text)
# ---------------------------------------------------------------------------

# RepCodec / mel-code base codebook size (ids: [0, MEL_CODEBOOK_SIZE-1]).
MEL_CODEBOOK_SIZE: Final = 8192

# Special mel tokens appended to the code vocabulary.
START_MEL_TOKEN: Final = MEL_CODEBOOK_SIZE
END_MEL_TOKEN: Final = MEL_CODEBOOK_SIZE + 1

# Total mel-code vocab size = base codes + 2 special tokens.
MEL_VOCAB_SIZE: Final = MEL_CODEBOOK_SIZE + 2  # 8194

# RepCodec codebook embedding width (factorized VQ dimension).
REP_CODEC_CODE_DIM: Final = 8

# Text BPE vocab size (includes special tokens; pad=0, eos=1 in current pipeline).
TEXT_VOCAB_SIZE: Final = 12000 + 1


# ---------------------------------------------------------------------------
# Audio/STFT configuration
# ---------------------------------------------------------------------------

AUDIO_SAMPLE_RATE: Final = 22050

# STFT configuration used by `indextts.s2mel.modules.audio`.
STFT_N_FFT: Final = 1024
STFT_N_FREQS: Final = STFT_N_FFT // 2 + 1

# Convenience constant for mel filterbank construction.
AUDIO_NYQUIST_HZ: Final = AUDIO_SAMPLE_RATE // 2

VOCOS_DIM: Final = 384

DIFFUSION_STEPS: Final = 25

INFERENCE_CFG_RATE: Final = 0.7
