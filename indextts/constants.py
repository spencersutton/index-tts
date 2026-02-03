"""Project-wide architectural constants.

These values define the *shapes* used throughout IndexTTS2. The original short
names (e.g. `S2MEL_MODEL_DIM`) are kept for backward compatibility, but the more explicit
aliases below document what each constant represents.
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

# The semantic stream often uses a 1024-wide representation that is projected down
# to `S2MEL_MODEL_DIM` by the length regulator (`content_in_proj: Linear(S2MEL_MODEL_DIM*2 -> S2MEL_MODEL_DIM)`).
SEMANTIC_STREAM_DIM: Final = S2MEL_MODEL_DIM * 2  # 1024

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
