from typing import Any

import numpy as np
from gradio_client.documentation import document

@document()
def is_audio_correct_length(
    audio: tuple[int, np.ndarray], min_length: float | None, max_length: float | None
) -> dict[str, Any]: ...
@document()
def is_video_correct_length(video: str, min_length: float | None, max_length: float | None) -> dict[str, Any]: ...
