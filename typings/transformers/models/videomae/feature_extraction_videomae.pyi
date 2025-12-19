from ...utils.import_utils import requires
from .image_processing_videomae import VideoMAEImageProcessor

"""Feature extractor class for VideoMAE."""
logger = ...

@requires(backends=("vision",))
class VideoMAEFeatureExtractor(VideoMAEImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["VideoMAEFeatureExtractor"]
