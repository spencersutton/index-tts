from ...utils.import_utils import requires
from .image_processing_deit import DeiTImageProcessor

"""Feature extractor class for DeiT."""
logger = ...

@requires(backends=("vision",))
class DeiTFeatureExtractor(DeiTImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["DeiTFeatureExtractor"]
