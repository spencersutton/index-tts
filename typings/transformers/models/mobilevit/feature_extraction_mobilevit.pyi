from ...utils.import_utils import requires
from .image_processing_mobilevit import MobileViTImageProcessor

"""Feature extractor class for MobileViT."""
logger = ...

@requires(backends=("vision",))
class MobileViTFeatureExtractor(MobileViTImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["MobileViTFeatureExtractor"]
