from ...utils.import_utils import requires
from .image_processing_segformer import SegformerImageProcessor

"""Feature extractor class for SegFormer."""
logger = ...

@requires(backends=("vision",))
class SegformerFeatureExtractor(SegformerImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["SegformerFeatureExtractor"]
