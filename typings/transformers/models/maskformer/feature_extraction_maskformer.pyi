from ...utils.import_utils import requires
from .image_processing_maskformer import MaskFormerImageProcessor

"""Feature extractor class for MaskFormer."""
logger = ...

@requires(backends=("vision",))
class MaskFormerFeatureExtractor(MaskFormerImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["MaskFormerFeatureExtractor"]
