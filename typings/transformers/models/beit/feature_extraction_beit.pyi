from ...utils.import_utils import requires
from .image_processing_beit import BeitImageProcessor

"""Feature extractor class for BEiT."""
logger = ...

@requires(backends=("vision",))
class BeitFeatureExtractor(BeitImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["BeitFeatureExtractor"]
