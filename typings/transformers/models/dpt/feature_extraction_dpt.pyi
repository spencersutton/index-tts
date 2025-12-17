from ...utils.import_utils import requires
from .image_processing_dpt import DPTImageProcessor

"""Feature extractor class for DPT."""
logger = ...

@requires(backends=("vision",))
class DPTFeatureExtractor(DPTImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["DPTFeatureExtractor"]
