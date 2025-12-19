from ...utils.import_utils import requires
from .image_processing_detr import DetrImageProcessor

"""Feature extractor class for DETR."""
logger = ...

def rgb_to_id(x):  # -> NDArray[signedinteger[Any]] | NDArray[signedinteger[_32Bit]] | NDArray[Any] | int:
    ...

@requires(backends=("vision",))
class DetrFeatureExtractor(DetrImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["DetrFeatureExtractor"]
