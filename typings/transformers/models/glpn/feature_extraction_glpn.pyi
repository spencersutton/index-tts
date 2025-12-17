from ...utils.import_utils import requires
from .image_processing_glpn import GLPNImageProcessor

"""Feature extractor class for GLPN."""
logger = ...

@requires(backends=("vision",))
class GLPNFeatureExtractor(GLPNImageProcessor):
    def __init__(self, *args, **kwargs) -> None: ...

__all__ = ["GLPNFeatureExtractor"]
