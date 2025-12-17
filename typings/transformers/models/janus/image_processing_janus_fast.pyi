import torch
from torchvision.transforms import functional as F
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils_fast import BaseImageProcessorFast, DefaultFastImageProcessorKwargs
from ...image_utils import ImageInput, SizeDict
from ...processing_utils import Unpack
from ...utils import is_torch_available, is_torchvision_v2_available

if is_torch_available(): ...
if is_torchvision_v2_available(): ...

class JanusFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    min_size: int

class JanusImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    min_size = ...
    do_resize = ...
    do_rescale = ...
    do_normalize = ...
    valid_kwargs = JanusFastImageProcessorKwargs
    def __init__(self, **kwargs: Unpack[JanusFastImageProcessorKwargs]) -> None: ...
    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        min_size: int,
        interpolation: F.InterpolationMode = ...,
        antialias: bool = ...,
        **kwargs,
    ) -> torch.Tensor: ...
    def pad_to_square(
        self, images: torch.Tensor, background_color: int | tuple[int, int, int] = ...
    ) -> torch.Tensor: ...
    def postprocess(
        self,
        images: ImageInput,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_normalize: bool | None = ...,
        image_mean: list[float] | None = ...,
        image_std: list[float] | None = ...,
        return_tensors: str | None = ...,
    ) -> torch.Tensor: ...

__all__ = ["JanusImageProcessorFast"]
