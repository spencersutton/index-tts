import torch
from torchvision.transforms import functional as F
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast, DefaultFastImageProcessorKwargs
from ...image_utils import ImageInput, SizeDict
from ...processing_utils import Unpack
from ...utils import is_torch_available, is_torchvision_available, is_torchvision_v2_available

"""Fast Image processor class for SAM."""
if is_torch_available(): ...
if is_torchvision_available() and is_torchvision_v2_available(): ...

class SamFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    mask_size: dict[str, int] | None
    do_pad: bool | None
    pad_size: dict[str, int] | None
    mask_pad_size: dict[str, int] | None

class SamImageProcessorFast(BaseImageProcessorFast):
    resample = ...
    image_mean = ...
    image_std = ...
    size = ...
    mask_size = ...
    do_resize = ...
    do_rescale = ...
    do_normalize = ...
    do_convert_rgb = ...
    valid_kwargs = SamFastImageProcessorKwargs
    do_pad = ...
    pad_size = ...
    mask_pad_size = ...
    def __init__(self, **kwargs: Unpack[SamFastImageProcessorKwargs]) -> None: ...
    def pad_image(self, images: torch.Tensor, pad_size: SizeDict): ...
    def resize(
        self, image: torch.Tensor, size: SizeDict, interpolation: F.InterpolationMode | None, **kwargs
    ) -> torch.Tensor: ...
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None = ...,
        **kwargs: Unpack[SamFastImageProcessorKwargs],
    ) -> BatchFeature: ...
    def generate_crop_boxes(
        self,
        image: torch.Tensor,
        target_size,
        crop_n_layers: int = ...,
        overlap_ratio: float = ...,
        points_per_crop: int | None = ...,
        crop_n_points_downscale_factor: list[int] | None = ...,
        device: torch.device | None = ...,
    ):  # -> tuple[Any, Any | int | None, Any, Any]:

        ...
    def filter_masks(
        self,
        masks,
        iou_scores,
        original_size,
        cropped_box_image,
        pred_iou_thresh=...,
        stability_score_thresh=...,
        mask_threshold=...,
        stability_score_offset=...,
    ):  # -> tuple[list[Any], Any, Tensor]:

        ...
    def post_process_masks(
        self, masks, original_sizes, reshaped_input_sizes, mask_threshold=..., binarize=..., pad_size=...
    ):  # -> list[Any]:

        ...
    def post_process_for_mask_generation(
        self, all_masks, all_scores, all_boxes, crops_nms_thresh
    ):  # -> tuple[list[Tensor], Any, list[Any], Any]:

        ...

__all__ = ["SamImageProcessorFast"]
