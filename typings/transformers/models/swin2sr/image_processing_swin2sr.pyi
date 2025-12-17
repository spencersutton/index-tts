import numpy as np

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ChannelDimension, ImageInput
from ...utils import TensorType, filter_out_non_signature_kwargs

"""Image processor class for Swin2SR."""
logger = ...

class Swin2SRImageProcessor(BaseImageProcessor):
    model_input_names = ...
    def __init__(
        self,
        do_rescale: bool = ...,
        rescale_factor: float = ...,
        do_pad: bool = ...,
        pad_size: int = ...,
        **kwargs,
    ) -> None: ...
    def pad(
        self,
        image: np.ndarray,
        size: int,
        data_format: str | ChannelDimension | None = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> ndarray[_AnyShape, dtype[Any]]:

        ...
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_rescale: bool | None = ...,
        rescale_factor: float | None = ...,
        do_pad: bool | None = ...,
        pad_size: int | None = ...,
        return_tensors: str | TensorType | None = ...,
        data_format: str | ChannelDimension = ...,
        input_data_format: str | ChannelDimension | None = ...,
    ):  # -> BatchFeature:

        ...

__all__ = ["Swin2SRImageProcessor"]
