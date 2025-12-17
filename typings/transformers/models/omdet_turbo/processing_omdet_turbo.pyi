from collections import UserDict
from typing import TYPE_CHECKING

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TensorType, is_torch_available, is_torchvision_available
from ...utils.import_utils import requires
from .modeling_omdet_turbo import OmDetTurboObjectDetectionOutput

"""
Processor class for OmDet-Turbo.
"""
if TYPE_CHECKING: ...

class OmDetTurboTextKwargs(TextKwargs, total=False):
    task: str | list[str] | TextInput | PreTokenizedInput | None

if is_torch_available(): ...
if is_torchvision_available(): ...

class OmDetTurboProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: OmDetTurboTextKwargs
    _defaults = ...

class DictWithDeprecationWarning(UserDict):
    message = ...
    def __getitem__(self, key): ...
    def get(self, key, *args, **kwargs): ...

def clip_boxes(box, box_size: tuple[int, int]):  # -> Tensor:

    ...
def compute_score(boxes):  # -> tuple[Tensor, Tensor]:

    ...

@requires(backends=("vision", "torchvision"))
class OmDetTurboProcessor(ProcessorMixin):
    attributes = ...
    image_processor_class = ...
    tokenizer_class = ...
    def __init__(self, image_processor, tokenizer) -> None: ...
    def __call__(
        self,
        images: ImageInput = ...,
        text: list[str] | list[list[str]] | None = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[OmDetTurboProcessorKwargs],
    ) -> BatchFeature: ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    def post_process_grounded_object_detection(
        self,
        outputs: OmDetTurboObjectDetectionOutput,
        text_labels: list[str] | list[list[str]] | None = ...,
        threshold: float = ...,
        nms_threshold: float = ...,
        target_sizes: TensorType | list[tuple] | None = ...,
        max_num_det: int | None = ...,
    ):  # -> list[Any]:

        ...

__all__ = ["OmDetTurboProcessor"]
