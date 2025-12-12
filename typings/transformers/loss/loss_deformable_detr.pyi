import torch
from ..utils import is_scipy_available
from .loss_for_object_detection import HungarianMatcher, ImageLoss

if is_scipy_available(): ...

class DeformableDetrHungarianMatcher(HungarianMatcher):
    @torch.no_grad()
    def forward(self, outputs, targets):  # -> list[tuple[Tensor, Tensor]]:

        ...

class DeformableDetrImageLoss(ImageLoss):
    def __init__(self, matcher, num_classes, focal_alpha, losses) -> None: ...
    def loss_labels(self, outputs, targets, indices, num_boxes):  # -> dict[str, Any]:

        ...

def DeformableDetrForSegmentationLoss(
    logits, labels, device, pred_boxes, pred_masks, config, outputs_class=..., outputs_coord=..., **kwargs
):  # -> tuple[int, Any, list[dict[str, Any]] | None]:
    ...
def DeformableDetrForObjectDetectionLoss(
    logits, labels, device, pred_boxes, config, outputs_class=..., outputs_coord=..., **kwargs
):  # -> tuple[int, Any, list[dict[str, Any]] | None]:
    ...
