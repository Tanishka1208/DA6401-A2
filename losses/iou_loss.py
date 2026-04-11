import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    Expected format: [x_center, y_center, width, height]
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.eps = eps

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        # ========================
        # Ensure positive width/height
        # ========================
        pred_w = torch.clamp(pred_boxes[:, 2], min=self.eps)
        pred_h = torch.clamp(pred_boxes[:, 3], min=self.eps)

        target_w = torch.clamp(target_boxes[:, 2], min=self.eps)
        target_h = torch.clamp(target_boxes[:, 3], min=self.eps)

        # ========================
        # Convert to (x1, y1, x2, y2)
        # ========================
        pred_x1 = pred_boxes[:, 0] - pred_w / 2
        pred_y1 = pred_boxes[:, 1] - pred_h / 2
        pred_x2 = pred_boxes[:, 0] + pred_w / 2
        pred_y2 = pred_boxes[:, 1] + pred_h / 2

        target_x1 = target_boxes[:, 0] - target_w / 2
        target_y1 = target_boxes[:, 1] - target_h / 2
        target_x2 = target_boxes[:, 0] + target_w / 2
        target_y2 = target_boxes[:, 1] + target_h / 2

        # ========================
        # Intersection
        # ========================
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h

        # ========================
        # Areas
        # ========================
        pred_area = pred_w * pred_h
        target_area = target_w * target_h

        union_area = pred_area + target_area - inter_area
        union_area = torch.clamp(union_area, min=self.eps)

        # ========================
        # IoU
        # ========================
        iou = inter_area / union_area
        loss = 1 - iou

        # ========================
        # Reduction
        # ========================
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss