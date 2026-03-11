from __future__ import annotations

import numpy as np
import torch


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Voxel-wise accuracy for segmentation tensors."""
    compare = pred.float() == target.float()
    return compare.sum().item() / len(pred.flatten())


def iou_fnc(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Per-class IoU for segmentation predictions and targets."""
    ious: list[float] = []
    pred = pred.view(-1)
    target = target.view(-1)

    count = np.zeros(num_classes)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds[target_inds]).long().sum().cpu().item()
        union = pred_inds.long().sum().cpu().item() + target_inds.long().sum().cpu().item() - intersection

        if union == 0:
            ious.append(0.0)
        else:
            count[cls] += 1
            ious.append(float(intersection) / float(max(union, 1)))

    return np.array(ious), count
