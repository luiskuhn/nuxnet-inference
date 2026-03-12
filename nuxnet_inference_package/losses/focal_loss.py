from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class FocalLoss(nn.Module):
    """Focal loss for dense multi-class segmentation.

    Expects logits/probabilities shaped [B, C, D, H, W] (or flattened variants)
    and labels shaped [B, D, H, W] or [B, 1, D, H, W].
    """

    def __init__(
        self,
        apply_nonlin: nn.Module | None = None,
        alpha: float | Sequence[float] | np.ndarray | None = None,
        gamma: float = 2.0,
        balance_index: int = 0,
        smooth: float = 1e-5,
        size_average: bool = True,
    ) -> None:
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None and (self.smooth < 0 or self.smooth > 1.0):
            raise ValueError("smooth value should be in [0,1]")

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_classes = logit.shape[1]

        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))

        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)

        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_classes, 1)
        elif isinstance(alpha, (list, tuple, np.ndarray)):
            if len(alpha) != num_classes:
                raise ValueError("alpha length must match num_classes")
            alpha = torch.tensor(alpha, dtype=torch.float32).view(num_classes, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha_tensor = torch.ones(num_classes, 1)
            alpha_tensor = alpha_tensor * (1 - alpha)
            alpha_tensor[self.balance_index] = alpha
            alpha = alpha_tensor
        else:
            raise TypeError("Not support alpha type")

        alpha = alpha.to(logit.device)

        idx = target.long().cpu()
        one_hot_key = torch.zeros(target.size(0), num_classes)
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth / (num_classes - 1), 1.0 - self.smooth)

        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        alpha = torch.squeeze(alpha[idx])
        loss = -1 * alpha * torch.pow((1 - pt), self.gamma) * logpt
        return loss.mean() if self.size_average else loss.sum()
