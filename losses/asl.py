"""Asymmetric Loss (ASL) for multi-label classification.

Reference implementation from the official repo
https://github.com/Alibaba-MIIL/ASL (asl_losses.py), adapted for raw logits
input.  Handles the severe OMI class imbalance (only ~6.4% positives) by
down-weighting easy negatives via ``gamma_neg`` and a probability margin
``clip``.
"""
import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8,
                 disable_torch_grad_focal_loss=True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        """Args:
            x: [B, C] raw logits.
            y: [B, C] targets in {0, 1}.
        Returns:
            scalar loss (sum over all elements; divide by batch size as needed).
        """
        x = x.float()
        y = y.float()

        p = torch.sigmoid(x)
        xs_pos = p
        xs_neg = 1.0 - p
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Basic cross-entropy terms (with eps for numerical stability).
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric focusing.
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1.0 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()
