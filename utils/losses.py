import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import F1Score, BinaryF1Score


class DiceLoss(nn.Module):
    """Dice Loss for image segmentation"""

    def __init__(self, smooth=1e-5, ignore_index=0, num_classes=4, device=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.f1_score = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
        self.binary_f1_score = BinaryF1Score().to(device)

    def forward(self, outputs, targets):
        if self.num_classes > 1:
            outputs = F.softmax(outputs, dim=1)
            return 1.0 - self.f1_score(outputs, targets) + self.smooth
        else:
            return 1.0 - self.binary_f1_score(outputs, targets) + self.smooth


class FocalLoss(nn.Module):
    """Focal Loss for image segmentation"""

    def __init__(self, gamma=2, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha if alpha is None else torch.tensor(alpha)
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(self, outputs, targets):
        ce_loss = self.cross_entropy(outputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(targets.device)[targets.view(-1)].view(targets.shape)
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class DiceFocalLoss(nn.Module):
    """
    Dice Focal Loss combines Dice Loss and Focal Loss
    
    Args:
        lambda_dice (float): Weight for Dice Loss
        lambda_focal (float): Weight for Focal Loss
        dice_smooth (float): Smoothing factor for Dice Loss
        focal_gamma (float): Gamma parameter for Focal Loss
        focal_alpha (float): Alpha parameter for Focal Loss
        num_classes (int): Number of classes
        ignore_index (int): Index to ignore in loss calculation
        device (str): Device to use for loss calculation
    """

    def __init__(
        self,
        lambda_dice=0.5,
        lambda_focal=0.5,
        dice_smooth=1e-5,
        focal_gamma=2,
        focal_alpha=None,
        num_classes=4,
        ignore_index=0,
        device=None,
    ):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.dice_loss = DiceLoss(smooth=dice_smooth, ignore_index=ignore_index, num_classes=num_classes, device=device)
        self.focal_loss = FocalLoss(
            gamma=focal_gamma, alpha=focal_alpha, reduction="mean"
        )

    def forward(self, outputs, targets):
        dice_loss = self.dice_loss(outputs, targets)
        focal_loss = self.focal_loss(outputs, targets)
        return self.lambda_dice * dice_loss + self.lambda_focal * focal_loss
