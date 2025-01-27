import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for image segmentation"""

    def __init__(self, smooth=1e-5, ignore_index=0):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        num_classes = outputs.shape[1]
        if num_classes > 1:
            outputs = F.softmax(outputs, dim=1)
            # Labels to one-hot format
            targets_one_hot = (
                F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
            )

            dice_loss = 0.0
            valid_classes = 0
            for class_index in range(num_classes):
                if class_index == self.ignore_index:
                    continue
                output_class = outputs[:, class_index, :, :]
                target_class = targets_one_hot[:, class_index, :, :]
                intersection = (output_class * target_class).sum()
                cardinality = output_class.sum() + target_class.sum()
                dice_score = (2.0 * intersection + self.smooth) / (
                    cardinality + self.smooth
                )
                dice_loss += 1.0 - dice_score
                valid_classes += 1
            return dice_loss / valid_classes  # Average over non-background classes

        else:  # Binary case
            outputs = torch.sigmoid(outputs).squeeze(1)  # Squeeze to [B, H, W]
            intersection = (outputs * targets).sum()
            cardinality = outputs.sum() + targets.sum()
            dice_score = (2.0 * intersection + self.smooth) / (
                cardinality + self.smooth
            )
            return 1.0 - dice_score


class FocalLoss(nn.Module):
    """Focal Loss for image segmentation"""

    def __init__(self, gamma=2, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(self, outputs, targets):
        ce_loss = self.cross_entropy(outputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # Get alpha for each target class
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class DiceFocalLoss(nn.Module):
    """Dice Focal Loss combines Dice Loss and Focal Loss"""

    def __init__(
        self,
        lambda_dice=0.5,
        lambda_focal=0.5,
        dice_smooth=1e-5,
        focal_gamma=2,
        focal_alpha=None,
    ):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.dice_loss = DiceLoss(smooth=dice_smooth, ignore_index=0)
        self.focal_loss = FocalLoss(
            gamma=focal_gamma, alpha=focal_alpha, reduction="mean"
        )

    def forward(self, outputs, targets):
        dice_loss = self.dice_loss(outputs, targets)
        focal_loss = self.focal_loss(outputs, targets)
        return self.lambda_dice * dice_loss + self.lambda_focal * focal_loss
