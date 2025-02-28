# ---------------------------
# U-Net model implementation.
# Based on https://github.com/milesial/Pytorch-UNet/tree/master
# ---------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ConvDouble(nn.Module):
    """
    2 * 3x3 conv layers with batch normalization and ReLU activation.
    Using padding=1 to maintain feature map size.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        layers = OrderedDict()
        layers["conv1"] = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        layers["bn1"] = nn.BatchNorm2d(out_channels)
        layers["act1"] = nn.ReLU()
        layers["conv2"] = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        layers["bn2"] = nn.BatchNorm2d(out_channels)
        layers["act2"] = nn.ReLU()

        self.conv_double = nn.Sequential(layers)

    def forward(self, x):
        return self.conv_double(x)


class DownPool(nn.Module):
    """
    Max pooling layer with 2x2 kernel.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), ConvDouble(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_pool(x)


class UpPool(nn.Module):
    """
    Upsampling layer with concatenation of contracting path features.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = ConvDouble(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle different sizes for concatenation
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 convolution layer to get the output.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net model with up and down paths.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.in_conv = ConvDouble(in_channels, 64)
        self.down1 = DownPool(64, 128)
        self.down2 = DownPool(128, 256)
        self.down3 = DownPool(256, 512)
        self.up1 = UpPool(512, 256)
        self.up2 = UpPool(256, 128)
        self.up3 = UpPool(128, 64)
        self.out_conv = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.out_conv(x)


if __name__ == "__main__":
    model = UNet(in_channels=1, num_classes=4)
    print(model)
    x = torch.randn(1, 1, 64, 64)
    y = model(x)
    print(y.shape)
    print(sum(p.numel() for p in model.parameters()))
