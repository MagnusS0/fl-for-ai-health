import torch
from torch import nn
from collections import OrderedDict


class DefaultConv2d(nn.Conv2d):
    """
    Default convolutional layer with padding and initialization.

    Uses He normal initialization for the weights.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding="same",
        use_bias=False,
        init_type="he_normal",
    ):
        if padding == "same":
            padding = kernel_size // 2
        else:
            padding = 0
        super(DefaultConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=use_bias,
        )
        if init_type == "he_normal":
            nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        else:
            nn.init.xavier_normal_(self.weight)

        if use_bias:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = super(DefaultConv2d, self).forward(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with ReLU activation.
    """

    def __init__(
        self, in_channels, out_channels, strides=1, activation=nn.ReLU(), **kwargs
    ):
        super(ResidualBlock, self).__init__(**kwargs)
        self.strides = strides
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        main_layers = OrderedDict()
        main_layers['conv1'] = DefaultConv2d(in_channels, out_channels, stride=strides)
        main_layers['bn1'] = nn.BatchNorm2d(out_channels)
        main_layers['act1'] = activation
        main_layers['conv2'] = DefaultConv2d(out_channels, out_channels)
        main_layers['bn2'] = nn.BatchNorm2d(out_channels)
        self.main_layers = nn.Sequential(main_layers)

        # Skip layers with named layers
        self.skip_layers = nn.Sequential()
        if strides > 1 or in_channels != out_channels:
            skip_layers = OrderedDict()
            skip_layers['conv'] = DefaultConv2d(in_channels, out_channels, kernel_size=1, stride=strides)
            skip_layers['bn'] = nn.BatchNorm2d(out_channels)
            self.skip_layers = nn.Sequential(skip_layers)

    def forward(self, x):
        main = self.main_layers(x)
        skip = self.skip_layers(x)
        return self.activation(main + skip)


class ResNet18(nn.Module):
    """
    ResNet18 model with default convolutional layers and ReLU activation.
    """

    def __init__(self, in_channels=3, num_classes=10):
        super(ResNet18, self).__init__()
        self.conv1 = DefaultConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        prev_channels = 64
        self.blocks = nn.ModuleList()
        # Adjust here to change the number of layers in the network
        for filters in [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2:
            strides = 1 if filters == prev_channels else 2
            self.blocks.append(ResidualBlock(prev_channels, filters, strides))
            prev_channels = filters

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        for block in self.blocks:
            x = block(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = nn.functional.softmax(x, dim=1)
        return x


if __name__ == "__main__":
    model = ResNet18()
    # Print model architecture
    print(model)
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print(y.shape)
    print(y)
