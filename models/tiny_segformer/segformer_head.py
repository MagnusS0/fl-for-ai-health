# -------------------------------------------------------
# SegFormer Head
# Based on: https://github.com/NVlabs/SegFormer/tree/master
# -------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class SimpleConvModule(nn.Module):
    """
    A simplified ConvModule for Conv-Norm-Act pattern.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU,
    ):
        super().__init__()
        layers = OrderedDict()
        layers['conv'] = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=not norm_layer,
        )

        if norm_layer:
            layers['bn'] = norm_layer(out_channels)

        if act_layer:
            layers['act'] = act_layer(inplace=True)

        self.conv_block = nn.Sequential(layers)

    def forward(self, x):
        return self.conv_block(x)


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=320, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    Simpledfied version of the original SegFormer Head
    """

    def __init__(
        self,
        in_channels,
        embed_dim,
        num_classes,
        dropout_rate=0.1,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout_rate)

        # Linear layers for each scale
        self.linear_layers = nn.ModuleList(
            [MLP(in_chan, embed_dim) for in_chan in in_channels]
        )

        # Fusion and prediction
        self.linear_fuse = SimpleConvModule(
            embed_dim * len(in_channels),
            embed_dim,
            1,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, inputs):
        B, C, H, W = inputs[0].shape
        # Transform and reshape each feature level
        x = []
        for i in range(len(inputs)):
            feat = self.linear_layers[i](inputs[i])  # (B, H*W, embed_dim)
            feat = feat.permute(0, 2, 1).reshape(
                B, -1, inputs[i].shape[2], inputs[i].shape[3]
            )
            x.append(feat)

        # Upsample all to highest resolution
        x = [
            F.interpolate(i, size=(H, W), mode="bilinear", align_corners=False)
            for i in x
        ]

        x = torch.cat(x, dim=1)
        x = self.linear_fuse(x)
        x = self.dropout(x)
        x = self.linear_pred(x)
        return x


if __name__ == "__main__":
    model = SegFormerHead(in_channels=[64, 128, 160, 320], embed_dim=256, num_classes=1)
    print(model)

    # Create a list of feature maps at different scales
    x = [
        torch.randn(1, 64, 16, 16),
        torch.randn(1, 128, 8, 8),
        torch.randn(1, 160, 4, 4),
        torch.randn(1, 320, 2, 2),
    ]
    y = model(x)
    assert y.shape == (1, 1, 64, 64), (
        f"Output shape should be (1, 1, 64, 64) got {y.shape}"
    )
    print(y)
