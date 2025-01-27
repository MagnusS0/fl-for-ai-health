import torch
import torch.nn as nn
from torch.nn import functional as F
from models.tiny_vit.tiny_vit import tiny_vit_5m_224
from models.tiny_segformer.segformer_head import SegFormerHead

class SegMed(nn.Module):
    """
    SegMed: Medical Image Segmentation model combining TinyViT encoder with SegFormer head
    """
    def __init__(self, img_size=224, in_chans=3, num_classes=1):
        super().__init__()

        embed_dims=[64, 128, 160, 320] # TinyViT's embed_dims
        
        # Initialize TinyViT encoder
        self.encoder = tiny_vit_5m_224(
            img_size=img_size,
            in_chans=in_chans,
            num_classes=0,  # Set to 0 to remove classification head
            output_intermediate_features=True
        )
        
        # Initialize SegFormer head
        self.decoder = SegFormerHead(
            in_channels=embed_dims,
            embed_dim=256,
            num_classes=num_classes,
            dropout_rate=0.1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.ReLU
        )


    def forward(self, image):
        # Get encoder features
        features = self.encoder.forward_features(image)

        # Pass through decoder head
        x = self.decoder(features)

        # Upsample to original image size
        output = F.interpolate(x, size=image.shape[2:], mode='bilinear', align_corners=False)
        
        return output

if __name__ == "__main__":
    model = SegMed(
        img_size=64,
        in_chans=3,
        num_classes=1
    )
    print(model)
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    assert y.shape == torch.Size([1, 1, 64, 64]), f"Expected shape (1, 1, 64, 64), but got {y.shape}"
    print(y.shape)