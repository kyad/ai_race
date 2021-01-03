import torch.nn as nn

from vit_pytorch import ViT

class ViTNet(nn.Module):
    def __init__(self):
        super(ViTNet, self).__init__()
        self.bn = nn.BatchNorm2d(3)
        self.vit = ViT(
            image_size=(240, 320),
            patch_size=40,
            num_classes=3,
            dim=32,
            depth=1,
            heads=2,
            mlp_dim=32,
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.vit(x)
        return x
