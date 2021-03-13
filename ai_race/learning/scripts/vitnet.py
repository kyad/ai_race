import torch.nn as nn

from vit2 import ViT2

class ViT2Net(nn.Module):
    def __init__(self):
        super(ViT2Net, self).__init__()
        self.bn = nn.BatchNorm2d(3)
        self.vit2 = ViT2(
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
        x = self.vit2(x)
        return x
