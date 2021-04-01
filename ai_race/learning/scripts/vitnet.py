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


class ViT2NetS(nn.Module):
    def __init__(self):
        super(ViT2NetS, self).__init__()
        self.bn = nn.BatchNorm2d(3)
        self.vit2 = ViT2(
            image_size=(240, 320),
            patch_size=40,
            num_classes=3,
            dim=64,
            depth=2,
            heads=4,
            mlp_dim=64,
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.vit2(x)
        return x


class ViT2NetM(nn.Module):
    def __init__(self):
        super(ViT2NetM, self).__init__()
        self.bn = nn.BatchNorm2d(3)
        self.vit2 = ViT2(
            image_size=(240, 320),
            patch_size=40,
            num_classes=3,
            dim=96,
            depth=3,
            heads=6,
            mlp_dim=96,
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.vit2(x)
        return x


class ViT2NetL(nn.Module):
    def __init__(self):
        super(ViT2NetL, self).__init__()
        self.bn = nn.BatchNorm2d(3)
        self.vit2 = ViT2(
            image_size=(240, 320),
            patch_size=40,
            num_classes=3,
            dim=128,
            depth=4,
            heads=8,
            mlp_dim=128,
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.vit2(x)
        return x
