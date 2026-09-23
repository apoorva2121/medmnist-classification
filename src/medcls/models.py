"""Two architectures sized for 28x28 inputs on CPU."""

from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """~120k params. Three conv blocks, global pooling, linear head."""

    def __init__(self, in_ch: int, n_classes: int) -> None:
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(block(in_ch, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class BasicBlock(nn.Module):
    def __init__(self, i, o, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(o)
        self.short = nn.Sequential()
        if stride != 1 or i != o:
            self.short = nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        out = torch.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return torch.relu(out + self.short(x))


class ResNet18_28(nn.Module):
    """ResNet-18 topology with a 3x3 stem and no initial max-pool, as MedMNIST uses for 28px."""

    def __init__(self, in_ch: int, n_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.features = nn.Sequential(self.stem, self.layer1, self.layer2, self.layer3, self.layer4)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def make_model(name: str, in_ch: int, n_classes: int) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN(in_ch, n_classes)
    if name == "resnet18_28":
        return ResNet18_28(in_ch, n_classes)
    raise ValueError(name)
