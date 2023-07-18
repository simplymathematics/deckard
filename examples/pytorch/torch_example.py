import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

__all__ = [
    "ResNet18",
    "ResNet50",
    "ResNet101",
    "ResNet152",
]


def ResNet18(num_channels=3, num_classes=10):
    model = models.resnet18()
    model.conv1 = nn.Conv2d(
        num_channels,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )
    model.fc = nn.Linear(512, num_classes)
    return model


def ResNet50(num_channels=3, num_classes=10):
    model = models.resnet50()
    model.conv1 = nn.Conv2d(
        num_channels,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )
    model.fc = nn.Linear(2048, num_classes)
    return model


def ResNet101(num_channels=3, num_classes=10):
    model = models.resnet101()
    model.conv1 = nn.Conv2d(
        num_channels,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )
    model.fc = nn.Linear(2048, num_classes)
    return model


def ResNet152(num_channels=3, num_classes=10):
    model = models.resnet152()
    model.conv1 = nn.Conv2d(
        num_channels,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )
    model.fc = nn.Linear(2048, num_classes)
    return model
