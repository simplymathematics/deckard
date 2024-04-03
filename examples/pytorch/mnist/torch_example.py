import torch.nn as nn
from torchvision import models

__all__ = [
    "ResNet18",
    "ResNet50",
    "ResNet34",
    "ResNet101",
    "ResNet152",
]


def ResNet18(num_channels=1, num_classes=10):
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


def ResNet34(num_channels=1, num_classes=10):
    model = models.resnet34()
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


def ResNet50(num_channels=1, num_classes=10):
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


def ResNet101(num_channels=1, num_classes=10):
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


def ResNet152(num_channels=1, num_classes=10):
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

def VGG16(num_channels=1, num_classes=10):
    model = models.vgg16()
    model.features[0] = nn.Conv2d(
        num_channels,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def VGG19(num_channels=1, num_classes=10):
    model = models.vgg19()
    model.features[0] = nn.Conv2d(
        num_channels,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def AlexNet(num_channels=1, num_classes=10):
    model = models.alexnet()
    model.features[0] = nn.Conv2d(
        num_channels,
        64,
        kernel_size=11,
        stride=4,
        padding=2,
    )
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def MobileNet(num_channels=1, num_classes=10):
    model = models.mobilenet_v2()
    model.features[0][0] = nn.Conv2d(
        num_channels,
        32,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False,
    )
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model

def Inceptionv3(num_channels=1, num_classes=10):
    model = models.inception_v3()
    model.Conv2d_1a_3x3 = nn.Conv2d(
        num_channels,
        32,
        kernel_size=3,
        stride=2,
    )
    model.fc = nn.Linear(2048, num_classes)
    return model

def calculate_number_of_layers(model):
    return sum(1 for _ in model.parameters())

def calculate_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters())

def calculate_number_of_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_number_of_non_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)

def calculate_number_of_trainable_layers(model):
    return sum(1 for p in model.parameters() if p.requires_grad)

def calculate_number_of_non_trainable_layers(model):
    return sum(1 for p in model.parameters() if not p.requires_grad)

def calculate_all_model_metrics(model):
    return {
        "number_of_layers": calculate_number_of_layers(model),
        "number_of_parameters": calculate_number_of_parameters(model),
        "number_of_trainable_parameters": calculate_number_of_trainable_parameters(model),
        "number_of_non_trainable_parameters": calculate_number_of_non_trainable_parameters(model),
        "number_of_trainable_layers": calculate_number_of_trainable_layers(model),
        "number_of_non_trainable_layers": calculate_number_of_non_trainable_layers(model),
    }