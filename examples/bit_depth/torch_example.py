from torchvision import models
import torch

__all__ = [
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "LogisticRegression",
]


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dtype="torch.FloatTensor"):
        self.dtype = dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.type(torch.FloatTensor)
        outputs = torch.sigmoid(self.linear(x))
        return outputs


def ResNet18(num_channels=1, num_classes=10):
    model = models.resnet18()
    model.conv1 = torch.nn.Conv2d(
        num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False,
    )
    model.fc = torch.nn.Linear(512, num_classes)
    return model


def ResNet34(num_channels=1, num_classes=10):
    model = models.resnet34()
    model.conv1 = torch.nn.Conv2d(
        num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False,
    )
    model.fc = torch.nn.Linear(512, num_classes)
    return model


def ResNet50(num_channels=1, num_classes=10):
    model = models.resnet50()
    model.conv1 = torch.nn.Conv2d(
        num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False,
    )
    model.fc = torch.nn.Linear(2048, num_classes)
    return model


def ResNet101(num_channels=1, num_classes=10):
    model = models.resnet101()
    model.conv1 = torch.nn.Conv2d(
        num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False,
    )
    model.fc = torch.nn.Linear(2048, num_classes)
    return model


def ResNet152(num_channels=1, num_classes=10):
    model = models.resnet152()
    model.conv1 = torch.nn.Conv2d(
        num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False,
    )
    model.fc = torch.nn.Linear(2048, num_classes)
    return model
