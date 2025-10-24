
import logging
import torch.nn as nn
from torchvision import models
import torch
from dataclasses import dataclass

from deckard.data.pytorch import PytorchDataConfig
from deckard.model.pytorch import PytorchTemplateClassifier, PytorchModelConfig

__all__ = [
    "ResNet18",
]
logger = logging.getLogger(__name__)

# You can edit your model architecture here
# For example, a simple ResNet18 model for image classification

@dataclass
class ResNet18(nn.Module):
    num_channels: int = 3
    num_classes: int = 1000
    
    def __init__(self, num_channels: int = 3, num_classes: int = 1000):
        self.num_channels = num_channels
        self.num_classes = num_classes
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(
            self.num_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.model.fc = nn.Linear(512, self.num_classes)
        self.loss_curve = []

    
    def forward(self, x):
        return self.model(x)
    
    def __hash__(self):
        arch = str(self.model)
        params = f"num_channels={self.num_channels}, num_classes={self.num_classes}"
        model_params = b"".join(p.detach().cpu().numpy().tobytes() for p in self.model.parameters())
        return hash((arch, params, model_params))


if __name__ == "__main__":
    # Example usage
    model = ResNet18(num_channels=1, num_classes=10)
    classifier = PytorchTemplateClassifier(
        model=model,
        criterion="CrossEntropyLoss",
        optimizer="SGD",
    )
    data_conf = PytorchDataConfig(
        dataset_name="torch_mnist",
        train_size=128,
        test_size=128,
        random_state=128,
        classifier=True,
        stratify=True,
    )
    data_conf() # Initialize data
    classifier.fit(data_conf.X_train, data_conf.y_train, epochs=1, batch_size=32, verbose=True)
    # Predict
    predictions = classifier.predict(data_conf.X_test)
    score = (predictions == data_conf.y_test).float().mean().item()
    print(f"Accuracy on dummy data: {score}")
    assert score >=0.0, "Score should be between 0 and 1"
    
    
    assert len(data_conf.X_train) == 128, f"Expected 128 training samples, got {len(data_conf.X_train)}"
    assert len(data_conf.X_test) == 128, f"Expected 128 test samples, got {len(data_conf.X_test)}"
    model_conf = PytorchModelConfig(
        model_type="torch_example.ResNet18", # file.ClassName
        model_params={
            "num_channels": 1,
            "num_classes": 10,
        },
        criterion="CrossEntropyLoss",
        optimizer="SGD",
        classifier=True,
        fit_params={
            "epochs": 100,
            "verbose": True,
            "log_interval": 10,
        },
        
    )
    
    new_score = model_conf(data_conf)['accuracy']
    print(f"Score from model config on dummy data: {new_score}")
    assert new_score >=0.0, "Score should be between 0 and 1"