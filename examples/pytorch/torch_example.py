
import logging
import torch.nn as nn
from torchvision import models
import torch


from deckard.data.pytorch import PytorchDataConfig
from deckard.model.pytorch import PytorchTemplateClassifier, PytorchModelConfig

__all__ = [
    "ResNet18",
]
logger = logging.getLogger(__name__)

# You can edit your model architecture here
# For example, a simple ResNet18 model for image classification
class ResNet18(nn.Module):
    def __init__(self, num_channels=1, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(
            num_channels,
            64,
            kernel_size=7,
            stride=2,
        padding=3,
        bias=False,
    )
        self.model.fc = nn.Linear(512, num_classes)
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.loss_curve = []
        self.val_scores = []
    
    def forward(self, x):
        return self.model(x)
    

    
    




if __name__ == "__main__":
    # Example usage
    model = ResNet18(num_channels=1, num_classes=10)
    classifier = PytorchTemplateClassifier(
        model=model,
        criterion="CrossEntropyLoss",
        optimizer="SGD",
    )
    # Dummy data
    n_samples = 100
    X_dummy = torch.randn(n_samples, 1, 28, 28)  # 100 samples of 1 channel 28x28 images
    y_dummy = torch.randint(0, 10, (n_samples,))  # 100 labels for 10 classes
    # Fit the model
    classifier.fit(X_dummy, y_dummy, epochs=1, batch_size =32, verbose=True)
    # Predict
    predictions = classifier.predict(X_dummy)
    score = (predictions == y_dummy).float().mean().item()
    print(f"Accuracy on dummy data: {score}")
    assert score >=0.0, "Score should be between 0 and 1"
    
    class DummyDataConfig(PytorchDataConfig):
        def __call__(self, **kwargs):
            self.X_train = X_dummy
            self.y_train = y_dummy
            self.X_test = X_dummy
            self.y_test = y_dummy
            return {}
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
    data_conf = DummyDataConfig()
    data_conf() # Initialize data
    new_score = model_conf(data_conf, model_file ="tmp.pkl")['accuracy']
    print(f"Score from model config on dummy data: {new_score}")
    assert new_score >=0.0, "Score should be between 0 and 1"