from typing import Union
import importlib
import logging
import torch.nn as nn
from torchvision import models
import torch
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

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
    

    
    


class TemplateClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
            self, 
            model: nn.Module, 
            num_channels=1, 
            num_classes=2, 
            device=None,
            criterion:Union[dict, str]="CrossEntropyLoss",
            optimizer:Union[dict, str]="SGD",
        ):
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.model = model
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = self._initialize_criterion(criterion)
        self.optimizer = self._initialize_optimizer(optimizer)
        self.loss_curve = []

    def _initialize_criterion(self, criterion):
        if isinstance(criterion, str):
            criterion_name = criterion
            criterion_params = {}
        elif isinstance(criterion, dict):
            criterion_name = criterion.get("name")
            criterion_params = {k: v for k, v in criterion.items() if k != "name"}
        else:
            raise ValueError("Criterion must be a string or a dictionary.")
        # Use torch.nn to get the criterion class
        module = importlib.import_module("torch.nn")
        criterion_class = getattr(module, criterion_name)
        criterion = criterion_class(**criterion_params)
        return criterion
    
    def _initialize_optimizer(self, optimizer):
        if isinstance(optimizer, str):
            optimizer_name = optimizer
            optimizer_params = {}
        elif isinstance(optimizer, dict):
            optimizer_name = optimizer.get("name")
            optimizer_params = {k: v for k, v in optimizer.items() if k != "name"}
        else:
            raise ValueError("Optimizer must be a string or a dictionary.")
        # Use torch.optim to get the optimizer class
        module = importlib.import_module("torch.optim")
        optimizer_class = getattr(module, optimizer_name)
        return optimizer_class(self.model.parameters(), **optimizer_params)
    
    def fit(self, X, y, epochs=1, batch_size=32, verbose=False, log_interval=10):
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(self.X_, self.y_)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        batch_index = 0
        # Move data and model to device
        self.X_ = self.X_.to(self.device)
        self.y_ = self.y_.to(self.device)
        self.model.to(self.device)
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
            for batch_X, batch_y in pbar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                batch_index += 1
                if verbose:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_index}, Loss: {loss.item()}")
            if epoch % log_interval == 0:
                train_loss = loss.item()
                self.loss_curve.append(train_loss)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss}")
        # Return the classifier
        return self
    
    
    
    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        self.model.eval()
        # Create DataLoader
        # Move data to device
        X = X.to(self.device)
        return self.model(X)
    
    
    def predict(self, X):
        self.predict_proba(X)
        with torch.no_grad():
            return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        with torch.no_grad():
            y_pred = self.predict(X)
            y_true = y.cpu().numpy()
            return accuracy_score(y_true, y_pred)

if __name__ == "__main__":
    # Example usage
    model = ResNet18(num_channels=1, num_classes=10)
    classifier = TemplateClassifier(model=model, num_channels=1, num_classes=10)
    # Dummy data
    X_dummy = torch.randn(100, 1, 28, 28)  # 100 samples of 1 channel 28x28 images
    y_dummy = torch.randint(0, 10, (100,))  # 100 labels for 10 classes
    # Fit the model
    classifier.fit(X_dummy, y_dummy, epochs=1, batch_size=32, verbose=True)
    # Predict
    predictions = classifier.predict(X_dummy)
    score = classifier.score(X_dummy, y_dummy)
    assert 1.0 >= score >=0.0, "Score should be between 0 and 1"