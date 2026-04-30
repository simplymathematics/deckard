# OS imports
import logging
import time

# Typing imports
from dataclasses import dataclass, field
from omegaconf import DictConfig
from typing import Union
import numpy as np

# Torch imports
import torch


# Sklearn imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ART imports
from art.estimators.classification import PyTorchClassifier
from art.estimators.regression import PyTorchRegressor

from . import ModelConfig
from ..data import DataConfig
from ..utils import load_class

logger = logging.getLogger(__name__)

__all__ = ["PytorchModelConfig"]

def initialize_criterion(criterion_spec):
    """Initialize criterion from string or dict using load_class."""
    if isinstance(criterion_spec, str):
        if "." not in criterion_spec and ":" not in criterion_spec:
            criterion_spec = f"torch.nn.{criterion_spec}"
        return load_class(criterion_spec)
    elif isinstance(criterion_spec, (dict, DictConfig)):
        criterion_name = criterion_spec.get("name") or criterion_spec.get("_target_")
        criterion_params = {k: v for k, v in criterion_spec.items() if k not in ["name", "_target_"]}
        return load_class(criterion_name, **criterion_params)
    else:
        raise ValueError(f"criterion must be str or dict, got {type(criterion_spec)}")


def initialize_optimizer(optimizer_spec, model_params):
    """Initialize optimizer from string or dict using load_class."""
    if isinstance(optimizer_spec, str):
        if "." not in optimizer_spec and ":" not in optimizer_spec:
            optimizer_spec = f"torch.optim.{optimizer_spec}"
        return load_class(optimizer_spec, model_params)
    elif isinstance(optimizer_spec, (dict, DictConfig)):
        optimizer_name = optimizer_spec.get("name") or optimizer_spec.get("_target_")
        if isinstance(optimizer_name, str) and "." not in optimizer_name and ":" not in optimizer_name:
            optimizer_name = f"torch.optim.{optimizer_name}"
        optimizer_params = {k: v for k, v in optimizer_spec.items() if k not in ["name", "_target_"]}
        optimizer_params["params"] = model_params
        return load_class(optimizer_name, **optimizer_params)
    else:
        raise ValueError(f"optimizer must be str or dict, got {type(optimizer_spec)}")


@dataclass
class PytorchModelConfig(ModelConfig):
    """Configuration for PyTorch models using load_class for generic instantiation.
    
    Attributes:
        model_type: Fully qualified class name (e.g., "torch.nn.Linear" or "torchvision.models.resnet18")
        model_params: Constructor parameters for the model
        device: torch.device ("cpu", "cuda", etc.)
        criterion: Loss function spec (str name or dict with _target_)
        optimizer: Optimizer spec (str name or dict with _target_)
        fit_params: Parameters for model training
        classifier: Whether model is classifier (True) or regressor (False)
    """
    
    model_type: str = "torch.nn.Linear"
    model_params: dict = field(default_factory=dict)
    classifier: bool = True
    fit_params: dict = field(default_factory=dict)
    library: str = "pytorch"
    device: Union[str, torch.device] = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    criterion: Union[dict, str] = field(default="torch.nn.CrossEntropyLoss")
    optimizer: Union[dict, str] = field(default="torch.optim.SGD")
    clip_values: Union[tuple, None] = None
    random_seed: int = 42
    channels_first: bool = True

    def __post_init__(self):
        """Initialize model using load_class, following parent pattern."""
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        
        torch.manual_seed(self.random_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_seed)
        
        # Call parent __post_init__ for shared initialization
        super().__post_init__()

    def _initialize_model(self):
        """Initialize PyTorch model using load_class."""
        if self.model_params is not None:
            self._model = load_class(self.model_type, **self.model_params)
        else:
            self._model = load_class(self.model_type)
        
        # Move model to device
        self._model = self._model.to(self.device)
        logger.info(f"Initialized model {self.model_type} on device {self.device}")

    def get_model(self):
        """Return the underlying PyTorch model."""
        if self._model is None:
            raise ValueError("Model not initialized")
        return self._model

    def _train(self, X: torch.Tensor, y: torch.Tensor):
        """Train the PyTorch model."""
        if self._model is None:
            raise ValueError("Model not initialized")
        
        start_time = time.process_time()
        
        criterion = initialize_criterion(self.criterion)
        optimizer = initialize_optimizer(self.optimizer, self._model.parameters())
        
        nb_epochs = self.fit_params.get("nb_epochs", 1)
        batch_size = self.fit_params.get("batch_size", 32)
        
        self._model.train()
        for epoch in range(nb_epochs):
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size].to(self.device)
                batch_y = y[i:i+batch_size].to(self.device)
                
                optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        end_time = time.process_time()
        self.training_time = end_time - start_time
        self.training_n = len(y)
        logger.info(f"Model trained in {self.training_time:.2f} seconds")

    def _predict(self, X: Union[torch.Tensor, torch.utils.data.DataLoader]):
        """Make predictions, handling both Tensor and DataLoader inputs."""
        if self._model is None:
            raise ValueError("Model not initialized")
        
        self._model.eval()
        predictions = []
        
        with torch.no_grad():
            if isinstance(X, torch.utils.data.DataLoader):
                for batch in X:
                    if isinstance(batch, (tuple, list)):
                        batch_X = batch[0]
                    else:
                        batch_X = batch
                    batch_X = batch_X.to(self.device)
                    batch_pred = self._model(batch_X)
                    predictions.append(batch_pred.cpu())
            else:
                X = X.to(self.device)
                predictions.append(self._model(X).cpu())
        
        y_pred = torch.cat(predictions, dim=0)
        return y_pred

    def _classification_scores(self, y_true, y_pred) -> dict:
        """Compute classification scores from predictions."""
        y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
        
        if isinstance(y_pred, torch.Tensor):
            if y_pred.ndim > 1:
                y_pred_np = y_pred.argmax(dim=1).cpu().numpy()
            else:
                y_pred_np = y_pred.cpu().numpy()
        else:
            y_pred_np = y_pred
        
        scores = {
            "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
            "precision": float(precision_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        }
        return scores

    def _regression_scores(self, y_true, y_pred) -> dict:
        """Compute regression scores from predictions."""
        y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
        y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
        
        mse = float(np.mean((y_true_np - y_pred_np) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_true_np - y_pred_np)))
        
        scores = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
        }
        return scores

    def get_art_model(self, data: DataConfig):
        """Get ART-compatible model wrapper for adversarial robustness."""
        if self.clip_values is None or len(self.clip_values) == 0:
            clip_values = (0.0, 1.0)
        else:
            clip_values = self.clip_values
        
        if isinstance(data.X_train, torch.utils.data.DataLoader):
            batch = next(iter(data.X_train))
            if isinstance(batch, (tuple, list)):
                input_shape = batch[0][0].shape[1:]
            else:
                input_shape = batch[0].shape[1:]
        else:
            input_shape = data.X_train.shape[1:]
        
        nb_classes = len(torch.unique(data.y_train))
        if self.classifier:
            return PyTorchClassifier(
                model=self._model,
                loss=initialize_criterion(self.criterion),
                optimizer=initialize_optimizer(self.optimizer, self._model.parameters()),
                input_shape=input_shape,
                nb_classes=nb_classes,
                clip_values=clip_values,
                device_type="gpu" if "cuda" in str(self.device) else "cpu",
            )
        else:
            return PyTorchRegressor(
                model=self._model,
                loss=initialize_criterion(self.criterion),
                optimizer=initialize_optimizer(self.optimizer, self._model.parameters()),
                input_shape=input_shape,
                nb_classes=nb_classes,
                clip_values=clip_values,
                device_type="gpu" if "cuda" in str(self.device) else "cpu",
            )