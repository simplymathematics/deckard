from typing import Union
import sys
import os
import importlib
import logging
import torch.nn as nn
from dataclasses import dataclass, field
import torch
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from omegaconf import DictConfig
from pathlib import Path
from . import ModelConfig
logger = logging.getLogger(__name__)

supported_sklearn_libraries = ["sklearn"]

__all__ = ["PytorchModelConfig"]




class PytorchTemplateClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
            self, 
            model: nn.Module, 
            device=None,
            criterion:Union[dict, str]="CrossEntropyLoss",
            optimizer:Union[dict, str]="SGD",
        ):
        self.model = model
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.criterion = self._initialize_criterion(criterion)
        self.optimizer = self._initialize_optimizer(optimizer)
        self.loss_curve = []

    def _set_random_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _initialize_criterion(self, criterion):
        if isinstance(criterion, str):
            criterion_name = criterion
            criterion_params = {}
        elif isinstance(criterion, (dict, DictConfig)):
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
        elif isinstance(optimizer, (dict, DictConfig)):
            optimizer_name = optimizer.get("name")
            optimizer_params = {k: v for k, v in optimizer.items() if k != "name"}
        else:
            raise ValueError("Optimizer must be a string or a dictionary.")
        # Use torch.optim to get the optimizer class
        module = importlib.import_module("torch.optim")
        optimizer_class = getattr(module, optimizer_name)
        return optimizer_class(self.model.parameters(), **optimizer_params)
    
    def to(self, device):
        self.device = device
        self.model.to(device)
    
    def fit(self, X, y, epochs=1, batch_size=32, verbose=False, log_interval=10):
        # Store the classes seen during fit
        self.classes_ = torch.unique(y)
        batch_index = 0
        # Move data and model to device
        self.model.to(self.device)
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            pbar = tqdm(zip(X.split(batch_size), y.split(batch_size)), total=len(X) // batch_size, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_X, batch_y in pbar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
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
        # Move data to device
        X = X.to(self.device)
        with torch.no_grad():
            return self.model(X)
    
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return torch.argmax(probs, dim=1)

    def score(self, X, y):
        check_is_fitted(self)
        self.model.eval()
        X = X.to(self.device)
        y = y.to(self.device)
        with torch.no_grad():
            y_pred = self.model(X)
            y_pred = torch.tensor(y_pred).to(self.device)
            loss_function = self.criterion
            score = loss_function(y_pred, y).item()
            score /= len(y)
            return score

@dataclass
class PytorchModelConfig(ModelConfig):
    model_type: str = "torch_example.ResNet18"
    model_params: dict = field(default_factory=dict)
    classifier: bool = False
    fit_params: dict = field(default_factory=dict)
    library: str = "pytorch"
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    criterion: Union[dict, str] = field(default="CrossEntropyLoss")
    optimizer: Union[dict, str] = field(default="SGD")
    clip_values: Union[tuple, None] = None
    random_seed: int = 42
    """
    Overview
    --------
    Configuration for PyTorch models.

    Attributes
    ----------
    model_type : str
        The full class name of the PyTorch model (e.g., 'torch_example.ResNet18').
    model_params : dict
        Parameters to initialize the PyTorch model.
    classifier : bool
        Whether the model is a classifier.
    fit_params : dict
        Parameters for the fit method.
    library : str
        The deep learning library to use (default is 'pytorch').
    criterion : dict
        Dictionary specifying the loss function.
    optimizer : dict
        Dictionary specifying the optimizer.
    clip_values : tuple
        Tuple of the form (min, max) to clip input features.
    

    Methods
    -------
    __post_init__(): Initializes the model based on the provided type and parameters.
    __hash__(): Computes a hash value for the instance based on its attributes.
    _train(X, y): Trains the model using the provided feature matrix and target vector.
    _predict(X): Generates predictions for the input data.
    _predict_proba(X): Predicts class probabilities for the input data (if supported).
    _classification_scores(y_true, y_pred): Computes classification metrics.
    _regression_scores(y_true, y_pred): Computes regression metrics.
    _score(y_true, y_pred, train): Computes and logs performance scores.
    __call__(X, y, train, score, filepath): Executes the model workflow including training, prediction, scoring, and model persistence.
    """
    
    def to(self, device):
        self.device = device
        self._model.to(device)
    
    def load_class(self, file_path, class_name, module_name=None):
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Module file {file_path} does not exist.")
        if module_name is None:
            module_name = os.path.splitext(os.path.basename(file_path))[0]

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module   # <-- this is the key line
        spec.loader.exec_module(module)
        return getattr(module, class_name)

    def _initialize_model(self):
        # Add the directory containing the model file to sys.path
        model_dir = os.path.dirname(os.path.abspath(self.model_type))
        sys.path.append(model_dir)
        module_name, class_name = self.model_type.rsplit(".", 1)
        model_class = self.load_class(
            file_path=Path(model_dir, f"{module_name}.py"),
            class_name=class_name,
            module_name=module_name,
        )
        if self.model_params is not None:
            model = model_class(**self.model_params)
        else:
            model = model_class()
        if self.classifier:
            self._model = PytorchTemplateClassifier(
                model=model,
                criterion=self.criterion,
                optimizer=self.optimizer,
            )
        else:
            raise NotImplementedError("Only classifier models are currently supported.")
        self._model.to(self.device)
        self._model._set_random_seed(self.random_seed)
        
    
    def __post_init__(self):
        if self.clip_values is not None:
            self.clip_values = (float(self.clip_values[0]), float(self.clip_values[1]))
        self._initialize_model()
        self.training_prediction_time = None
        self.test_prediction_time = None
        self.training_time = None
        self.scoring_time = None
        self.training_predictions = None
        self.predictions = None
        self.prediction_n = None
        self.classes_ = None
        self.training_n = None
        self.training_score_time = None
        self.prediction_score_time = None
        self.score_time = None
        self.score_dict = None
    


    