from typing import Union
import importlib
import logging
import torch.nn as nn
from dataclasses import dataclass, field
import torch
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels

from . import ModelConfig
logger = logging.getLogger(__name__)

supported_sklearn_libraries = ["sklearn"]

__all__ = ["PytorchModelConfig"]




class TemplateClassifier(ClassifierMixin, BaseEstimator):
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
    
    def to(self, device):
        self.device = device
        self.model.to(device)
    
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
        check_is_fitted(self)
        self.model.eval()
        X = X.to(self.device)
        self.predict_proba(X)
        with torch.no_grad():
            return np.argmax(self.predict_proba(X), axis=1)

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

    def _initialize_model(self):
        super()._initialize_model()
        assert hasattr(self, "_model"), "Model initialization failed."
        if self.classifier:
            self._model = TemplateClassifier(
                model=self._model,
                criterion=self.criterion,
                optimizer=self.optimizer,
            )
        else:
            raise NotImplementedError("Only classifier models are currently supported.")
        self._model.to(self.device)
        
    
    def __post_init__(self):
        super().__post_init__()
        if self.clip_values is not None:
            self.clip_values = (float(self.clip_values[0]), float(self.clip_values[1]))
    
    


    