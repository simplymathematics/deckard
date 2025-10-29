
# OS imports
import sys
import os
import importlib
import logging
from pathlib import Path
from tqdm import tqdm
# Typing imports
from dataclasses import dataclass, field
from omegaconf import DictConfig
from typing import Union
# Torch imports
import torch.nn as nn
import torch
# Sklearn imports
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
# ART imports
from art.estimators.classification import PyTorchClassifier
from art.estimators.regression import PyTorchRegressor

from .  import ModelConfig
from ..data import DataConfig
logger = logging.getLogger(__name__)


__all__ = ["PytorchModelConfig"]




class PytorchTemplateClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
            self, 
            model: nn.Module, 
            device=None,
            criterion:Union[dict, str]="CrossEntropyLoss",
            optimizer:Union[dict, str]="SGD",
            clip_values:Union[tuple, None]=None,
        ):
        self.model = model
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.criterion = self._initialize_criterion(criterion)
        self.optimizer = self._initialize_optimizer(optimizer)
        self.clip_values = clip_values
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
        assert callable(criterion), f"Criterion must be callable, got {type(criterion)}"
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
        optimizer_class = optimizer_class(self.model.parameters(), **optimizer_params)
        assert isinstance(optimizer_class, torch.optim.Optimizer), f"Optimizer must be an instance of torch.optim.Optimizer, got {type(optimizer_class)}"
        return optimizer_class
    
    def to(self, device):
        self.device = device
        self.model.to(device)
    
    def fit(self, X, y, nb_epochs=1, batch_size=32, verbose=False, log_interval=1):
        # Store the classes seen during fit
        self.classes_ = torch.unique(y)
        batch_index = 0
        # Move data and model to device
        self.model.to(self.device)
        # Training loop
        self.model.train()
        for epoch in range(nb_epochs):
            pbar = tqdm(zip(X.split(batch_size), y.split(batch_size)), total=len(X) // batch_size, desc=f"Epoch {epoch+1}/{nb_epochs}")
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
                    logger.info(f"Epoch {epoch+1}/{nb_epochs}, Batch {batch_index}, Loss: {loss.item()}")
            if log_interval > 0 and (epoch) % log_interval == 0:
                train_loss = loss.item()
                self.loss_curve.append(train_loss)
        # Return the classifier
        return self
    
    
    
    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Move data to device
        with torch.no_grad():
            probs = self.model.forward(X)
            return probs
    
    
    def predict(self, X):
        with torch.no_grad():
            model_dtype = next(self.model.parameters()).dtype
            X_dtype = X.dtype
            if X_dtype != model_dtype:
                model_device = next(self.model.parameters()).device
                X = X.to(device=model_device, dtype=model_dtype)
            probs = self.model(X)
            # predictions = torch.argmax(probs, dim=1)
            return probs

    def get_art_model(self, data):
        if isinstance(self, (PyTorchClassifier, PyTorchRegressor)):
            return self
        if self.clip_values is None or len(self.clip_values) == 0:
            clip_values = (0.0, 1.0)
        else:
            clip_values = self.clip_values
        art_class = PyTorchClassifier
        init_params = {
            "model": self.model,
            "loss": self.criterion,
            "optimizer": self.optimizer,
            "input_shape": data.X_train.shape[1:],
            "nb_classes": len(torch.unique(data.y_train)),
            "clip_values": clip_values,
            "device_type": "gpu" if "cuda" in str(self.device) else "cpu",
        }
        return art_class(**init_params)
    
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
    channels_first: bool = True
    defense_name: Union[str, None] = None
    defense_params: dict = field(default_factory=dict)
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
    random_seed : int
        Random seed for reproducibility.
    channels_first : bool
        Whether the input data has channels first format.
    defense_name : str or None
        Name of the defense method to apply.
    defense_params : dict
        Parameters for the defense method.
    

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
    
    
    def get_art_model(self, data:DataConfig ):
        loss = self._model._initialize_criterion(self.criterion)
        assert isinstance(loss, nn.Module), "Loss must be a torch.nn.Module."
        input_shape = data.X_train.shape[1:]
        if self.classifier:
            
            nb_classes = len(torch.unique(data.y_train))
            if self.clip_values is None or len(self.clip_values) == 0:
                clip_values = (data.X_train.min().item(), data.X_train.max().item())
            else:
                clip_values = self.clip_values
            art_class = PyTorchClassifier
            init_params = {
                "loss": loss,
                "input_shape": input_shape,
                "nb_classes": nb_classes,
                "clip_values": clip_values,
                "device_type": "gpu" if "cuda" in str(self.device) else "cpu",
                "channels_first": self.channels_first,
            }
        else:
            if self.clip_values is None or len(self.clip_values) == 0:
                clip_values = (data.X_train.min().item(), data.X_train.max().item())
            else:
                clip_values = self.clip_values
            art_class = PyTorchRegressor
            init_params = {
                "loss": loss,
                "input_shape": input_shape,
                "clip_values": clip_values,
                "device_type": "gpu" if "cuda" in str(self.device) else "cpu",
                "channels_first": self.channels_first,
            }
        return art_class, init_params

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
        
    def get_art_class(self, data):
        if self.classifier:
            loss = self._model._initialize_criterion(self.criterion)
            input_shape = data.X_train.shape[1:]
            optimizer = self._model._initialize_optimizer(self.optimizer)
            nb_classes = len(torch.unique(data.y_train))
            if self.clip_values is None or len(self.clip_values) == 0:
                clip_values = (data.X_train.min().item(), data.X_train.max().item())
            else:
                clip_values = self.clip_values
            art_class = PyTorchClassifier
        else:
            input_shape = data.X_train.shape[1:]
            loss = self._model._initialize_criterion(self.criterion)
            optimizer = self._model._initialize_optimizer(self.optimizer)
            if self.clip_values is None or len(self.clip_values) == 0:
                clip_values = (data.X_train.min().item(), data.X_train.max().item())
            else:
                clip_values = self.clip_values
            art_class = PyTorchRegressor
        return art_class, {
            "loss": loss,
            "input_shape": input_shape,
            "nb_classes": nb_classes if self.classifier else None,
            "clip_values": clip_values,
            "device_type": "gpu" if "cuda" in str(self.device) else "cpu",
            "optimizer": optimizer,
        }
        
    def get_model(self):
        assert hasattr(self, "_model"), "Model is not initialized. Call _initialize_model() first."
        assert hasattr(self._model, "model"), "Model does not have 'model' attribute."
        assert isinstance(self._model.model, nn.Module), "'model' attribute is not a torch.nn.Module."
        return self._model.model
    
    def __call__(self, data:DataConfig, **kwargs):
        self._initialize_model()
        return super().__call__(data, **kwargs)
    
    def __post_init__(self):
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
    
    def __hash__(self):
        self._model.model.hash = super().__hash__
        arch = str(self._model.model)
        params = f"model_type={self.model_type}, model_params={self.model_params}, classifier={self.classifier}"
        model_params = b"".join(p.detach().cpu().numpy().tobytes() for p in self._model.model.parameters())
        return hash((arch, params, model_params))
    
    
    def _score(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        if hasattr(self._model, "loss_curve") and len(self._model.loss_curve) > 0:
            loss_curve = self._model.loss_curve
        elif hasattr(self._model, "model") and hasattr(self._model.model, "loss_curve") and len(self._model.model.loss_curve) > 0:
            loss_curve  = self._model.model.loss_curve
        else:
            loss_curve = None
        scores = super()._score(y_true, y_pred)
        if loss_curve is not None:
            scores["loss_curve"] = loss_curve
        if "train_loss_curve" in scores:
            del scores["train_loss_curve"]
        return scores
    
def input_shape_from_data_config(data:DataConfig):
    # Assuming data.X_train is a torch.Tensor
    return data.X_train.shape[1:]
    
    
        
        


    