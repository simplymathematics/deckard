# OS imports
import sys
import os
import importlib
import logging
from pathlib import Path
from tqdm import tqdm
import time

# Typing imports
from dataclasses import dataclass, field
from omegaconf import DictConfig
from typing import Union
import pandas as pd

# Torch imports
import torch.nn as nn
import torch

# Sklearn imports
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

# ART imports
from art.estimators.classification import PyTorchClassifier
from art.estimators.regression import PyTorchRegressor

from . import ModelConfig
from ..data import DataConfig
from ..utils import load_class
from dataclasses import dataclass, field
from omegaconf import DictConfig
from typing import Union
import pandas as pd

logger = logging.getLogger(__name__)


__all__ = ["PytorchModelConfig"]

torch_dict = {
    "classifier": PyTorchClassifier,
    "regressor": PyTorchRegressor,
}

def initialize_criterion(criterion):
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

def initialize_optimizer(optimizer, model):
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
    optimizer_class = optimizer_class(model.parameters(), **optimizer_params)
    assert isinstance(
        optimizer_class,
        torch.optim.Optimizer,
    ), f"Optimizer must be an instance of torch.optim.Optimizer, got {type(optimizer_class)}"
    return optimizer_class

class PytorchTemplateEsimator(BaseEstimator):
    def __init__(
        self,
        model: nn.Module,
        device=None,
        criterion: Union[dict, str] = "CrossEntropyLoss",
        optimizer: Union[dict, str] = "SGD",
        clip_values: Union[tuple, None] = None,
    ):
        self.model = model
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.criterion = initialize_criterion(criterion)
        self.optimizer = initialize_optimizer(optimizer, model)
        self.clip_values = clip_values
        self.loss_curve = []

    def _set_random_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    

    def to(self, device):
        self.device = device
        self.model.to(device)

    def fit(self, X, y, nb_epochs=1, batch_size=32, verbose=False, log_interval=1):
        if isinstance(X, torch.utils.data.DataLoader):
            return self.fit_loader(X, y, nb_epochs=nb_epochs, verbose=verbose, log_interval=log_interval)
        batch_index = 0
        # Move data and model to device
        self.model.to(self.device)
        # Training loop
        self.model.train()
        for epoch in range(nb_epochs):
            pbar = tqdm(
                zip(X.split(batch_size), y.split(batch_size)),
                total=len(X) // batch_size,
                desc=f"Epoch {epoch+1}/{nb_epochs}",
            )
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
                    logger.info(
                        f"Epoch {epoch+1}/{nb_epochs}, Batch {batch_index}, Loss: {loss.item()}",
                    )
            if log_interval > 0 and (epoch) % log_interval == 0:
                train_loss = loss.item()
                self.loss_curve.append(train_loss)
        # Return the classifier
        return self
    
    def fit_loader(self, X, y, nb_epochs=1, verbose=False, log_interval=1):
        # Store the classes seen during fit
        _ = y # Ignore y since it's just a copy of the dataloader
        classes_ = []
        batch_index = 0
        last_loss: Union[torch.Tensor, None] = None
        # Move data and model to device
        self.model.to(self.device)
        # Training loop
        self.model.train()
        epoch_pbar = tqdm(range(nb_epochs), desc="Training epochs")
        for epoch in epoch_pbar:
            batch_pbar = tqdm(
                X,
                total=len(X) if hasattr(X, "__len__") else None,
                desc=f"Epoch {epoch+1}/{nb_epochs}",
                leave=False,
            )
            for batch_X, batch_y in batch_pbar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device, dtype=torch.long)
                if len(batch_y.shape) < len(batch_X.shape):
                    batch_y = batch_y.float().unsqueeze(1)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                batch_index += 1
                batch_pbar.set_postfix(loss=loss.item())
                if verbose:
                    logger.info(
                        f"Epoch {epoch+1}/{nb_epochs}, Batch {batch_index}, Loss: {loss.item()}",
                    )
                unique_ys_in_batch = torch.unique(batch_y).detach().cpu().reshape(-1).tolist()
                classes_.extend(unique_ys_in_batch)
            if last_loss is not None and log_interval > 0 and (epoch) % log_interval == 0:
                train_loss = last_loss.item()
                self.loss_curve.append(train_loss)
                epoch_pbar.set_postfix(loss=train_loss)
        self.classes_ = list(torch.as_tensor(classes_).unique().tolist()) if classes_ else []
        # Return the classifier
        return self

    def predict_proba(self, X):
        # Check if fit has been called
        with torch.no_grad():
            probs = self.model.forward(X)
            return probs

    def predict(self, X):
        with torch.no_grad():
            probs = self.model(X)
            # predictions = torch.argmax(probs, dim=1)
            return probs
    
    def predict_loader(self, X): 
        preds = []
        with torch.no_grad():
            for batch_X, y in tqdm(X, desc="Predicting probabilities", total=len(X)):
                batch_X = batch_X.to(self.device)
                batch_preds =  self.model(batch_X)
                preds.extend(batch_preds)
        return torch.tensor(preds, dtype=y.dtype)
               

    def get_art_model(self, data):
        if isinstance(self, (PyTorchClassifier)):
            return self
        if self.clip_values is None or len(self.clip_values) == 0:
            clip_values = (0.0, 1.0)
        else:
            clip_values = self.clip_values
        art_class = PyTorchClassifier 
        if isinstance(data.X_train, torch.utils.data.DataLoader):
            x0, _ = next(iter(data.X_train))
            input_shape = x0[0].shape
        else:
            input_shape = data.X_train.shape[1:]
        init_params = {
            "model": self.model if not hasattr(self.model, "_model") else self.model._model,
            "loss": self.criterion,
            "optimizer": self.optimizer,
            "input_shape":  input_shape,
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

class PytorchTemplateRegressor(PytorchTemplateEsimator, RegressorMixin):
    def __init__(
        self,
        model: nn.Module,
        device=None,
        criterion: Union[dict, str] = "MSELoss",
        optimizer: Union[dict, str] = "SGD",
        clip_values: Union[tuple, None] = None,
    ):
        super().__init__(
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            clip_values=clip_values,
        )

class PytorchTemplateClassifier(PytorchTemplateEsimator, ClassifierMixin):
    def __init__(
        self,
        model: nn.Module,
        device=None,
        criterion: Union[dict, str] = "CrossEntropyLoss",
        optimizer: Union[dict, str] = "SGD",
        clip_values: Union[tuple, None] = None,
    ):
        super().__init__(
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            clip_values=clip_values,
        )
    

@dataclass
class PytorchModelConfig(ModelConfig):
    model_type: str = "torch_example.ResNet18"
    model_params: dict = field(default_factory=dict)
    classifier: bool = False
    fit_params: dict = field(default_factory=dict)
    library: str = "pytorch"
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        ),
    )
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
        
    def __post_init__(self):
        super().__post_init__()
        self._target_ = "deckard.model.pytorch.PytorchModelConfig"

    def get_art_model(self, data: DataConfig):
        loss = initialize_criterion(self.criterion)
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

    def fit(self, X, y, nb_epochs=1, batch_size=32, verbose=False, log_interval=1):
        if isinstance(X, torch.utils.data.DataLoader):
            return self.fit_loader(X, y, nb_epochs=nb_epochs, verbose=verbose, log_interval=log_interval)
        # Store the classes seen during fit
        self.classes_ = torch.unique(y)
        batch_index = 0
        # Move data and model to device
        self.model.to(self.device)
        # Training loop
        self.model.train()
        for epoch in range(nb_epochs):
            pbar = tqdm(
                zip(X.split(batch_size), y.split(batch_size)),
                total=len(X) // batch_size,
                desc=f"Epoch {epoch+1}/{nb_epochs}",
            )
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
                    logger.info(
                        f"Epoch {epoch+1}/{nb_epochs}, Batch {batch_index}, Loss: {loss.item()}",
                    )
            if log_interval > 0 and (epoch) % log_interval == 0:
                train_loss = loss.item()
                self.loss_curve.append(train_loss)
        # Return the classifier
        return self
    
    def fit_loader(self, X, y, nb_epochs=1, verbose=False, log_interval=1):
        # Store the classes seen during fit
        _ = y # Ignore y since it's just a copy of the dataloader
        classes_ = []
        batch_index = 0
        last_loss: Union[torch.Tensor, None] = None
        # Move data and model to device
        self.model.to(self.device)
        # Training loop
        self.model.train()
        epoch_pbar = tqdm(range(nb_epochs), desc="Training epochs")
        for epoch in epoch_pbar:
            batch_pbar = tqdm(
                X,
                total=len(X) if hasattr(X, "__len__") else None,
                desc=f"Epoch {epoch+1}/{nb_epochs}",
                leave=False,
            )
            for batch_X, batch_y in batch_pbar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device, dtype=torch.long)
                if len(batch_y.shape) < len(batch_X.shape):
                    batch_y = batch_y.float().unsqueeze(1)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                batch_index += 1
                batch_pbar.set_postfix(loss=loss.item())
                if verbose:
                    logger.info(
                        f"Epoch {epoch+1}/{nb_epochs}, Batch {batch_index}, Loss: {loss.item()}",
                    )
                unique_ys_in_batch = torch.unique(batch_y).detach().cpu().reshape(-1).tolist()
                classes_.extend(unique_ys_in_batch)
            if last_loss is not None and log_interval > 0 and (epoch) % log_interval == 0:
                train_loss = last_loss.item()
                self.loss_curve.append(train_loss)
                epoch_pbar.set_postfix(loss=train_loss)
        self.classes_ = list(torch.as_tensor(classes_).unique().tolist()) if classes_ else []
        # Return the classifier
        return self
    
    
    def _initialize_model(self):
        model = load_class(self.model_type, **self.model_params)
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
        if isinstance(self._model, (PyTorchClassifier, PyTorchRegressor)):
            base_model = self._model.model
            input_shape = data.X_train.shape[1:]
            
        else:
            base_model = self.get_model()
        if self.classifier:
            loss = initialize_criterion(self.criterion)
            input_shape = data.X_train.shape[1:]
            optimizer = initialize_optimizer(self.optimizer, base_model)
            nb_classes = len(torch.unique(data.y_train))
            if self.clip_values is None or len(self.clip_values) == 0:
                clip_values = (data.X_train.min().item(), data.X_train.max().item())
            else:
                clip_values = self.clip_values
            art_class = torch_dict["classifier"]
        else:
            input_shape = data.X_train.shape[1:]
            loss = initialize_criterion(self.criterion)
            optimizer = initialize_optimizer(self.optimizer, base_model)
            if self.clip_values is None or len(self.clip_values) == 0:
                clip_values = (data.X_train.min().item(), data.X_train.max().item())
            else:
                clip_values = self.clip_values
            art_class = torch_dict["regressor"]
        return art_class, {
            "loss": loss,
            "input_shape": input_shape,
            "nb_classes": nb_classes if self.classifier else None,
            "clip_values": clip_values,
            "device_type": "gpu" if "cuda" in str(self.device) else "cpu",
            "optimizer": optimizer,
        }

    def get_model(self):
        assert hasattr(
            self,
            "_model",
        ), "Model is not initialized. Call _initialize_model() first."
        assert hasattr(self._model, "model"), "Model does not have 'model' attribute."
        assert isinstance(
            self._model.model,
            nn.Module,
        ), "'model' attribute is not a torch.nn.Module."
        return self._model.model
    
        
    def __call__(self, data: DataConfig, **kwargs):
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
        model_params = b"".join(
            p.detach().cpu().numpy().tobytes() for p in self._model.model.parameters()
        )
        return hash((arch, params, model_params))

    def _score(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        if hasattr(self._model, "loss_curve") and len(self._model.loss_curve) > 0:
            loss_curve = self._model.loss_curve
        elif (
            hasattr(self._model, "model")
            and hasattr(self._model.model, "loss_curve")
            and len(self._model.model.loss_curve) > 0
        ):
            loss_curve = self._model.model.loss_curve
        else:
            loss_curve = None
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true)
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tesnor(y_pred)
        scores = super()._score(y_true, y_pred)
        if loss_curve is not None:
            scores["loss_curve"] = loss_curve
        if "train_loss_curve" in scores:
            del scores["train_loss_curve"]
        return scores

    def _load_or_train_model(self, data, model_file, times):
            """
            Loads a model from the specified filepath if it exists and is trained, or trains a new model using the provided data.
            If a model file exists at `model_file`, attempts to load and validate that the model is fitted.
            If the loaded model is not fitted, or if no model file exists, trains a new model using `data.X_train` and `data.y_train`.
            Updates the `times` dictionary with training time and number of training samples.
            Saves the trained model to `model_file` if provided and a new model was trained.
            Raises:
                ValueError: If neither a model nor a filepath is provided, or if the model is not trained after loading/training.
                NotFittedError: If the model is not initialized.
            Args:
                data: An object containing training data (`X_train`, `y_train`).
                model_file (str or Path or None): Path to the model file to load or save.
                times (dict): Dictionary to store training time and number of training samples.
            Returns:
                dict: Updated `times` dictionary with training metadata.
            """
            match self._model, model_file:
                case None, None:  # Neither model nor filepath provided
                    raise ValueError(
                        "Model not trained or loaded. Please train or load a model before prediction.",
                    )
                case _, _:  # Model and/or filepath provided
                    if model_file is not None and Path(model_file).exists():
                        logger.info(f"Model file {model_file} exists. Loading model.")
                        self = self.load(model_file)
                    if self.defense is not None:
                        self._model = self._apply_defense(data)
                    if not hasattr(self, "training_time"):
                        self._train(data.X_train, data.y_train)
                    times["training_time"] = self.training_time
                    times["training_n"] = self.training_n
                    if model_file is not None:
                        self.save(filepath=model_file)
            # Validate model is trained
            if self._model is None:
                raise NotFittedError("Model is not initialized")
            return times
    
    def _predict(self, X: Union[pd.DataFrame, torch.utils.data.DataLoader]) -> pd.Series:
        """
        Generates predictions for the input data using the initialized model.

        Args:
            X (pd.DataFrame): Input features for prediction.

        Returns:
            pd.Series: Predicted values.

        Raises:
            ValueError: If the model has not been initialized.

        """
        if isinstance(X, pd.DataFrame):
            return super()._predict(X)
        else:
            return self._model.predict_loader(X)
    
    def _evaluate_and_score(self, data: DataConfig, times: dict = None):
        """
        Evaluates the model by making predictions and scoring them on both training and test data.

        This method performs the following steps:
        1. Makes predictions on the training data if not already available, and records the prediction time.
        2. Scores the training predictions if true labels are available and scores have not already been computed.
        3. Makes predictions on the test data if not already available, and records the prediction time.
        4. Scores the test predictions if true labels are available and scores have not already been computed.
        5. Updates the internal score dictionary with timing and scoring information.

        Parameters
        ----------
        data : DataConfig
            The data configuration object containing training and test data (X_train, y_train, X_test, y_test).
        times : dict, optional
            A dictionary to store timing information for predictions and scoring.

        Raises
        ------
        ValueError
            If training predictions are not available when attempting to score them.

        Notes
        -----
        - Timing information for predictions and scoring is logged and stored in the `times` dictionary.
        - Score metrics are prefixed with 'train_' for training scores.
        - The method updates `self.score_dict` with all computed scores and timing information.
        """
        # Make predictions on training data if not already done
        if self.training_predictions is None or self.training_prediction_time is None:
            start_time = time.process_time()
            self.training_predictions = self._predict(data.X_train)
            end_time = time.process_time()
            self.training_prediction_time = end_time - start_time
            times["training_prediction_time"] = self.training_prediction_time
            logger.info(
                f"Training predictions made in {self.training_prediction_time:.2f} seconds",
            )
            times["training_n"] = len(self.training_predictions)
            times["training_prediction_time"] = self.training_prediction_time
        else:
            logger.info(
                "Training predictions already available, skipping prediction step.",
            )

        # Score training predictions if true labels are available and scoring not already done
        if self.training_score_time is None or self.score_dict is None:
            if self.training_predictions is not None:
                start = time.process_time()
                train_scores = self._score(data.y_train, self.training_predictions)
                self.training_score_time = time.process_time() - start
                # Prefix training scores with 'train_'
                train_scores = {
                    f"training_{key}": value for key, value in train_scores.items()
                }
                if "training_loss_curve" in train_scores:
                    del train_scores["training_loss_curve"]
                if self.score_dict is None:
                    self.score_dict = {}
                self.score_dict.update(train_scores)
                times["training_score_time"] = self.training_score_time
                logger.info(
                    f"Training scores computed in {self.training_score_time:.2f} seconds",
                )
            else:
                raise ValueError("Training predictions not available for scoring.")
        else:
            pass
        # Make predictions on test data if not already done
        if self.predictions is None or self.prediction_time is None:
            if data.X_test is not None:
                start_time = time.process_time()
                self.predictions = self._predict(data.X_test)
                end_time = time.process_time()
                self.prediction_time = end_time - start_time
                times["prediction_time"] = self.prediction_time
                logger.info(f"Predictions made in {self.prediction_time:.2f} seconds")
                times["prediction_n"] = len(self.predictions)
            else:
                raise ValueError("No test data available for prediction.")
        # Score test predictions if true labels are available and scoring not already done
        if (
            self.training_score_time is None
            or self.prediction_score_time is None
            or self.score_dict is None
        ):
            if data.y_test is not None and self.predictions is not None:
                test_scores = self._score(data.y_test, self.predictions)
                if self.score_dict is None:
                    self.score_dict = {}
                self.score_dict = {**self.score_dict, **test_scores}
                self.prediction_score_time = time.process_time() - start
                times["prediction_score_time"] = self.prediction_score_time
                logger.info(
                    f"Prediction scores computed in {self.prediction_score_time:.2f} seconds",
                )
            else:
                raise ValueError("No test labels available for scoring.")
        else:
            pass
        self.score_dict.update(times)

    def _train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the internal model using the provided feature matrix and target vector.

        Args
        -------
            X (pd.DataFrame): Feature matrix for training.
            y (pd.Series): Target vector for training.

        Raises
        -------
            ValueError: If the internal model is not initialized.

        Side Effects
        -------
            - Fits the internal model to the data.
            - Records the training time in seconds.
            - Logs the training duration.
        """
        if self._model is None:
            raise ValueError("Model not initialized")
        start_time = time.process_time()
        assert hasattr(self._model, "fit"), "Model does not have a fit method"
        # Make sure that model, X, and y are on the same device and model/X have the same dtype
        model_dtype = next(self._model.model.parameters()).dtype
        model_device = next(self._model.model.parameters()).device
        if isinstance(X, torch.Tensor):
            X = X.to(device=model_device, dtype=model_dtype)
            if isinstance(y, torch.Tensor):
                y = y.to(device=model_device, dtype=torch.float32)
        fit_params = {} if not hasattr(self, "fit_params") else self.fit_params
        if isinstance(X, torch.utils.data.DataLoader):
            self._model.fit_loader(X, y, **fit_params)
        else:
            self._model.fit(X,y, **fit_params)
        end_time = time.process_time()
        self.training_time = end_time - start_time
        self._training_n = len(y)
        logger.info(f"Model trained in {self.training_time:.2f} seconds")
    
    def _classification_scores(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        start = time.process_time()
        scores = {}

        # Cast to tensors if needed
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true)
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred)

        # Get predicted class labels if y_pred contains probabilities/logits
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred_labels = torch.argmax(y_pred, dim=1)
        else:
            y_pred_labels = y_pred.reshape(-1)

        y_true_labels = y_true.reshape(-1).long()

        # Calculate Accuracy
        correct = (y_pred_labels == y_true_labels).sum().item()
        scores["accuracy"] = correct / len(y_true_labels)

        # Calculate F1, Precision, Recall per class then average (macro)
        classes = torch.unique(y_true_labels)
        precisions, recalls, f1s = [], [], []
        for cls in classes:
            tp = ((y_pred_labels == cls) & (y_true_labels == cls)).sum().item()
            fp = ((y_pred_labels == cls) & (y_true_labels != cls)).sum().item()
            fn = ((y_pred_labels != cls) & (y_true_labels == cls)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        scores["precision"] = sum(precisions) / len(precisions) if precisions else 0.0
        scores["recall"] = sum(recalls) / len(recalls) if recalls else 0.0
        scores["f1"] = sum(f1s) / len(f1s) if f1s else 0.0

        end = time.process_time()
        scores["scoring_time"] = end - start
        scores["score_n"] = len(y_pred)
        return scores
        
    
    def _regression_scores(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        # Cast y_true, y_pred to numpy arrays if they are tensors
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        return super()._regression_scores(y_true, y_pred)

def input_shape_from_data_config(data: DataConfig):
    # Assuming data.X_train is a torch.Tensor
    return data.X_train.shape[1:]
