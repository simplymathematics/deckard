import logging, os, json, pickle
from time import process_time
from urllib.parse import urlparse as is_url
from copy import deepcopy
from hashlib import md5 as my_hash
from pathlib import Path
import numpy as np

from art.estimators.classification import (
    PyTorchClassifier,
    TensorFlowClassifier,
    KerasClassifier,
    TensorFlowV2Classifier,
)
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from art.estimators import ScikitlearnEstimator
from art.estimators.regression import ScikitlearnRegressor
from art.utils import get_file

from .data import Data
from typing import Callable, Union

logger = logging.getLogger(__name__)

supported_estimators = [
    PyTorchClassifier,
    TensorFlowClassifier,
    KerasClassifier,
    ScikitlearnClassifier,
    ScikitlearnRegressor,
    ScikitlearnEstimator,
    TensorFlowV2Classifier,
]

from deckard.base.hashable import BaseHashable


class Model(BaseHashable):
    """Creates a model object that includes a dicitonary of passed parameters."""

    def __init__(
        self,
        model: Union[str, object],
        model_type: str = "sklearn",
        defence: object = None,
        path=".",
        is_fitted: bool = False,
        classifier=True,
        art: bool = True,
        url=None,
    ):
        """
        Initialize the model object.
        estimator: the estimator to use
        model_type: the type of the model. Use 'tf1' for TensorFlow 1.x models.
        verbose: the verbosity level
        """
        self.model_type = model_type
        self.path = path
        self.is_fitted = is_fitted
        self.url = url
        self.classifier = classifier
        self.model = model
        self.art = art
        self.defence = defence if defence is not None else None
        self.params = dict(vars(self))
        if defence is not None:
            self.params["defence"] = {}
            self.params["defence"]["name"] = (
                str(type(self.defence)).split(".")[-1].split("'")[0]
            )
            self.params["defence"]["params"] = dict(vars(self.defence))
            # print(self.params)
            # input("Inside defense init. Press Enter to continue...")
            for key in vars(defence):
                if key != "estimator" or "classifier" or "regressor":
                    self.params["defence"]["params"][key] = getattr(defence, key)
            if "preprocessor" in str(type(self.defence)):
                self.params["defence"]["params"].update({"type": "preprocessor"})
            elif "postprocessor" in str(type(self.defence)):
                self.params["defence"]["params"].update({"type": "postprocessor"})
            elif "trainer" in str(type(self.defence)):
                self.params["defence"]["params"].update({"type": "trainer"})
            elif "transformer" in str(type(self.defence)):
                self.params["defence"]["params"].update({"type": "transformer"})
        else:
            self.params["defence"] = {}

    def set_params(self, params: dict = None):
        """
        :param params: A dictionary of parameters to set.
        Sets the extra parameters of the Model object.
        """
        assert params is not None, "Params must be specified"
        assert isinstance(params, dict), "Params must be a dictionary"
        for param, value in params.items():
            if hasattr(self, param):
                # Attempts to set attribute on self first
                setattr(self, param, value)
                # Tries a generic set_params method
            elif hasattr(self.model, "set_params"):
                self.model.set_params(**{param: value})
                # Tries to set the art attribute if it exists
            elif (
                hasattr(self.model, "model")
                and hasattr(self.model.model, "set_params")
                and param in self.model.model.get_params()
            ):
                self.model.model.set_params(**{param: value})
            else:
                raise ValueError(
                    "Parameter {} not found in \n {} or \n {}".format(
                        param, self.model.model.get_params(), self.model.__dict__.keys()
                    )
                )
            self.params.update({param: value})
            # self.params.update({param : value})

    def load_from_string(self, filename: str) -> None:
        """
        Load a model from a pickle file.
        filename: the pickle file to load the model from
        model_type: the type of model to load
        """
        logger.debug("Loading model")
        # load the model
        model_type = self.model_type
        url = self.url
        path = self.path
        assert model_type is not None, "model_type must be specified"
        assert filename is not None, "filename must be specified"
        assert path is not None or url is not None, "path or url must be specified"
        output_dir = os.path.dirname(path)
        if url is not None:
            # download model
            model_path = get_file(
                filename=filename, extract=False, path=path, url=url, verbose=True
            )
            logging.info("Downloaded model from {} to {}".format(url, model_path))
        filename = os.path.join(path, filename)
        if model_type == "keras" or filename.endswith(".h5"):
            from tensorflow.keras.models import load_model as keras_load_model

            model = keras_load_model(filename)
        elif model_type == "torch" or model_type == "pytorch":
            from torch import load as torch_load

            model = torch_load(filename)
        elif model_type == "tf" or model_type == "tensorflow":
            from tensorflow.keras.models import load_model as tf_load_model

            model = tf_load_model(filename)
        elif (
            model_type == "tfv1" or model_type == "tensorflowv1" or model_type == "tf1"
        ):
            import tensorflow.compat.v1 as tfv1

            tfv1.disable_eager_execution()
            from tensorflow.keras.models import load_model as tf_load_model

            model = tf_load_model(filename)
        elif (
            model_type == "sklearn"
            or model_type == "pipeline"
            or model_type == "gridsearch"
            or model_type == "pickle"
        ):
            if path:
                filename = os.path.join(path, filename)
            with open(filename, "rb") as f:
                model = pickle.load(f)
        else:
            raise ValueError("Model type {} not supported".format(model_type))
        logger.info("Loaded model")
        return model

    def __call__(self, art: bool = None) -> None:
        """
        Load a model from a pickle file.
        filename: the pickle file to load the model from
        """
        if art is None:
            art = self.art
        logger.debug("Loading model")
        if isinstance(self.model, (str, Path)):
            # load the model
            self.model = self.load_from_string(self.model)
            self.params["model"] = self.model
        else:
            logger.info("Model already in memory.")
            pass
        logger.info("Loaded model")
        if self.classifier == True and art == True:
            self.model = self.initialize_art_classifier()
        elif self.classifier == False and art == True:
            self.model = self.initialize_art_regressor()
        return self

    def initialize_art_classifier(
        self, clip_values: tuple = (0, 255), **kwargs
    ) -> None:
        """
        Initialize the classifier.
        """
        preprocessors = []
        postprocessors = []
        trainers = []
        transformers = []
        # Find defence type
        if hasattr(self, "defence"):
            self.params["defence"].update(
                {"type": str(type(self.defence)).split(".")[-1].split("'")[0]}
            )
            if "art" and "preprocessor" in str(type(self.defence)):
                preprocessors.append(self.defence)
            elif "art" and "postprocessor" in str(type(self.defence)):
                postprocessors.append(self.defence)
            elif "art" and "trainer" in str(type(self.defence)):
                trainers.append(self.defence)
            elif "art" and "transformer" in str(type(self.defence)):
                transformers.append(self.defence)
            elif self.defence is not None:
                raise ValueError(
                    "defence type {} not supported".format(
                        self.params["defence"]["type"]
                    )
                )
        else:
            pass
        # Iinitialize model by type
        if isinstance(
            self.model,
            (
                PyTorchClassifier,
                TensorFlowV2Classifier,
                ScikitlearnClassifier,
                TensorFlowClassifier,
                KerasClassifier,
            ),
        ):
            self.model = self.model.model
        if self.model_type in ["pytorch", "torch", "pyt"]:
            model = PyTorchClassifier(
                self.model,
                preprocessing_defences=preprocessors,
                postprocessing_defences=postprocessors,
                clip_values=clip_values,
                **kwargs
            )
        elif self.model_type in ["keras"]:
            model = KerasClassifier(
                self.model,
                preprocessing_defences=preprocessors,
                postprocessing_defences=postprocessors,
                clip_values=clip_values,
                **kwargs
            )
        elif self.model_type in ["tensorflow", "tf", "tf1", "tfv1", "tensorflowv1"]:
            model = KerasClassifier(
                self.model,
                preprocessing_defences=preprocessors,
                postprocessing_defences=postprocessors,
                clip_values=clip_values,
                **kwargs
            )
        elif self.model_type in ["tf2", "tensorflow2", "tfv2", "tensorflowv2"]:
            model = KerasClassifier(
                self.model,
                preprocessing_defences=preprocessors,
                postprocessing_defences=postprocessors,
                clip_values=clip_values,
                **kwargs
            )
        elif self.model_type in [
            "sklearn",
            "pipeline",
            "gridsearch",
            "pickle",
            "scikit",
            "scikit-learn",
        ]:
            model = ScikitlearnClassifier(
                self.model,
                preprocessing_defences=preprocessors,
                postprocessing_defences=postprocessors,
                clip_values=clip_values,
                **kwargs
            )
        return model

    def initialize_art_regressor(self, **kwargs) -> None:
        preprocessors = []
        postprocessors = []
        trainers = []
        transformers = []
        # Find defence type
        if hasattr(self, "defence"):

            if "preprocessor" in str(type(self.defence)):
                preprocessors.append(self.defence)
            elif "postprocessor" in str(type(self.defence)):
                postprocessors.append(self.defence)
            elif "trainer" in str(type(self.defence)):
                trainers.append(self.defence)
            elif "transformer" in str(type(self.defence)):
                transformers.append(self.defence)
            elif self.defence is not None:
                raise ValueError(
                    "defence type {} not supported".format(
                        self.params["defence"]["type"]
                    )
                )
        else:
            pass
        # Initialize model by type
        if self.model_type in ["sklearn", "scikit-learn", "scikit"]:
            model = ScikitlearnRegressor(
                model=self.model,
                preprocessing_defences=preprocessors,
                postprocessing_defences=postprocessors,
                **kwargs
            )
        else:
            raise ValueError("Model type {} not supported".format(self.model_type))
        return model

    def save_model(self, filename: str = None, path: str = None):
        """
        Saves the experiment to a pickle file or directory, depending on model type.
        """
        assert path or self.path is not None, "No path specified"
        if path:
            self.path = path
        else:
            path = self.path
        if not os.path.isdir(path):
            os.mkdir(path)
        if filename:
            self.filename = filename
        else:
            filename = self.filename
        if hasattr(self, "model") and hasattr(self.model, "save"):
            if filename.endswith(".pickle"):
                filename = filename[:-7]
            try:
                self.model.save(filename=filename, path=path)
            except:
                ART_DATA_PATH = path
                self.model.save(filename=filename)
        else:
            with open(os.path.join(path, filename), "wb") as f:
                pickle.dump(self.model, f)
        return os.path.join(path, filename)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray = None) -> None:
        """
        Fits model.
        """
        if self.is_fitted:
            logger.warning("Model is already fitted")
            self.time_dict = {"fit_time": np.nan}
        else:
            start = process_time()
            try:
                self.model.fit(X_train, y_train)
            except ValueError as e:
                if "y should be a 1d array" in str(e):
                    y_train = np.argmax(y_train, axis=1)
                    self.model.fit(X_train, y_train)
                else:
                    raise e
            end = process_time()
            self.time_dict = {"fit_time": end - start}
            self.is_fitted = True
        return None

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        start = process_time()
        predictions = self.model.predict(X_test)
        end = process_time()
        if not hasattr(self, "time_dict"):
            self.time_dict = {}
            self.time_dict["fit_time"] = None
        self.time_dict["pred_time"] = end - start
        return predictions

    # TODO:
    # def transform(self, X_train, y_train = None):
