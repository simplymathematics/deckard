import json
import logging
import os
import pickle
from copy import deepcopy
from pathlib import Path
from time import process_time
from typing import Callable, Union

import numpy as np
from art.estimators import ScikitlearnEstimator
from art.estimators.classification import (KerasClassifier, PyTorchClassifier,
                                           TensorFlowClassifier,
                                           TensorFlowV2Classifier)
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from art.estimators.regression import ScikitlearnRegressor
from art.utils import get_file

from .data import Data
from .hashable import my_hash
from .parse import generate_object_from_tuple, generate_tuple_from_yml

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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

__all__ = ["Model"]

class Model(BaseHashable):
    """Creates a model object that includes a dicitonary of passed parameters."""

    def __init__(
        self,
        model: Union[str, object],
        model_type: str = "sklearn",
        defence:dict = None,
        pipeline:dict = None,
        path=".",
        is_fitted: bool = False,
        classifier=True,
        art: bool = True,
        url=None,
        fit_params: dict = None,
        predict_params: dict = None,
        clip_values: tuple = None,
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
        self.model = model
        self.clip_values = clip_values
        
        self.classifier = classifier
        self.art = art if art else False
        self.defence = defence if defence is not None else None
        self.pipeline = pipeline if pipeline is not None else None
        self.fit_params = fit_params if fit_params is not None else {}
        self.predict_params = predict_params if predict_params is not None else {}
        _ = dict(vars(self))
        self.params = _
        if defence is not None:
            self.art = True
            self.insert_art_defence(defence)
        if pipeline is not None:
           self.insert_sklearn_preprocessor(pipeline)

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

    def __call__(self, pipeline:dict = None, art:dict = None, fit = None, predict = None) -> None:
        """
        Load a model from a pickle file.
        filename: the pickle file to load the model from
        """
        if art is not None:
            if isinstance(art, dict):
                art_kwargs = art
                self.art = True
            elif isinstance(art, bool):
                art_kwargs = {}
        logger.debug("Loading model")
        if isinstance(self.model, (str, Path)):
            # load the model
            self.model = self.load_from_string(self.model)
            self.params["model"] = self.model
        else:
            logger.info("Model already in memory.")
            pass
        logger.info("Loaded model")
        if pipeline is not None:
            self.insert_sklearn_preprocessor(**pipeline)
        if self.classifier == True and art == True:
            self.model = self.initialize_art_classifier(**art_kwargs)
        elif self.classifier == False and art == True:
            self.model = self.initialize_art_regressor(**art_kwargs)
        if fit is not None:
            setattr(self, "fit_params", fit)
        if predict is not None:
            setattr(self, "predict_params", predict)
        return self
        
    def initialize_art_classifier(
        self, clip_values: tuple = None, **kwargs
    ) -> None:
        """
        Initialize the classifier.
        """
        preprocessors = []
        postprocessors = []
        trainers = []
        transformers = []
        # Find defence types
        if clip_values is None:
            clip_values = self.clip_values
        if hasattr(self, "Defence"):
            self.params["Defence"].update(
                {"type": str(type(self.defence)).split(".")[-1].split("'")[0]}
            )
            if "art" and "preprocessor" in self.params['Defence']['type'].lower():
                preprocessors.append(self.defence)
            elif "art" and "postprocessor" in self.params['Defence']['type'].lower():
                postprocessors.append(self.defence)
            elif "art" and "trainer" in self.params['Defence']['type'].lower():
                trainers.append(self.defence)
            elif "art" and "transformer" in self.params['Defence']['type'].lower():
                transformers.append(self.defence)
            elif self.defence is not None:
                raise ValueError(
                    "defence type {} not supported".format(
                        self.params["Defence"]["type"]
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
    
    def insert_sklearn_preprocessor(
        self, name: str, preprocessor:Union[str, Path, dict], position: int
    ):
        """
        Add a sklearn preprocessor to the experiment.
        :param name: name of the preprocessor
        :param preprocessor: preprocessor to add
        :param position: position to add preprocessor

        """
        if not isinstance(preprocessor, (str, Path, dict)):
            raise ValueError("Preprocessor must be a string or a dictionary, not type {}".format(type(preprocessor)))
        config_tuple = generate_tuple_from_yml(preprocessor)
        if 'Pipeline' not in self.params:
            self.params['Pipeline'] = {}
        id_ = my_hash(config_tuple) if isinstance(preprocessor, dict) else Path(preprocessor).name.split('.')[0]
        preprocessor = generate_object_from_tuple(config_tuple)
        # TODO: Fix pipeline params
        # self.params['Pipeline'][name] = {'id':id_, "type": str(type(preprocessor)).split(".")[-1].split("'")[0], "params" : config_tuple[1], "position" : position}      
        # If it's already a pipeline
        if isinstance(self.model, Pipeline):
            pipe = self.model
        elif hasattr(self.model, "model") and isinstance(
            self.model.model, Pipeline
        ):
            pipe = self.model.model
        elif "art.estimators" in str(type(self.model)) and not isinstance(
            self.model.model, Pipeline
        ):
            logger.warning("Model is already an ART classifier. If Defence is not None, it will be ignored. ART defences are meant to be applied to the final sklearn model.")
            pipe = Pipeline([("model", self.model.model)])
        elif isinstance(self.model, BaseEstimator) and not isinstance(
            self.model, Pipeline
        ):
            pipe = Pipeline([("model", self.model)])
        else:
            raise ValueError(
                "Cannot make model type {} into a pipeline".format(
                    type(self.model)
                )
            )
        new_model = deepcopy(pipe)
        assert isinstance(new_model, Pipeline)
        new_model.steps.insert(position, (name, preprocessor))
        self.model = new_model
    
    def insert_art_defence(self, defence:Union[str, Path, dict]):
        """
        Add a defence to the experiment.
        :param defence: defence to add
        """
        if not isinstance(defence, (str, Path, dict)):
            raise ValueError("Defence must be a string, Path or dict")
        defence_tuple = generate_tuple_from_yml(defence)
        if 'Defence' not in self.params:
            self.params['Defence'] = {}
        id_ = my_hash(defence_tuple) if isinstance(defence, dict) else Path(defence).name.split('.')[0]
        self.params['Defence'] = {}
        self.params['Defence']['name'] = defence_tuple[0].split('.')[-1]
        self.params['Defence']['params'] = defence_tuple[1]
        self.params['Defence']['id'] = id_
        try:
            self.defence = generate_object_from_tuple(defence_tuple)
        except TypeError as e:
            if "clip_values" in str(e):
                self.defence = generate_object_from_tuple(defence_tuple, self.clip_values)
            elif "estimator" in str(e):
                self.defence = generate_object_from_tuple(defence_tuple, self.model)
            elif "classifier" in str(e):
                self.defence = generate_object_from_tuple(defence_tuple, self.model)
            else:
                raise e
        self.params['Defence']['type'] = str(type(defence))
        if "preprocessor" in str(type(self.defence)):
            self.params["Defence"].update({"type": "preprocessor"})
        elif "postprocessor" in str(type(self.defence)):
            self.params["Defence"].update({"type": "postprocessor"})
        elif "trainer" in str(type(self.defence)):
            self.params["Defence"].update({"type": "trainer"})
        elif "transformer" in str(type(self.defence)):
            self.params["Defence"].update({"type": "transformer"})
        else:
            raise NotImplementedError(
                "Defence type {} not supported".format(type(self.defence))
            )


    def initialize_art_regressor(self, **kwargs) -> None:
        preprocessors = []
        postprocessors = []
        trainers = []
        transformers = []
        # Find defence type
        if hasattr(self, "Defence"):

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
                        self.params["Defence"]["type"]
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
            flag = False
            if filename.endswith(".pickle"):
                filename = filename[:-7]
                flag = True
            try:
                self.model.save(filename, path)
            except:
                os.mkdir(os.path.join(path, filename))
                fullpath = os.path.join(path, filename)
                self.model.save(fullpath)
            if flag == True:
                filename = filename + ".pickle"
        else:
            with open(os.path.join(path, filename), "wb") as f:
                pickle.dump(self.model, f)
        print("Saved model to {}".format(os.path.join(path, filename)))
        return filename

    def fit(self, X_train: np.ndarray, y_train: np.ndarray = None, **kwargs) -> None:
        """
        Fits model.
        """
        if hasattr(self, 'fit_params'):
            opts = self.fit_params
        else:
            opts = {}
        if self.is_fitted:
            logger.warning("Model is already fitted")
            self.time_dict = {"fit_time": np.nan}
        else:
            start = process_time()
            try:
                self.model.fit(X_train, y_train, **kwargs, **opts)
            except ValueError as e:
                if "y should be a 1d array" in str(e):
                    y_train = np.argmax(y_train, axis=1)
                    self.model.fit(X_train, y_train, **kwargs, **opts)
                else:
                    raise e
            end = process_time()
            self.time_dict = {"fit_time": end - start}
            self.is_fitted = True
        return None

    def predict(self, X_test: np.ndarray, pred_type:str = None) -> np.ndarray:
        if hasattr(self, 'predict_params'):
            opts = self.predict_params
        else:
            opts = {}
        if pred_type == 'proba':
            start = process_time()
            predictions = self.model.predict_proba(X_test, **opts)
        elif pred_type == 'log':
            start = process_time()
            predictions = self.model.predict_log_proba(X_test, **opts)
        elif pred_type is None  or pred_type =='decision':
            start = process_time()
            predictions = self.model.predict(X_test, **opts)
        else:
            raise ValueError("pred_type {} not supported".format(pred_type))
        end = process_time()
        if not hasattr(self, "time_dict"):
            self.time_dict = {}
            self.time_dict["fit_time"] = None
        self.time_dict["pred_time"] = end - start
        return predictions

    # TODO:
    # def transform(self, X_train, y_train = None):
