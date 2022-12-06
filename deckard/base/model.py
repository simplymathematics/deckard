import collections
import logging
import pickle
import warnings
from copy import deepcopy
from pathlib import Path
from art.estimators import ScikitlearnEstimator
from art.estimators.classification import (
    KerasClassifier,
    PyTorchClassifier,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
)
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from art.estimators.regression import ScikitlearnRegressor
from art.utils import get_file
from sklearn.base import BaseEstimator, is_regressor
from sklearn.pipeline import Pipeline
from validators import url as is_url

from .hashable import BaseHashable
from .utils import factory, load_from_tup

logger = logging.getLogger(__name__)
supported_estimators = (
    PyTorchClassifier,
    TensorFlowClassifier,
    KerasClassifier,
    ScikitlearnClassifier,
    ScikitlearnRegressor,
    ScikitlearnEstimator,
    TensorFlowV2Classifier,
    Pipeline,
    BaseEstimator,
)


warnings.filterwarnings("ignore", category=FutureWarning)

filetypes = {
    "pkl": "sklearn",
    "h5": "keras",
    "pt": "torch",
    "pth": "torch",
    "pb": "tensorflow",
    "pbtxt": "tensorflow",
    "tflite": "tflite",
    "pickle": "sklearn",
}


class Model(
    collections.namedtuple(
        typename="model",
        field_names="init, files, fit, predict, sklearn_pipeline, art_pipeline, url",
        defaults=({}, {}, {}, [], [], ""),
    ),
    BaseHashable,
):
    def __new__(cls, loader, node):
        return super().__new__(cls, **loader.construct_mapping(node))

    def load(self, filename, art=False):
        library = filetypes[Path(filename).suffix.split(".")[-1]]
        params = deepcopy(self.init)
        if is_url(self.url):
            filename = Path(self.init.pop("name"))
            library = self.init.pop("library")
            name = filename.name
            path = filename.parent
            model = get_file(name, self.url, path)
            if library == "keras":
                from keras.models import load_model
                model = load_model(model)
            elif library == "torch":
                from torch import load
                model = load(model)
            elif library == "tensorflow":
                from tensorflow import keras
                model = keras.models.load_model(model)
            elif library == "tfv1":
                model = keras.models.load_model(model)
            elif library == "sklearn":
                with open(model, "rb") as f:
                    model = pickle.load(f)
        elif isinstance(params["name"], str) or library == "sklearn":
            if params is None:
                params = {}
            model = factory(params.pop("name"), **params)
            # Build sklearn pipeline
            if len(self.sklearn_pipeline) > 0:
                model = self.build_sklearn_pipeline(model)
        if len(self.art_pipeline) > 0 or art is True:
            model = self.build_art_pipeline(model, library)
        return model

    def save_model(self, model, filename):
        library = filetypes[Path(filename).suffix.split(".")[-1]]
        filename.parent.mkdir(parents=True, exist_ok=True)
        if library == "torch":
            import torch

            torch.save(model.model, filename)
        elif library == "tensorflow":
            model.model.save(filename)
        elif library == "tfv1":
            model.model.save(filename.stem)
        elif library == "keras":
            model.model.save(filename)
        else:
            with open(filename, "wb") as f:
                pickle.dump(model, f)
        return Path(filename).resolve()

    def build_sklearn_pipeline(self, model):
        if not isinstance(model, Pipeline):
            model = Pipeline(steps=[("model", model)])
        i = 0
        for entry in self.sklearn_pipeline:
            config = deepcopy(self.sklearn_pipeline[entry])
            name = config.pop("name")
            object_ = factory(name, **config)
            model.steps.insert(i, (name, object_))
            object_ = factory(name, **config)
            i += 1
        return model

    def build_art_pipeline(self, model, library):
        init_params = deepcopy(dict(self.init))
        art = self.art_pipeline
        if "preprocessor_defence" in art:
            preprocessor_defences = [
                load_from_tup(
                    (
                        art["preprocessor_defence"]["name"],
                        art["preprocessor_defence"]["params"],
                    ),
                ),
            ]
        else:
            preprocessor_defences = None
        if "postprocessor_defence" in art:
            postprocessor_defences = [
                load_from_tup(
                    (
                        art["postprocessor_defence"]["name"],
                        art["postprocessor_defence"]["params"],
                    ),
                ),
            ]
        else:
            postprocessor_defences = None
        if library == "sklearn":
            if is_regressor(model) is False:
                model = ScikitlearnClassifier(
                    model,
                    postprocessing_defences=postprocessor_defences,
                    preprocessing_defences=preprocessor_defences,
                )
            else:
                model = ScikitlearnRegressor(
                    model,
                    postprocessing_defences=postprocessor_defences,
                    preprocessing_defences=preprocessor_defences,
                )
        elif library == "torch":
            model = PyTorchClassifier(
                model,
                optimizer=factory(
                    init_params["optimizer"].pop("name"),
                    params=model.parameters(),
                    **init_params["optimizer"],
                ),
                loss=factory(init_params["loss"].pop("name"), **init_params["loss"]),
                clip_values=init_params.pop("clip_values"),
                input_shape=init_params.pop("input_shape"),
                nb_classes=init_params.pop("num_classes"),
                postprocessing_defences=postprocessor_defences,
                preprocessing_defences=preprocessor_defences,
                **init_params,
                # output="logits",
            )
        elif library == "tensorflow":
            model = TensorFlowClassifier(
                model,
                postprocessing_defences=postprocessor_defences,
                preprocessing_defences=preprocessor_defences,
                output="logits",
                **init_params,
            )
        elif library == "tfv1":
            model = TensorFlowClassifier(
                model,
                postprocessing_defences=postprocessor_defences,
                preprocessing_defences=preprocessor_defences,
                output="logits",
                **init_params,
            )
        elif library == "keras":
            
            model = KerasClassifier(
                model,
                postprocessing_defences=postprocessor_defences,
                preprocessing_defences=preprocessor_defences,
                **init_params,
            )
        elif library == "tensorflowv2":
            model = TensorFlowV2Classifier(
                model,
                postprocessing_defences=postprocessor_defences,
                preprocessing_defences=preprocessor_defences,
                output="logits",
                **init_params,
            )
        else:
            raise ValueError(f"Library {library} not supported")
        if "transformer_defence" in art:
            model = load_from_tup(
                (
                    art["transformer_defence"]["name"],
                    art["transformer_defence"]["params"],
                ),
                model,
            )()
        if "trainer_defence" in art:
            model = load_from_tup(
                (art["trainer_defence"]["name"], art["trainer_defence"]["params"]),
                model,
            )()
        return model


config = """
    init:
        loss: "hinge"
        name: sklearn.linear_model.SGDClassifier
    # fit:
    #     epochs: 1000
    #     learning_rate: 1.0e-08
    #     log_interval: 10
    art_pipeline:
        preprocessor:
            name: art.defences.preprocessor.FeatureSqueezing
            bit_depth: 32
    sklearn_pipeline:
        feature_selection:
            name: sklearn.preprocessing.StandardScaler
            with_mean : true
            with_std : true
"""
