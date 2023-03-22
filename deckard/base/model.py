import collections
import logging
import pickle
import warnings
from copy import deepcopy
from pathlib import Path

from art.estimators import ScikitlearnEstimator
from art.estimators.classification import (KerasClassifier, PyTorchClassifier,
                                           TensorFlowClassifier,
                                           TensorFlowV2Classifier)
from art.estimators.classification.scikitlearn import (ScikitlearnClassifier,
                                                       ScikitlearnSVC)
from art.estimators.regression import ScikitlearnRegressor
from art.utils import get_file
from sklearn.base import BaseEstimator, is_regressor
from sklearn.pipeline import Pipeline
from validators import url as is_url

from .hashable import BaseHashable
from .utils import factory

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
    "tf1" : "h5"
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
        if is_url(self.url) or Path(self.init.pop("name")).exists():
            filename = Path(self.init.pop("name"))
            lib = filetypes[filename.suffix.split(".")[-1]]
            library = Path(self.init.pop("library", lib))
            name = filename.name
            path = filename.parent
            if is_url(self.url):
                model = get_file(name, self.url, path)
            if library == "keras":
                from keras.models import load_model
                model = load_model(model)
            elif library == "torch":
                from torch import load

                model = load(model)
            elif library == "tensorflow" or "tf2" or "tfv2":
                from keras.models import load_model
                model = load_model(model)
            elif library == "tfv1" or "tf1":
                from keras.models import load_model
                tf.compat.v1.disable_eager_execution()
                model = load_model(model)
            elif library == "sklearn":
                with open(model, "rb") as f:
                    model = pickle.load(f)
            else:
                raise ValueError("Unsupported library {}".format(library))
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
            model.model.save(filename)
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

    def build_art_pipeline(self, model, library  = "sklearn"):
        init_params = deepcopy(dict(self.init))
        art = self.art_pipeline
        if "preprocessor_defence" in art:
            preprocessor_defences = [
                factory(
                    (
                        art['preprocessor_defence'].pop("name"),
                        art["preprocessor_defence"],
                    ),
                ),
            ]
        else:
            preprocessor_defences = None
        if "postprocessor_defence" in art:
            postprocessor_defences = [
                factory(
                    (
                        art["postprocessor_defence"].pop("name"),
                        art["postprocessor_defence"],
                    ),
                ),
            ]
        else:
            postprocessor_defences = None
        if library == "sklearn":
            if is_regressor(model) is False:
                if hasattr(model, "steps"):
                    model = model.steps[-1][1]
                if "svm" in str(type(model)).lower():
                    model = ScikitlearnSVC(
                        model,
                        postprocessing_defences=postprocessor_defences,
                        preprocessing_defences=preprocessor_defences,
                    )
                else:
                    model = ScikitlearnClassifier(model=model, postprocessing_defences=postprocessor_defences,
                                                  preprocessing_defences=preprocessor_defences)
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
            model = KerasClassifier(
                model,
                use_logits = True,
                postprocessing_defences=postprocessor_defences,
                preprocessing_defences=preprocessor_defences,
                **init_params,
            )
        elif library == "tfv1" or "tf1":
            import tensorflow.compat.v1 as tf
            import tensorflow.compat.v1.keras as keras
            tf.compat.v1.disable_eager_execution()
            model = KerasClassifier(
                model = model,
                postprocessing_defences=postprocessor_defences,
                preprocessing_defences=preprocessor_defences,
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
            model = KerasClassifier(
                model,
                postprocessing_defences=postprocessor_defences,
                preprocessing_defences=preprocessor_defences,
                **init_params,
            )
        else:
            raise ValueError(f"Library {library} not supported")
        if "transformer_defence" in art:
            model = factory(
                    art["transformer_defence"].pop('name'),
                    model, **art["transformer_defence"],
            )()
        if "retrainer_defence" in art:
            assert "attack" in art, "Attack must be specified for retraining"
            assert "name" in art["attack"], "Attack name must be specified for retraining"
            try:
                name = art['attack'].pop('name')
                attack = factory(
                    name, 
                    model, 
                    **art['attack'],
                )
            except Exception as e:
                attack = factory(art['attack'].pop('name'), **art['attack'])
            model = factory(
                art["retrainer_defence"].pop('name'),
                model, attack, **art["retrainer_defence"],
            )
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
