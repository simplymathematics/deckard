import collections
import logging
import pickle
from pathlib import Path
import warnings
import yaml
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
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, is_regressor

from .utils import factory, load_from_tup
from validators import url as is_url
from .hashable import BaseHashable, my_hash
from copy import deepcopy

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
    "pt": "pytorch",
    "pth": "pytorch",
    "pb": "tensorflow",
    "pbtxt": "tensorflow",
    "tflite": "tf-lite",
    "pickle": "sklearn",
}


class Model(
    collections.namedtuple(
        typename="model",
        field_names="init, files, fit, predict, sklearn_pipeline, art_pipeline, url, library",
        defaults=({}, {}, {}, [], [], "", ""),
    ),
    BaseHashable,
):
    def __new__(cls, loader, node):
        return super().__new__(cls, **loader.construct_mapping(node))

    def load(self, art = False):
        filename = Path(
            self.files["model_path"],
            my_hash(self._asdict()) + "." + self.files["model_filetype"],
        )
        params = deepcopy(self.init)
        if is_url(self.url):
            name = filename.name
            path = filename.parent
            model = get_file(name, self.url, path)
        elif isinstance(params["name"], str):
            library = params["name"].split(".")[0]
            if params is None:
                params = {}
            model = factory(params.pop("name"), **params)
        else:
            raise ValueError(f"Unknown model: {params['name']}")

        # Build sklearn pipeline
        if len(self.sklearn_pipeline) > 0:
            if not isinstance(model, Pipeline):
                model = Pipeline(steps=[("model", model)])
            i = 0
            for entry in self.sklearn_pipeline:
                _ = list(entry.keys())[0]
                _ = entry[_]
                name = _["name"]
                params = _["params"]
                object_ = factory(name, **params)
                model.steps.insert(i, (name, object_))
                object_ = factory(name, **params)
                i += 1
        # Build art pipeline
        if len(self.art_pipeline) > 0 or art is True:
            art = self.art_pipeline
            if "preprocessor_defence" in art:
                preprocessor_defences = (
                    [
                        load_from_tup(
                            (
                                art["preprocessor_defence"]["name"],
                                art["preprocessor_defence"]["params"],
                            ),
                        ),
                    ]
                )
            else:
                preprocessor_defences = None
            if "postprocessor_defence" in art:
                postprocessor_defences = (
                    [
                        load_from_tup(
                            (
                                art["postprocessor_defence"]["name"],
                                art["postprocessor_defence"]["params"],
                            ),
                        ),
                    ]
                )
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
                    postprocessing_defences=postprocessor_defences,
                    preprocessing_defences=preprocessor_defences,
                    output="logits",
                )
            elif library == "tensorflow":
                model = TensorFlowClassifier(
                    model,
                    postprocessing_defences=postprocessor_defences,
                    preprocessing_defences=preprocessor_defences,
                    output="logits",
                )
            elif library == "tfv1":
                model = TensorFlowClassifier(
                    model,
                    postprocessing_defences=postprocessor_defences,
                    preprocessing_defences=preprocessor_defences,
                    output="logits",
                )
            elif library == "keras":
                model = KerasClassifier(
                    model,
                    postprocessing_defences=postprocessor_defences,
                    preprocessing_defences=preprocessor_defences,
                    output="logits",
                )
            elif library == "tensorflowv2":
                model = TensorFlowV2Classifier(
                    model,
                    postprocessing_defences=postprocessor_defences,
                    preprocessing_defences=preprocessor_defences,
                    output="logits",
                )
            if "transformer_defence" in art:
                model = (
                    load_from_tup(
                        (
                            art["transformer_defence"]["name"],
                            art["transformer_defence"]["params"],
                        ),
                        model,
                    )()
                )
            if "trainer_defence" in art:
                model = (
                    load_from_tup(
                        (art["trainer_defence"]["name"], art["trainer_defence"]["params"]),
                        model,
                    )()
                )
        return model

    def save(self, model):
        filename = Path(
            self.files["model_path"],
            my_hash(self._asdict()) + "." + self.files["model_filetype"],
        )
        filename.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self, "model") and hasattr(model, "save"):
            flag = False
            # Hacky workaround for art sklearn saving due to a bug in art.
            if filename.endswith(".pickle"):
                filename = filename[:-7]
                flag = True
            ##############################################################
            # Using art to save models
            model.save(filename)
            ##############################################################
            # Hacky workaround for art sklearn saving due to a bug in art.
            if flag is True:
                filename = filename + ".pickle"
            ##############################################################
        else:
            with open(filename, "wb") as f:
                pickle.dump(model, f)
        return Path(filename).resolve()
