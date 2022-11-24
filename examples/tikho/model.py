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


class Model(
    collections.namedtuple(
        typename="Data",
        field_names="name,  params, fit, predict, transform, sklearn_pipeline, art_pipeline, url, library",
        defaults=({}, {}, {}, {}, [], [], "", ""),
    ),
):
    def __new__(cls, loader, node):
        return super().__new__(cls, **loader.construct_mapping(node))

    def load(self):
        if is_url(self.url):
            name = Path(self.name).name
            path = Path(self.name).parent
            assert hasattr(
                self,
                "library",
            ), "library must be specified if model is loaded from url"
            library = self.library
            model = get_file(name, self.url, path)
        elif isinstance(self.name, str):
            library = self.name.split(".")[0]
            if self.params is None:
                self.params = {}
            model = factory(self.name, **self.params)
        else:
            raise ValueError(f"Unknown model: {self.name}")
        if Path(str(model)).exists():
            assert hasattr(
                self,
                "library",
            ), "library must be specified if model is loaded from file"
            library = self.library
            if library == "sklearn":
                with open(model, "rb") as f:
                    model = pickle.load(f)
            elif library == "torch":
                from torch import load

                model = load(model)
            elif library == "tensorflow":
                from tensorflow import keras

                model = keras.models.load_model(model)
            elif library == "tensorflow2":
                from tensorflow import keras

                model = keras.models.load_model(model)
            elif library == "keras":
                from tensorflow import keras

                model = keras.models.load_model(model)
            elif library == "tfv1":
                import tensorflow.compat.v1 as tfv1

                tfv1.disable_eager_execution()
                from tensorflow.keras.models import load_model as tf_load_model

                model = tf_load_model(model)
            else:
                raise ValueError(
                    f"library must be one of 'sklearn', 'torch', 'tensorflow', 'tensorflow2', 'keras'. It is {library}",
                )

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
        if len(self.art_pipeline) > 0:
            art = self.art_pipeline
            preprocessor_defences = (
                [
                    load_from_tup(
                        (
                            art["preprocessor_defence"]["name"],
                            art["preprocessor_defence"]["params"],
                        ),
                    ),
                ]
                if "preprocessor_defence" in art
                else None
            )
            postprocessor_defences = (
                [
                    load_from_tup(
                        (
                            art["postprocessor_defence"]["name"],
                            art["postprocessor_defence"]["params"],
                        ),
                    ),
                ]
                if "postprocessor_defence" in art
                else None
            )
            if library == "sklearn":
                if is_regressor is False:
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
            model = (
                load_from_tup(
                    (
                        art["transformer_defence"]["name"],
                        art["transformer_defence"]["params"],
                    ),
                    model,
                )()
                if "transformer_defence" in art
                else model
            )
            model = (
                load_from_tup(
                    (art["trainer_defence"]["name"], art["trainer_defence"]["params"]),
                    model,
                )()
                if "trainer_defence" in art
                else model
            )
        return model

    def save_model(self, model):
        filename = Path(self.name).name
        if hasattr(self, "model") and hasattr(model, "save"):
            flag = False
            if filename.endswith(".pickle"):
                filename = filename[:-7]
                flag = True
            model.save(filename)
            if flag is True:
                filename = filename + ".pickle"
        else:
            with open(filename, "wb") as f:
                pickle.dump(model, f)
        return filename


if "__main__" == __name__:
    model_document = """
        name : sklearn.linear_model.SGDClassifier
        params:
            loss: log
        sklearn_pipeline:
        - preprocessor : {name: sklearn.preprocessing.StandardScaler, params: {with_mean: True, with_std: True}}
        - feature_selection : {name: sklearn.feature_selection.SelectKBest, params: {k: 10}}
        art_pipeline:
            preprocessor_defence : {name: art.defences.preprocessor.FeatureSqueezing, params: {bit_depth: 4, clip_values: [0, 1]}}
            postprocessor_defence : {name: art.defences.postprocessor.HighConfidence, params: {cutoff: 0.9}}
    """

    yaml.add_constructor("!Model:", Model)
    model_document_tag = """!Model:""" + model_document
    model = yaml.load(model_document_tag, Loader=yaml.Loader)
    assert hasattr(model.load(), "fit")
    assert hasattr(model.load(), "predict")
    assert isinstance(model.load(), supported_estimators)
    # model_document = """
    #     name : "art_models/model.h5"
    #     url : https://www.dropbox.com/s/hbvua7ynhvara12/cifar-10_ratio%3D0.h5?dl=1
    #     library : tensorflow
    #     TODO: Test on GPU
    #     art_pipeline:
    #         preprocessor_defence : {name: art.defences.preprocessor.FeatureSqueezing, params: {bit_depth: 4, clip_values: [0, 1]}}
    #         postprocessor_defence : {name: art.defences.postprocessor.HighConfidence, params: {cutoff: 0.9}}
    #         transformer_defence : {name: art.defences.transformer.evasion.DefensiveDistillation, params: {batch_size: 128}}
    #         trainer_defence : {name: art.defences.trainer.AdversarialTrainerMadryPGD, params: {nb_epochs: 10}}
    # """
    # model_document_tag = u"""!Model:""" + model_document
    # # print(model_document_tag)
    # model = yaml.load(model_document_tag, Loader = yaml.Loader)
    # print(model.load())
