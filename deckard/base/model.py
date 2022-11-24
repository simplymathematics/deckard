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

from utils import factory, load_from_tup
from validators import url as is_url
from hashable import BaseHashable, my_hash
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


# classification_visualizers = {
#     "confusion": confusion_matrix,
#     "classification": classification_report,
#     "roc_auc": roc_auc
# }

# regression_visualizers = {
#     "error" : prediction_error,
#     "residuals" : residuals_plot,
#     "alphas" : alphas
# }

# clustering_visualizers = {
#     "silhouette" : silhouette_visualizer,
#     "elbow" : kelbow_visualizer,
#     "intercluster" : intercluster_distance
# }
# # elbow requires k
# model_selection_visualizers = {
#     "validation" : validation_curve,
#     "learning" : learning_curve,
#     "cross_validation" : cross_validation,
#     "feature_importances" : feature_importances,
#     "recursive" : rfecv,
#     "dropping_curve" : dropping_curve
# }
# # cross_, recursive, validation needs cv
# # dropping, feats do not need score

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

    def load(self):
        filename = Path(
            self.files["model_path"],
            my_hash(self._asdict()) + "." + self.files["model_filetype"],
        )
        params = deepcopy(self.init)
        library = filetypes[self.files["model_filetype"]]
        if filename.exists():
            model = filename
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
        elif is_url(self.url):
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


if "__main__" == __name__:
    model_document = """
        init:
            loss: "log_loss"
            name: sklearn.linear_model.SGDClassifier
        files:
            model_path : model
            model_filetype : pickle
        fit:
            epochs: 1000
            learning_rate: 1.0e-08
            log_interval: 10

    """

    data_document = """
        sample:
            shuffle : True
            random_state : 42
            train_size : 800
            stratify : True
        add_noise:
            train_noise : 1
            time_series : True
        name: classification
        files:
            data_path : data
            data_filetype : pickle
        generate:
            n_samples: 1000
            n_features: 2
            n_informative: 2
            n_redundant : 0
            n_classes: 2
        sklearn_pipeline:
            - sklearn.preprocessing.StandardScaler
        transform:
            sklearn.preprocessing.StandardScaler:
                with_mean : true
                with_std : true
                X_train : true
                X_test : true
    """
    yaml.add_constructor("!Model:", Model)
    model_document_tag = """!Model:""" + model_document
    # Test that model yaml loads correctly
    model = yaml.load(model_document_tag, Loader=yaml.Loader)
    pre1 = str(model)
    model_ = model.load()
    ret1 = model.save(model_)
    model2 = yaml.load(model_document_tag, Loader=yaml.Loader)
    model2_ = model2.load()
    ret2 = model2.save(model2_)
    filename1 = Path(
        model.files["model_path"],
        my_hash(model._asdict()) + "." + model.files["model_filetype"],
    )
    filename2 = Path(
        model2.files["model_path"],
        my_hash(model2._asdict()) + "." + model2.files["model_filetype"],
    )
    assert filename1 == filename2, f"{filename1} != {filename2}"
    assert ret1 == ret2, f"{ret1} != {ret2}"
    assert filename1.resolve() == ret1.resolve(), f"{filename1} != {ret1}"
