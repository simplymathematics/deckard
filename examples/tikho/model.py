import logging
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
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from deckard.base.model import Model

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
