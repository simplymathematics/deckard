import logging
# import pipeline from sklearn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from hashlib import md5 as my_hash
import json
from copy import deepcopy
from art.estimators.classification import PyTorchClassifier, TensorFlowClassifier, KerasClassifier
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from art.estimators import ScikitlearnEstimator
from art.estimators.regression import ScikitlearnRegressor
logger = logging.getLogger(__name__)


supported_estimators = [PyTorchClassifier, TensorFlowClassifier, KerasClassifier, ScikitlearnClassifier, ScikitlearnRegressor, ScikitlearnEstimator]


class Model(object):
    """Creates a model object that includes a dicitonary of passed parameters."""
    def __init__(self, estimator:callable, model_type:str = None, verbose : int = 1, params:dict = {}):
        """
        Initialize the model object.
        estimator: the estimator to use
        model_type: the type of the model. Use 'tf1' for TensorFlow 1.x models.
        verbose: the verbosity level
        """
        logger.info("Model type during init: {}".format(type(estimator)))
        if isinstance(estimator, (Pipeline, BaseEstimator)):
            self.params = dict(estimator.get_params(deep=True))
            self.params.update(params)
        elif isinstance(estimator, (PyTorchClassifier, TensorFlowClassifier, KerasClassifier, ScikitlearnClassifier)):
            self.params = dict(estimator.get_params())                
            self.params.update(params)
        else:
            logger.warning("Cannot auto detect reproducible model parameters. Please specify params manually.")
            self.params = params
        self.model = estimator
        self.verbose = verbose
        if model_type is None:
            if isinstance(estimator, PyTorchClassifier):
                self.model_type = 'torch'
            elif isinstance(estimator, TensorFlowClassifier):
                self.model_type = 'tf'
            elif isinstance(estimator, KerasClassifier):
                self.model_type = 'keras'
            elif isinstance(estimator, (ScikitlearnClassifier, Pipeline, BaseEstimator)):
                self.model_type = 'sklearn'
            else:
                raise ValueError("Model type not specified and cannot be auto detected. Type is {}".format(type(estimator)))
        else:
            self.model_type = model_type   
        self.name = self._set_name(params)
        self.params.update({"id_": self.name})
    
    def __hash__(self) -> str:
        """
        Return the hash of the model, using the params from __init__. 
        """
        new_string = str(self.params)
        return int(my_hash(json.dumps(new_string).encode('utf-8')).hexdigest(), 36)

    def __eq__(self, other) -> bool:
        """
        Returns True if the models are equal, using the has of the params from __init__.
        """
        return self.__hash__() == other.__hash__()
    
    def _set_name(self, params):
        """
        :param params: A dictionary of parameters to set.
        Sets the name of the model. 
        """
        assert params is not None, "Params must be specified"
        assert isinstance(params, dict), "Params must be a dictionary"
        name = str(type(self.model)).replace("<class '", "").replace("'>", "") + "_"
        for key, value in params.items():
            name += "_" + str(key) + ":" + str(value)
        return name

    def set_params(self, params:dict = None):
        """
        :param params: A dictionary of parameters to set.
        Sets the extra parameters of the Model object. 
        """
        assert params is not None, "Params must be specified"
        assert isinstance(params, dict), "Params must be a dictionary"
        for key, value in params.items():
                self.params[key] = value
    
    def get_params(self):
        return self.params