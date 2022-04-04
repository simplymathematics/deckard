import logging
# import pipeline from sklearn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from hashlib import md5 as my_hash
import json
from copy import deepcopy
logger = logging.getLogger(__name__)

class Model(object):
    """Creates a model object that includes a dicitonary of passed parameters."""
    def __init__(self, estimator:callable, model_type:str = None, verbose : int = 1):
        """
        Initialize the model object.
        estimator: the estimator to use
        model_type: the type of the model. Use 'tf1' for TensorFlow 1.x models.
        verbose: the verbosity level
        """
        logger.info("Model type during init: {}".format(type(estimator)))
        if isinstance(estimator, (Pipeline, BaseEstimator)):
            self.params = dict(estimator.get_params(deep=True))
        else:
            self.params = estimator.__dict__
        self.model = estimator
        self.verbose = verbose
        if model_type is None:
            if isinstance(estimator, (Pipeline, BaseEstimator)):
                self.model_type = 'sklearn'
            #TODO elif statements for other classifier types, automatically convert to ART model. Port some attributes and methods from experiment.py
            else:
                raise ValueError("Non-sklearn model type must be specified.")
        else:
            self.model_type = model_type     
        self.name = str(hash(self))
        self.params.update({"Name": self.name})
    
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
            
def validate_model():
    raise NotImplementedError("validate_model not implemented")

if __name__ == "__main__":
    # set logging to debug
    from sklearn.model_selection import GridSearchCV
    import sys
    logging.basicConfig(level=logger.DEBUG)
    # import standard scaler
    from sklearn.preprocessing import StandardScaler
    # import linear regression
    from sklearn.linear_model import LinearRegression
    # import pipeline
    from sklearn.pipeline import Pipeline
    model = LinearRegression()
    params = {'fit_intercept': [True, False], 'normalize': [True, False]}
    model = Model(model)
    model2 = Model(GridSearchCV(LinearRegression(), params, cv=5))
    assert model.name == model2.name
    sys.exit(0)