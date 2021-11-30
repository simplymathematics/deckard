import logging
# import pipeline from sklearn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from hashlib import md5 as hash
import json
from copy import deepcopy





#TODO Verify that params are working and hashing is working

class Model(object):
    """Creates a model object that includes a dicitonary of passed parameters."""
    def __init__(self, estimator, verbose : int = 1):
        logging.info("Model type during init: {}".format(type(estimator)))
        assert isinstance(estimator, BaseEstimator), "Estimator must be a sklearn estimator. It is {}".format(type(estimator))
        self.params = dict(estimator.get_params(deep = False))
        self.model = estimator
        self.verbose = verbose
        # self.model_name = self.__dict__['estimator'].__class__.__name__
        # logging.info("Model name: " + self.model_name)
        self.name = str(type(self.model)).split('.')[-1][:-2]
        if 'GridSearch' in self.name:
            self.name = str(type(self.model.__dict__['estimator'])).split('.')[-1][:-2]
        logging.info("Model object: " + self.name)
        self.params.update({"Name": self.name})
    
    def __hash__(self) -> str:
        new_string = str(self.params)
        return int(hash(json.dumps(new_string).encode('utf-8')).hexdigest(), 36)

    def __eq__(self, other) -> bool:
        return self.__hash__() == other.__hash__()
            


if __name__ == "__main__":
    # set logging to debug
    from sklearn.model_selection import GridSearchCV
    import sys
    logging.basicConfig(level=logging.DEBUG)
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