import logging, os, json, pickle
import numpy as np
# import pipeline from sklearn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from hashlib import md5 as my_hash
from urllib.parse import urlparse as is_url
from copy import deepcopy
from art.estimators.classification import PyTorchClassifier, TensorFlowClassifier, KerasClassifier, TensorFlowV2Classifier
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from art.estimators import ScikitlearnEstimator
from art.estimators.regression import ScikitlearnRegressor
from art.utils import get_file
from sklearn.base import is_classifier, is_regressor
from typing import Callable, Union
from deckard.base import Data
from time import process_time

logger = logging.getLogger(__name__)


supported_estimators = [PyTorchClassifier, TensorFlowClassifier, KerasClassifier, ScikitlearnClassifier, ScikitlearnRegressor, ScikitlearnEstimator, TensorFlowV2Classifier]


class Model(object):
    """Creates a model object that includes a dicitonary of passed parameters."""
    def __init__(self, model:Union[str, object], model_type:str,  defence: object = None, path = ".", url = None, is_fitted:bool = False):
        """
        Initialize the model object.
        estimator: the estimator to use
        model_type: the type of the model. Use 'tf1' for TensorFlow 1.x models.
        verbose: the verbosity level
        """
        self.model_type =  model_type
        self.defence = defence
        self.path = path
        self.url = url
        self.is_fitted = is_fitted
        if isinstance(model, str):
            self.filename = model
            self.load(model)
        elif isinstance(model, object):
            self.filename = str(type(model))
            self.model = model
            self.load(model)
        else:
            raise ValueError("Model must be a string or a callable")
        assert self.classifier or self.regressor, "Model is neither classifier nor regressor"
        assert not (self.classifier and self.regressor), "Model detected as both classifier and regressor"
        assert hasattr(self, 'model'), "Error initializing model"
        
       

    def __hash__(self) -> str:
        """
        Return the hash of the model, using the params from __init__. 
        """
        new_string = str(self.get_params())
        return int(my_hash(str(new_string).encode('utf-8')).hexdigest(), 36)

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
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        elif hasattr(self.model.model, 'set_params'):
            self.model.model.set_params(**params)
        else:
            self.model.__dict__.update(params)

    
    def get_params(self):
        params = {}
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'get_params'):
            try:
                params = dict(self.model.model.get_params(deep=True))
            except:
                params = dict(self.model.model.get_params())
        elif hasattr(self.model, 'get_params'):
            try:
                params = dict(self.model.get_params(deep=True))
            except:
                params = dict(self.model.get_params())
        else:
            params = {'model' : self.model, 'model_type' : self.model_type, 'path' : self.path, 'url' : self.url, 'defence' : self.defence}
        for key, value in params.items():
            if isinstance(value, int):
                params[key] = value
            elif isinstance(value, float):
                params[key] = value
            elif isinstance(value, str):
                params[key] = value
            elif isinstance(value, Callable):
                params[key] = str(type(value))
            else:
                params[key] = str(type(value))
        self.name = self._set_name(params)
        params.update({"Name": self.name})
        return params
    
    def set_defence_params(self) -> None:
        """
        Adds a defence to an experiment
        :param experiment: experiment to add defence to
        :param defence: defence to add
        """
        assert isinstance(self.defence, object)
        if self.defence is not None:
            def_params = {}
            for key, value in self.defence.__dict__.items():
                # Parses the parameters
                if isinstance(value, int):
                    def_params[key] = value
                elif isinstance(value, float):
                    def_params[key] = value
                elif isinstance(value, str):
                    def_params[key] = value
                elif isinstance(value, tuple):
                    def_params[key] = value
                elif isinstance(value, Callable):
                    def_params[key] = str(type(value))
                else:
                    def_params[key] = str(type(value))
                # Finds type and records it
                if 'trainer' in str(type(self.defence)):
                    raise NotImplementedError("Trainer defence not supported")
                elif 'transformer' in str(type(self.defence)):
                    raise NotImplementedError("Transformer defence not supported")
                elif 'preprocessor' in str(type(self.defence)):
                    defence_type = 'preprocessor'
                elif 'postprocessor' in str(type(self.defence)):
                    defence_type = 'postprocessor'
                else:
                    raise NotImplementedError("Defence {} not supported".format(type(self.defence)))
            self.params['Defence'] = {'name': str(type(self.defence)), 'params': def_params, 'type' : defence_type}
        else:
            self.params['Defence'] = {'name': None, 'params': None, 'type' : None}


    def load_from_string(self, filename:str) -> None:
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
        assert filename is not None,  "filename must be specified"
        assert path is not None or url is not None, "path or url must be specified"
        output_dir = os.path.dirname(path)
        if url is not None:
            # download model
            model_path = get_file(filename = filename, extract=False, path=path, url=url, verbose = True)
            logging.info("Downloaded model from {} to {}".format(url, model_path))
            filename = os.path.join(path, filename)   
        if model_type == 'keras' or filename.endswith('.h5'):
            from tensorflow.keras.models import load_model as keras_load_model
            model = keras_load_model(filename)
        elif model_type == 'torch' or  model_type == 'pytorch':
            from torch import load as torch_load
            model = torch_load(filename)
        elif model_type == 'tf' or model_type == 'tensorflow':
            from tensorflow.keras.models import load_model as tf_load_model
            model = tf_load_model(filename)
        elif model_type == 'tfv1' or model_type == 'tensorflowv1' or model_type == 'tf1':
            import tensorflow.compat.v1 as tfv1
            tfv1.disable_eager_execution()
            from tensorflow.keras.models import load_model as tf_load_model
            model = tf_load_model(filename)
        elif model_type == 'sklearn' or model_type == 'pipeline' or model_type == 'gridsearch' or model_type == 'pickle':
            if self.path:
                filename = os.path.join(self.path, filename)
            with open(filename, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError("Model type {} not supported".format(model_type))
            logger.info("Loaded model")
        return model
    
    def load(self, filename:str) -> None:
        """
        Load a model from a pickle file.
        filename: the pickle file to load the model from
        """
        logger.debug("Loading model")
        if not hasattr(self, 'model'):
            # load the model
            self.model = self.load_from_string(filename)
        else:
            logger.info("Model already in memory.")
        self.regressor = is_regressor(self.model)
        self.classifier = not self.regressor
        self.params = self.get_params()
        self.set_defence_params()
        self._set_name(self.params)
        logger.info("Loaded model")
        if self.classifier:
            self.initialize_art_classifier()
        elif self.regressor:
            self.initialize_art_regressor()
        self.is_supervised = self._is_supervised()
 
        
       
    
    
    def _is_supervised(self)-> bool:
        """
        Returns true if supervised, false if unsupervised. 
        """
        if hasattr(self.model, 'fit_predict') or  hasattr(self.model, 'fit_transform'):
            result = False
            logger.info("Model is unsupervised")
        elif hasattr(self.model, 'model') and (hasattr(self.model.model, 'fit_predict') or  hasattr(self.model.model, 'fit_transform')):
            result = False
            logger.info("Model is unsupervised")
        elif hasattr(self.model, 'fit'):
            result = True
            logger.info("Model is supervised")
        else:
            raise ValueError("Model is not a classifier or regressor. It is type {}".format(type(self.model)))
        return result

    def initialize_art_classifier(self):
        """
        Initialize the classifier.
        """
        preprocessors = []
        postprocessors = []
        trainers = []
        transformers = []
        # Find defence type
        if self.params['Defence']['type'] == 'preprocessor':
            preprocessors.append(self.defence)
        elif self.params['Defence']['type'] == 'postprocessor':
            postprocessors.append(self.defence)
        elif self.params['Defence']['type'] == 'trainer':
            trainers.append(self.defence)
        elif self.params['Defence']['type'] == 'transformer':
            transformers.append(self.defence)
        elif self.params['Defence']['type'] == None:
            pass
        else:
            raise ValueError("Defence type {} not supported".format(self.params['Defence']['type']))
        # Iinitialize model by type
        if self.model_type in ['pytorch', 'torch', 'pyt']:
            from art.estimators.classification import PyTorchClassifier
            model = PyTorchClassifier(self.model, preprocessing_defences = preprocessors, postprocessing_defences = postprocessors)
        elif self.model_type in ['keras']:
            from art.estimators.classification import KerasClassifier
            model = KerasClassifier(self.model, preprocessing_defences = preprocessors, postprocessing_defences = postprocessors)
        elif self.model_type in ['tensorflow', 'tf', 'tf1', 'tfv1', 'tensorflowv1']:
            from art.estimators.classification import KerasClassifier
            model = KerasClassifier(self.model, preprocessing_defences = preprocessors, postprocessing_defences = postprocessors)
        elif self.model_type in ['tf2', 'tensorflow2', 'tfv2', 'tensorflowv2']:
            from art.estimators.classification import KerasClassifier
            model = KerasClassifier(self.model)
        elif self.model_type in ['sklearn', 'pipeline', 'gridsearch', 'pickle', 'scikit', 'scikit-learn']:
            model = ScikitlearnClassifier(self.model, preprocessing_defences = preprocessors, postprocessing_defences = postprocessors)
        self.model = model
        
    
    def initialize_art_regressor(self):
        preprocessors = []
        postprocessors = []
        trainers = []
        transformers = []
        # Find defence type
        if self.params['Defence']['type'] == 'preprocessor':
            preprocessors.append(self.defence)
        elif self.params['Defence']['type'] == 'postprocessor':
            postprocessors.append(self.defence)
        elif self.params['Defence']['type'] == 'trainer':
            trainers.append(self.defence)
        elif self.params['Defence']['type'] == 'transformer':
            transformers.append(self.defence)
        elif self.params['Defence']['type'] == None:
            pass
        else:
            raise ValueError("Defence type {} not supported".format(self.params['Defence']['type']))
        # Initialize model by type
        if self.model_type in ['sklearn', 'scikit-learn', 'scikit']:
            model = ScikitlearnRegressor(model=self.model, preprocessing_defences = preprocessors, postprocessing_defences = postprocessors)
        else:
            raise ValueError("Model type {} not supported".format(self.model_type))
        self.model = model
    
    def save(self, filename:str = None, path:str = None):
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
        if hasattr(self.model, "save"):
            self.model.save(filename = filename, path = path)
        else:
            with open(os.path.join(path, filename), 'wb') as f:
                pickle.dump(self.model, f)
        return os.path.join(path, filename)
    
    def run_model(self, data:Data, **kwargs) -> None:
        """
        Builds model and returns self with added time_dict and predictions attributes.
        """
        self.time_dict = {}
        self.fit(data.X_train, data.y_train, **kwargs)
        self.predictions = self.predict(data.X_test)
        return None


    
    def fit(self, X_train:np.ndarray, y_train:np.ndarray = None) -> None:
        """
        Fits model.
        """
        if self.defence is not None:
            # Skips defences that don't need to be fit
            if self.defence._apply_fit == True and self.is_fitted == False:
                self.is_fitted = False 
            elif self.defence._apply_fit == False and self.is_fitted == True:
                self.is_fitted = True
            elif self.defence._apply_fit == False and self.is_fitted == False:
                self.is_fitted = False
            else:
                self.is_fitted = False
        if self.is_fitted:
            logger.warning("Model is already fitted")
            self.time_dict = {'fit_time': np.nan}
        else:
            try:
                start = process_time()
                self.model.fit(X_train, y_train)
                end = process_time()
            except np.AxisError as e:
                from sklearn.preprocessing import LabelBinarizer
                y_train = LabelBinarizer().fit(y_train).transform(y_train)
                start = process_time()
                self.model.fit(X_train, y_train)
                end = process_time()
                # raise e
            self.time_dict = {'fit_time': end - start}
            self.is_fitted = True
        return None
    
    def predict(self, X_test:np.ndarray) -> np.ndarray:
        start = process_time()
        predictions = self.model.predict(X_test)
        end = process_time()
        if not hasattr(self, 'time_dict'):
            self.time_dict['fit_time'] = None
        self.time_dict['pred_time'] = end - start
        return predictions