import logging, os, pickle


# Operating System
from time import process_time
import json
from typing import Union
from copy import deepcopy
from pandas import DataFrame, Series

# Math Stuff
import numpy as np
from pandas import Series
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


from hashlib import md5 as my_hash
from deckard.base.model import Model
from deckard.base.data import Data
from deckard.base.hashable import BaseHashable

from art.defences.postprocessor import Postprocessor
from art.defences.preprocessor import Preprocessor
from art.defences.trainer import Trainer
from art.defences.transformer import Transformer
DEFENCE_TYPES = [Preprocessor, Trainer, Transformer, Postprocessor]

logger = logging.getLogger(__name__)


# Create experiment object
class Experiment(BaseHashable):
    """
    Creates an experiment object
    """
    def __init__(self, data:Data, model:Model, verbose : int = 1, is_fitted:bool = False):
        """
        Creates an experiment object
        :param data: Data object
        :param model: Model object
        :param params: Dictionary of other parameters you want to add to this object. Obviously everything in self.__dict__.keys() should be treated as a reserved keyword, however.
        :param verbose: Verbosity level
        :param scorers: Dictionary of scorers
        :param name: Name of experiment
        """
        self.model = model
        self.data = data
        self.verbose = verbose
        self.is_fitted = is_fitted
        self.time_dict = None
        self.params = {}
        self.params['Model'] = dict(self.model.params)
        self.params['Data'] = dict(self.data.params)
        self.hash = hash(self)
        self.params['Experiment'] = {'name': self.hash, 'verbose': self.verbose, 'is_fitted': self.is_fitted, 'id': hash(self), 'model': hash(model), 'data': hash(data)}
        
        
    def run(self, **kwargs) -> None:
        """
        Builds model.
        """
        time_dict = {}
        self.model.is_fitted = self.is_fitted
        if not self.is_fitted:
            start = process_time()
            self.model.fit(self.data.X_train, self.data.y_train, **kwargs)
            end = process_time()
            time_dict['fit'] = end - start
        else:
            time_dict['fit'] = np.nan
        start = process_time()
        self.predictions =self.model.predict(self.data.X_test)
        end = process_time()
        time_dict['predict'] = end - start
        self.time_dict = time_dict
        self.hash = hash(self)
    

    def __call__(self, path, filename:str = None, prefix = None, **kwargs) -> None:
        """
        Sets metric scorer. Builds model. Runs evaluation. Updates scores dictionary with results. Returns self with added scores, predictions, and time_dict attributes.
        """
        self.data()
        self.model()
        self.ground_truth = self.data.y_test
        if not os.path.isdir(path):
            os.mkdir(path)
        self.run(**kwargs)
        params_file = self.save_params(path = path, prefix = prefix, )
        preds_file = self.save_predictions(path = path, prefix = prefix)
        truth_File = self.save_ground_truth(path = path, prefix = prefix)
        time_file = self.save_time_dict(path = path, prefix = prefix)
        model_name = str(hash(self.model)) if filename is None else filename
        model_file = self.save_model(filename = model_name, path = path)
        # TODO: Fix scoring
        return (params_file, preds_file, truth_File, model_file, time_file)
    
    
    ##########################################################################################################
    
    def insert_sklearn_preprocessor(self, name:str, preprocessor: object, position:int):
        """
        Add a sklearn preprocessor to the experiment.
        :param name: name of the preprocessor
        :param preprocessor: preprocessor to add
        :param position: position to add preprocessor
        
        """
        # If it's already a pipeline
        if isinstance(self.model.model, Pipeline):
            pipe = self.model.model
        elif hasattr(self.model.model, 'model') and isinstance(self.model.model.model, Pipeline):
            pipe = self.model.model.model  
        elif 'art.estimators' in str(type(self.model.model)) and not isinstance(self.model.model.model, Pipeline):
            pipe = Pipeline([('model', self.model.model.model)])
        elif isinstance(self.model.model, BaseEstimator) and not isinstance(self.model.model, Pipeline):
            pipe = Pipeline([('model', self.model.model)])
        else:
            raise ValueError("Cannot make model type {} into a pipeline".format(type(self.model.model)))
        new_model = deepcopy(pipe)
        assert isinstance(new_model, Pipeline)
        new_model.steps.insert(position, (name, preprocessor))
        self.model.model = new_model
    
    ############################################################################################################
    
    def save_data(self, filename:str = "data.pkl", prefix = None, path:str = ".") -> None:
        """
        Saves data to specified file.
        :param filename: str, name of file to save data to. 
        :param path: str, path to folder to save data to. If none specified, data is saved in current working directory. Must exist.
        """
        assert path is not None, "Path to save data must be specified."
        if prefix is not None:
            filename = os.path.join(path, prefix + "_" + filename)
        else:
            filename = os.path.join(path, filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)
        assert os.path.exists(os.path.join(path, filename)), "Data not saved."
        return None

    def save_params(self, prefix = None, path:str = ".", filetype = '.json') -> None:
        """
        Saves data to specified file.
        :param data_params_file: str, name of file to save data parameters to.
        :param model_params_file: str, name of file to save model parameters to.
        :param path: str, path to folder to save data to. If none specified, data is saved in current working directory. Must exist.
        """
        if filetype != '.json':
            raise NotImplementedError("Only json files are supported at the moment")
        filenames = []
        assert path is not None, "Path to save data must be specified."
        if not os.path.isdir(path) and not os.path.exists(path):
            os.mkdir(path)
        
        for key, value in self.params.items():
            if prefix is not None:
                filename = prefix + key.lower() + "_" + key + filetype
            else:
                filename = key.lower() +"_params" + filetype
            filename = os.path.join(path, filename)
            with open(filename, 'w') as f:
                json.dump(str(self.params[key]), f, indent = 4)
            filenames.append(os.path.join(path,filename))
            logger.info("Saving {} parameters to {}".format(key, os.path.join(path,filename)))
        return filenames

    def save_model(self, filename:str = "model", prefix = None, path:str = ".") -> str:
        """
        Saves model to specified file (or subfolder).
        :param filename: str, name of file to save model to. 
        :param path: str, path to folder to save model. If none specified, model is saved in current working directory. Must exist.
        :return: str, path to saved model.
        """
        if prefix is not None:
            filename = prefix + "_" + filename
        assert os.path.isdir(path), "Path {} to experiment does not exist".format(path)
        logger.info("Saving model to {}".format(os.path.join(path,filename)))
        self.model.save_model(filename = filename, path = path)
    
    def save_predictions(self, filename:str = "predictions.json", prefix = None, path:str = ".") -> None:
        """
        Saves predictions to specified file.
        :param filename: str, name of file to save predictions to. 
        :param path: str, path to folder to save predictions. If none specified, predictions are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"    
        if prefix is not None:
            filename = prefix + "_" + filename
        prediction_file = os.path.join(path, filename)
        results = self.predictions
        results = DataFrame(results)
        results.to_json(prediction_file, orient='records')
        assert os.path.exists(prediction_file), "Prediction file not saved"
        return None
    
    def save_ground_truth(self, filename:str = "ground_truth.json", prefix = None, path:str = ".") -> None:
        """
        Saves ground_truth to specified file.
        :param filename: str, name of file to save ground_truth to. 
        :param path: str, path to folder to save ground_truth. If none specified, ground_truth are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist" 
        if prefix is not None:
            filename = prefix + "_" + filename  
        prediction_file = os.path.join(path, filename)
        results = self.ground_truth
        results = DataFrame(results)
        results.to_json(prediction_file, orient = 'records')
        assert os.path.exists(prediction_file), "Prediction file not saved"
        return None
    
    def save_cv_scores(self, filename:str = "cv_scores.json", prefix = None, path:str = ".") -> None:
        """
        Saves crossvalidation scores to specified file.
        :param filename: str, name of file to save crossvalidation scores to.
        :param path: str, path to folder to save crossvalidation scores. If none specified, scores are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        assert filename is not None, "Filename must be specified"
        if prefix is not None:
            filename = prefix + "_" + filename
        cv_file = os.path.join(path, filename)
        try:
            cv_results = Series(self.model.model.model.cv_results_, name = str(self.hash))
        except:
            cv_results = Series(self.model.model.cv_results_, name = str(self.hash))
        cv_results.to_json(cv_file, orient='records')
        assert os.path.exists(cv_file), "CV results file not saved"

    def save_time_dict(self, filename:str = "time_dict.json", prefix = None, path:str = "."):
        """
        Saves time dictionary to specified file.
        :param filename: str, name of file to save time dictionary to.
        :param path: str, path to folder to save time dictionary. If none specified, time dictionary is saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        assert hasattr(self, "time_dict"), "No time dictionary to save"
        if prefix is not None:
            filename = prefix + "_" + filename
        time_file = os.path.join(path, filename)
        time_results = Series(self.time_dict, name = str(self.hash))
        time_results.to_json(time_file, orient='records')
        assert os.path.exists(time_file), "Time dictionary file not saved"
        return None
    


    
        

