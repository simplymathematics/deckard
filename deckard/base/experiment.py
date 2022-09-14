import logging, os


# Operating System
from time import process_time
from json import dumps, load
from pickle import dump
from typing import Union
from copy import deepcopy


# Math Stuff
import numpy as np
from pandas import Series
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


from hashlib import md5 as my_hash
from deckard.base.model import Model
from deckard.base.data import Data
from deckard.base.storage import DiskStorageMixin

from art.defences.postprocessor import Postprocessor
from art.defences.preprocessor import Preprocessor
from art.defences.trainer import Trainer
from art.defences.transformer import Transformer
DEFENCE_TYPES = [Preprocessor, Trainer, Transformer, Postprocessor]

logger = logging.getLogger(__name__)


# Create experiment object
class Experiment(DiskStorageMixin):
    """
    Creates an experiment object
    """
    def __init__(self, data:Data, model:Model, verbose : int = 1, is_fitted:bool = False, filename:str = None):
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
        self.model_type = self.model.model_type
        self.data = data
        __dh = hash(self.data)
        __dh = __dh if int(__dh) > 0 else -1 * int(__dh)
        __mh = hash(self.model)
        __mh = __mh if int(__mh) > 0 else -1 * int(__mh)
        self.filename =  f"{__dh}_{__mh}" if filename is None else filename
        self.verbose = verbose
        self.is_fitted = is_fitted
        self.predictions = None
        self.ground_truth = self.data.y_test
        self.time_dict = None
        self.params = dict()
        self.params['Model'] = dict(model)
        self.params['Data'] = dict(data)
        self.params['Experiment'] = {'name': self.filename, 'verbose': self.verbose, 'is_fitted': self.is_fitted, 'id': self.filename}
    
    def __eq__(self, other) -> bool:
        """
        Returns true if two experiments are equal
        """
        return self.__hash__() == other.__hash__()
    
    def __str__(self) -> str:
        """
        Returns human-readable string representation of Experiment object.
        """
        return str(self.params)
    
    def __repr__(self) -> str:
        """
        Returns reproducible string representation of Experiment object.
        """
        return "deckard.base.experiment.Experiment({})".format(dumps(self.params, sort_keys = True))

    def __iter__(self):
        """
        Returns an iterator over the experiment object
        """
        for key, value in self.params.items():
            yield key, value
            
    def __hash__(self) -> str:
        """
        Hashes the params as specified in the __init__ method.
        """
        return int(my_hash(str(self.__str__()).encode('utf-8')).hexdigest(), 32)
    
    def _build_model(self, **kwargs) -> None:
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
        return 
    
    
    def set_filename(self, filename:str = None) -> None:
        """
        Sets filename attribute.
        """
        if filename is None:
            self.filename = str(hash(self))
        else:
            self.filename = filename
        return None

    def __call__(self, path, prefix = None, filename = None, **kwargs) -> None:
        """
        Sets metric scorer. Builds model. Runs evaluation. Updates scores dictionary with results. Returns self with added scores, predictions, and time_dict attributes.
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        self._build_model(**kwargs)
        # TODO: Fix params
        # self.save_params(path = path, prefix = prefix)
        self.save_predictions(path = path, prefix = prefix)
        self.save_ground_truth(path = path, prefix = prefix)
        model_name = str(hash(self.model)) if filename is None else filename
        self.save_model(filename = model_name, path = path)
        # if hasattr(self.model, 'defence'):
        #     self.save_defence_params(path = path)
        
    ####################################################################################################################
    #                                                     DEFENSES                                                     #
    ####################################################################################################################
    def set_defence(self, defence:Union[object,str]) -> None:
        """
        Adds a defence to an experiment
        :param experiment: experiment to add defence to
        :param defence: defence to add
        """
         
        model = self.model.model
        if isinstance(defence, str):
            defence = self.get_defence(defence)
        if isinstance(defence, object):
            model = Model(model, defence = defence, model_type = self.model.model_type, path = self.model.path, url = self.model.url, is_fitted = self.is_fitted)
        else:
            raise ValueError("Defence must be a string or an object")
        self.model.defence = defence
        self.model.set_defence_params()
        if hasattr(self.model.defence, '_apply_fit') and self.model.defence._apply_fit != True:
            self.is_fitted = True
        self.params['Model'] = self.model.params
        self.params['Defence'] = self.params['Model']['Defence']
        del self.params['Model']['Defence']
        self.filename = str(hash(self))
        self.params['Defence']['experiment'] = self.filename
        return None

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
   
    def get_defence(self, filename:str=None, path:str = "."):
        """
        Returns the defence from an experiment
        :param experiment: experiment to get defence from
        """
        from deckard.base.parse import generate_tuple_from_yml, generate_object_from_tuple
        return generate_object_from_tuple(generate_tuple_from_yml(os.path.join(path, filename)))
        
    
    def save_defence_params(self, filename:str = "defence_params.json", path:str = ".") -> None:
        """
        Saves defence params to specified file.
        :param filename: str, name of file to save defence params to.
        :param path: str, path to folder to save defence params. If none specified, defence params are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        
        defence_file = os.path.join(path, filename)
        if 'Defence' in self.params:
            results = self.params['Defence']
        elif 'Defence' in self.model.params:
            results = self.model.params['Defence']
        else:
            raise ValueError("No defence params found in experiment")
        results = Series(results.values(), name =  self.filename, index = results.keys())
        results.to_json(defence_file)
        assert os.path.exists(defence_file), "Defence file not saved."
        return None
    
    
    


    
        

