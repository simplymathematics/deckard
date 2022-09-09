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
from .model import Model
from .data import Data
from .storage import DiskstorageMixin
from art.defences.postprocessor import Postprocessor
from art.defences.preprocessor import Preprocessor
from art.defences.trainer import Trainer
from art.defences.transformer import Transformer
DEFENCE_TYPES = [Preprocessor, Trainer, Transformer, Postprocessor]

logger = logging.getLogger(__name__)


# Create experiment object
class Experiment(DiskstorageMixin):
    """
    Creates an experiment object
    """
    def __init__(self, data:Data, model:Model, verbose : int = 1, name:str = None, is_fitted:bool = False, filename:str = None):
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
        self.name = str(hash(self.data.dataset)) +"_"+ str(hash(model)) if name is None else name
        self.verbose = verbose
        self.is_fitted = is_fitted
        self.predictions = None
        self.time_dict = None
        self.params = dict()
        self.params['Model'] = dict(model)
        self.params['Data'] = dict(data)
        if filename is None:
            self.filename = str(int(my_hash(dumps(self.params, sort_keys = True).encode('utf-8')).hexdigest(), 16))
        else:
            self.filename = filename
        self.params['Experiment'] = {'name': self.name, 'verbose': self.verbose, 'is_fitted': self.is_fitted, 'id': self.filename}


    def __hash__(self):
        """
        Returns a hash of the experiment object
        """
        return int(my_hash(dumps(self.params, sort_keys = True).encode('utf-8')).hexdigest(), 16)
    
    def __eq__(self, other) -> bool:
        """
        Returns true if two experiments are equal
        """
        return self.__hash__() == other.__hash__()
    
    def __str__(self) -> str:
        """
        Returns human-readable string representation of Experiment object.
        """
        return str({"Data": self.data, "Model": self.model, "Params": self.params})
    
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

    def run(self, path, filename = "scores.json", **kwargs) -> None:
        """
        Sets metric scorer. Builds model. Runs evaluation. Updates scores dictionary with results. Returns self with added scores, predictions, and time_dict attributes.
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        
        self._build_model(**kwargs)
        self.save_params(path = path)
        model_name = str(hash(self.model))
        self.model.save(filename = model_name, path = path)

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
        if isinstance(self.model.model.model, (BaseEstimator, TransformerMixin)) and not isinstance(self.model.model, Pipeline):
            self.model.model = Pipeline([('model', self.model.model.model)])
        elif not isinstance(self.model.model, Pipeline):
            raise ValueError("Model {} is not a sklearn compatible estimator".format(type(self.model.model)))
        new_model = deepcopy(self.model)
        try:
            new_model.model.model.steps.insert(position, (name, preprocessor))
        except AttributeError:
            new_model.model.steps.insert(position, (name, preprocessor))
        self.model = new_model
   
    def get_defence(self, filename:str=None, path:str = "."):
        """
        Returns the defence from an experiment
        :param experiment: experiment to get defence from
        """
        from deckard.base.parse import generate_object_list_from_tuple
        if filename is None:
            defence = self.defence
        else:
            location = os.path.join(path, filename)
            with open(location, 'rb') as f:
                defence_json = load(f)
            if defence_json['name'] is not None:
                name = defence_json['name'].split("'")[1]
                params = defence_json['params']
                new_params = {}
                for param, value in params.items():
                    if param.startswith("_"):
                        continue
                    else:
                        new_params[param] = value
                defence_tuple = (name, new_params)
                defence_list = [defence_tuple]
                defences = generate_object_list_from_tuple(defence_list)
                defence = defences[0]
            else:
                defence = None
        return defence
    
    def save_defence_params(self, filename:str = "defence_params.json", path:str = ".") -> None:
        """
        Saves defence params to specified file.
        :param filename: str, name of file to save defence params to.
        :param path: str, path to folder to save defence params. If none specified, defence params are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        
        defence_file = os.path.join(path, filename)
        results = self.params['Defence']
        results = Series(results.values(), name =  self.filename, index = results.keys())
        results.to_json(defence_file)
        assert os.path.exists(defence_file), "Defence file not saved."
        return None
    
    
    


    
        

