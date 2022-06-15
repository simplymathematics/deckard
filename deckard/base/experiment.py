import numpy as np
from sklearn import multiclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, \
    mean_squared_error, r2_score, mean_absolute_percentage_error, explained_variance_score
import logging
from sklearn.base import is_regressor
from sklearn.preprocessing import LabelBinarizer
from collections.abc import Callable
from hashlib import md5 as my_hash
from deckard.base.model import Model
from deckard.base.data import Data
from deckard.base.storage import DiskstorageMixin
from time import process_time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, balanced_accuracy_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.base import is_regressor
from pandas import Series, DataFrame
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from copy import deepcopy
from art.defences.postprocessor import Postprocessor
from art.defences.preprocessor import Preprocessor
from art.defences.trainer import Trainer
from art.defences.transformer import Transformer
from json import dumps, load
from pickle import dump
from typing import Union
DEFENCE_TYPES = [Preprocessor, Trainer, Transformer, Postprocessor]
logger = logging.getLogger(__name__)

# Default scorers
REGRESSOR_SCORERS = {'MAPE': mean_absolute_percentage_error, "MSE" : mean_squared_error, 'MAE': mean_absolute_error,  "R2" : r2_score, "EXVAR" : explained_variance_score}
CLASSIFIER_SCORERS = {'F1' : f1_score, 'ACC' : accuracy_score, 'PREC' : precision_score, 'REC' : recall_score,'AUC': roc_auc_score}

# Create experiment object
class Experiment(DiskstorageMixin):
    """
    Creates an experiment object
    """
    def __init__(self, data:Data, model:Model, params:dict = {}, verbose : int = 1, scorers:dict = None, name:str = None, is_fitted:bool = False, defence = None, filename:str = None):
        """
        Creates an experiment object
        :param data: Data object
        :param model: Model object
        :param params: Dictionary of other parameters you want to add to this object. Obviously everything in self.__dict__.keys() should be treated as a reserved keyword, however.
        :param verbose: Verbosity level
        :param scorers: Dictionary of scorers
        :param name: Name of experiment
        """
        # assert isinstance(model, Model)
        # assert isinstance(data, Data)
        # assert isinstance(scorers, dict) or scorers is None
        self.model = model
        self.model_type = self.model.model_type
        self.data = data
        self.name = self.data.dataset +"_"+ self.model.name if name is None else name
        self.verbose = verbose
        self.is_fitted = is_fitted
        self.predictions = None
        self.time_dict = None
        self.scores = None
        if not scorers:
            self.scorers = REGRESSOR_SCORERS if (is_regressor(model.model.model) or is_regressor(model.model)) else CLASSIFIER_SCORERS
        else:
            self.scorers = scorers
        self.refit = list(self.scorers.keys())[0]
        for key in params.keys():
            setattr(self, key, params[key])
        self.params = dict()
        self.params['Model'] = self.model.params
        self.params['Data'] = self.data.params
        self.params['Experiment'] = {'name': self.name, 'verbose': self.verbose, 'is_fitted': self.is_fitted, 'refit' : self.refit, "has_pred" : bool(self.predictions), "has_scores" : bool(self.scores), "has_time_dict" : bool(self.time_dict)}
        if filename is None:
            self.filename = str(int(my_hash(dumps(self.params, sort_keys = True).encode('utf-8')).hexdigest(), 16))
        else:
            self.filename = filename


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
    
    
    def _build_attack(self, targeted: bool = False, filename:str = None, **kwargs) -> None:
        """
        Runs the attack on the model
        """
        if not hasattr(self, 'time_dict') or self.time_dict is None:
            self.time_dict = dict()
        assert hasattr(self, 'attack'), "Attack not set"
        start = process_time()
        if targeted == False:
            adv_samples = self.attack.generate(self.data.X_test)
        else:
            adv_samples = self.attack.generate(self.data.X_test, self.data.y_test, **kwargs)
        end = process_time()
        self.time_dict['adv_fit_time'] = end - start
        start = process_time()
        adv = self.model.model.predict(adv_samples)
        end = process_time()
        self.adv = adv
        self.adv_samples = adv_samples
        self.time_dict['adv_pred_time'] = end - start
        return None

    def run(self, path, **kwargs) -> None:
        """
        Sets metric scorer. Builds model. Runs evaluation. Updates scores dictionary with results. Returns self with added scores, predictions, and time_dict attributes.
        """
        self.save_params(path = path)
        self._build_model(**kwargs)
        self.evaluate()
        self.save_results(path = path)
        self.save_params(path = path)

    def run_attack(self, path, **kwargs):
        """
        Runs attack.
        """
        assert hasattr(self, 'attack')
        if not os.path.isdir(path):
            os.mkdir(path)
        self.save_params(path = path)
        self.save_attack_params(path = path)
        self._build_attack(**kwargs)
        self.evaluate_attack()
        self.save_attack_results(path = path)
        self.save_params(path = path)
        return None
        
    def set_attack(self, attack:object, filename:str = None) -> None:
        """
        Adds an attack to an experiment
        :param experiment: experiment to add attack to
        :param attack: attack to add
        """
        attack_params = {}
        for key, value in attack.__dict__.items():
            if isinstance(value, int):
                attack_params[key] = value
            elif isinstance(value, float):
                attack_params[key] = value
            elif isinstance(value, str):
                attack_params[key] = value
            elif isinstance(value, Callable):
                attack_params[key] = str(type(value))
            else:
                attack_params[key] = str(type(value))
        assert isinstance(attack, object)
        self.params['Attack'] = {'name': str(type(attack)), 'params': attack_params}
        self.attack = attack
        if filename is None:
            self.filename = str(hash(self))
        else:
            self.filename = filename
        self.params['Attack']['experiment'] = self.filename
        return None

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
        new_model.model.model.steps.insert(position, (name, preprocessor))
        self.model = new_model
   
    def set_filename(self, filename:str = None) -> None:
        """
        Sets filename attribute.
        """
        if filename is None:
            self.filename = str(hash(self))
        else:
            self.filename = filename
        return None


    def get_attack(self):
        """
        Returns the attack from an experiment
        :param experiment: experiment to get attack from
        """
        return self.attack

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
        
    def get_scorers(self):
        """
        Sets the scorer for an experiment
        :param experiment: experiment to set scorer for
        :param scorer: scorer to set
        """
        return self.scorers

    def set_scorers(self, scorers:list) -> None:
        """
        Sets the scorer for an experiment
        :param experiment: experiment to set scorer for
        :param scorer: scorer to set
        """
        for scorer in scorers:
            assert isinstance(scorer, Callable), "Scorer must be callable"
        self.scorers = scorers
        return None


    def evaluate(self) -> None:
        """
        Sets scorers for evalauation if specified, returns a dict of scores in general.
        """
        assert hasattr(self, "predictions"), "Model needs to be built before evaluation. Use the .run method."
        scores = {}
        if len(self.data.y_test.shape) > 1:
            self.data.y_test = np.argmax(self.data.y_test, axis=1)
        if len(self.predictions.shape) > 1:
            self.predictions = np.argmax(self.predictions, axis=1)
        for scorer in self.scorers:
            try:
                scores[scorer] = self.scorers[scorer](self.data.y_test, self.predictions)
            except ValueError as e:
                if "average=" in str(e):
                    scores[scorer] = self.scorers[scorer](self.data.y_test, self.predictions, average='weighted')
                elif 'multi_class must be in' in str(e):
                    y_test = LabelBinarizer().fit(self.data.y_train).transform(self.data.y_test)
                    predictions = LabelBinarizer().fit(self.data.y_train).transform(self.predictions)
                    scores[scorer] = self.scorers[scorer](y_test, predictions, multi_class='ovr')
                else:
                    raise e
            except AxisError as e:
                y_test = LabelBinarizer().fit(self.data.y_train).transform(self.data.y_test)
                predictions = LabelBinarizer().fit(self.data.y_train).transform(self.predictions)
                scores[scorer] = self.scorers[scorer](y_test, predictions, multi_class='ovr')
        self.scores = scores
        return None

    def evaluate_attack(self) -> None:
        """
        Sets scorers for evalauation if specified, returns a dict of scores in general.
        """
        assert hasattr(self, "adv"), "Attack needs to be built before evaluation. Use the .run_attack method."
        if len(self.data.y_test.shape) > 1:
            self.data.y_test = np.argmax(self.data.y_test, axis=1)
        if len(self.adv.shape) > 1:
            self.adv = np.argmax(self.adv, axis=1)
        scores = {}
        for scorer in self.scorers:
            try:
                scores[scorer] = self.scorers[scorer](self.data.y_test, self.adv)
            except ValueError as e:
                if "average=" in str(e):
                    scores[scorer] = self.scorers[scorer](self.data.y_test, self.adv, average='weighted')
                elif "multi_class must be in" in str(e):
                    y_test = LabelBinarizer().fit(self.data.y_train).transform(self.data.y_test)
                    adv = LabelBinarizer().fit(self.data.y_train).transform(self.adv)
                    scores[scorer] = self.scorers[scorer](y_test, adv, multi_class='ovr')
                else:
                    raise e
            except AxisError as e:
                y_test = LabelBinarizer().fit(self.data.y_train).transform(self.data.y_test)
                adv = LabelBinarizer().fit(self.data.y_train).transform(self.adv)
                scores[scorer] = self.scorers[scorer](y_test, adv, multi_class='ovr')
        self.adv_scores = scores
        return None

    
        

