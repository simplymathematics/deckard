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
from time import process_time_ns
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
from json import dumps
from pickle import dump
DEFENCE_TYPES = [Preprocessor, Trainer, Transformer, Postprocessor]
logger = logging.getLogger(__name__)

# Default scorers
REGRESSOR_SCORERS = {'MAPE': mean_absolute_percentage_error, "MSE" : mean_squared_error, 'MAE': mean_absolute_error,  "R2" : r2_score, "EXVAR" : explained_variance_score}
CLASSIFIER_SCORERS = {'F1' : f1_score, 'ACC' : accuracy_score, 'PREC' : precision_score, 'REC' : recall_score,'AUC': roc_auc_score}

# Create experiment object
class Experiment(object):
    """
    Creates an experiment object
    """
    def __init__(self, data:Data, model:Model, params:dict = {}, verbose : int = 1, scorers:dict = None, name:str = None, is_fitted:bool = False, filename = None):
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
        self.name = data.dataset +"_"+ model.name if name is None else name
        self.verbose = verbose
        self.is_fitted = is_fitted
        self.predictions = None
        self.time_dict = None
        self.scores = None
        self.filename = str(filename)
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
        


    def __hash__(self):
        """
        Returns a hash of the experiment object
        """
        return int(my_hash(dumps(self.params, sort_keys = True).encode('utf-8')).hexdigest(), 36)
    
    def __eq__(self, other) -> bool:
        """
        Returns true if two experiments are equal
        """
        return self.__hash__() == other.__hash__()
    
    def _build_model(self, **kwargs) -> None:
        """
        Builds model.
        """
        self.model.run_model(self.data, **kwargs)
        self.filename = str(int(my_hash(dumps(self.params, sort_keys = True).encode('utf-8')).hexdigest(), 36)) if self.filename is None else self.filename
        self.params['Experiment']['experiment'] = self.filename
        self.params['Model']['experiment'] = self.filename
        self.params['Data']['experiment'] = self.filename
        return None
    
    
    def _build_attack(self, targeted: bool = False, **kwargs) -> None:
        """
        Runs the attack on the model
        """
        assert hasattr(self, 'attack'), "Attack not set"
        start = process_time_ns()
        if targeted == False:
            adv_samples = self.attack.generate(self.data.X_test)
        else:
            adv_samples = self.attack.generate(self.data.X_test, self.data.y_test, **kwargs)
        end = process_time_ns()
        self.time_dict['adv_fit_time'] = end - start
        start = process_time_ns()
        adv = self.model.model.predict(adv_samples)
        end = process_time_ns()
        self.adv = adv
        self.adv_samples = adv_samples
        self.time_dict['adv_pred_time'] = end - start
        self.filename = str(int(my_hash(dumps(self.params, sort_keys = True).encode('utf-8')).hexdigest(), 36))
        self.params['Experiment']['experiment'] = self.filename
        self.params['Model']['experiment'] = self.filename
        self.params['Data']['experiment'] = self.filename
        return None
    

    

    def run(self, path, **kwargs) -> None:
        """
        Sets metric scorer. Builds model. Runs evaluation. Updates scores dictionary with results. Returns self with added scores, predictions, and time_dict attributes.
        """
        self.save_experiment_params(path = path)
        self._build_model(**kwargs)
        self.predictions = self.model.predictions
        self.time_dict = self.model.time_dict
        self.evaluate()
        self.save_results(path = path)

    def run_attack(self, path):
        """
        Runs attack.
        """
        assert hasattr(self, 'attack')
        self.save_attack_params(path = path)
        self._build_attack(**kwargs)
        self.evaluate_attack()
        self.save_attack_results(path = path)
        return None
        
    def set_attack(self, attack:object) -> None:
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
        self.filename = str(hash(self))
        self.params['Attack']['experiment'] = self.filename
        return None

    def set_defence(self, defence:object) -> None:
        """
        Adds a defence to an experiment
        :param experiment: experiment to add defence to
        :param defence: defence to add
        """
        model = self.model.model
        model = Model(model, defence = defence, model_type = self.model.model_type, path = self.model.path, url = self.model.url, is_fitted = self.is_fitted)
        self.model = model
        return None

    def insert_sklearn_preprocessor(self, name:str, preprocessor: object, position:int):
        """
        Add a sklearn preprocessor to the experiment.
        :param name: name of the preprocessor
        :param preprocessor: preprocessor to add
        :param position: position to add preprocessor
        """
        if isinstance(self.model.model, (BaseEstimator, TransformerMixin)) and not isinstance(self.model.model, Pipeline):
            self.model.model = Pipeline([('model', self.model.model)])
        elif not isinstance(self.model.model, Pipeline):
            raise ValueError("Model {} is not a sklearn compatible estimator".format(type(self.model.model)))
        new_model = deepcopy(self.model)
        new_model.model.steps.insert(position, (name, preprocessor))
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

    def get_defence(self):
        """
        Returns the defence from an experiment
        :param experiment: experiment to get defence from
        """
        return self.defence
        
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

    def save_data(self, filename:str = "data.pkl", path:str = ".") -> None:
        """
        Saves data to specified file.
        :param filename: str, name of file to save data to. 
        :param path: str, path to folder to save data to. If none specified, data is saved in current working directory. Must exist.
        """
        assert path is not None, "Path to save data must be specified."
        with open(os.path.join(path, filename), 'wb') as f:
            dump(self.data, f)
        assert os.path.exists(os.path.join(path, filename)), "Data not saved."
        return None
    
    def save_experiment_params(self, data_params_file:str = "data_params.json", model_params_file:str = "model_params.json", exp_params_file:str = "experiment_params.json", path:str = ".") -> None:
        """
        Saves data to specified file.
        :param data_params_file: str, name of file to save data parameters to.
        :param model_params_file: str, name of file to save model parameters to.
        :param path: str, path to folder to save data to. If none specified, data is saved in current working directory. Must exist.
        """
        assert path is not None, "Path to save data must be specified."
        if not os.path.isdir(path):
            os.mkdir(path)
        data_params = Series(self.params['Data'])
        model_params = Series(self.params['Model'])
        exp_params = Series(self.params['Experiment'])
        data_params.to_json(os.path.join(path, data_params_file),)
        model_params.to_json(os.path.join(path, model_params_file))
        exp_params.to_json(os.path.join(path, exp_params_file))
        assert os.path.exists(os.path.join(path, data_params_file)), "Data params not saved."
        assert os.path.exists(os.path.join(path, model_params_file)), "Model params not saved."
        assert os.path.exists(os.path.join(path, exp_params_file)), "Model params not saved."
        if 'Defence' in model_params:
            model_params['Defence']['experiment'] = str(hash(self))
            defence_params = Series(model_params['Defence'])
            defence_params.to_json(os.path.join(path, "defence_params.json"))
            assert os.path.exists(os.path.join(path, "defence_params.json")), "Defence params not saved."
        return None

    def save_model(self, filename:str = "model", path:str = ".") -> str:
        """
        Saves model to specified file (or subfolder).
        :param filename: str, name of file to save model to. 
        :param path: str, path to folder to save model. If none specified, model is saved in current working directory. Must exist.
        :return: str, path to saved model.
        """
        assert os.path.isdir(path), "Path {} to experiment does not exist".format(path)
        logger.info("Saving model to {}".format(os.path.join(path,filename)))
        self.model.save(filename = filename, path = path)
    
    def save_predictions(self, filename:str = "predictions.json", path:str = ".") -> None:
        """
        Saves predictions to specified file.
        :param filename: str, name of file to save predictions to. 
        :param path: str, path to folder to save predictions. If none specified, predictions are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"    
        prediction_file = os.path.join(path, filename)
        results = self.predictions
        results = Series(results)
        results.to_json(prediction_file)
        assert os.path.exists(prediction_file), "Prediction file not saved"
        return None

    def save_adv_predictions(self, filename:str = "adversarial_predictions.json", path:str = ".") -> None:
        """
        Saves adversarial predictions to specified file.
        :param filename: str, name of file to save adversarial predictions to.
        :param path: str, path to folder to save adversarial predictions. If none specified, predictions are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        adv_file = os.path.join(path, filename)
        adv_results = DataFrame(self.adv)
        adv_results.to_json(adv_file)
        assert os.path.exists(adv_file), "Adversarial example file not saved"
        return None

    def save_cv_scores(self, filename:str = "cv_scores.json", path:str = ".") -> None:
        """
        Saves crossvalidation scores to specified file.
        :param filename: str, name of file to save crossvalidation scores to.
        :param path: str, path to folder to save crossvalidation scores. If none specified, scores are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        assert filename is not None, "Filename must be specified"
        cv_file = os.path.join(path, filename)
        cv_results = Series(self.model.model.model.cv_results_, name = self.filename)
        cv_results.to_json(cv_file)
        assert os.path.exists(cv_file), "CV results file not saved"

    def save_scores(self, filename:str = "scores.json", path:str = ".") -> None:
        """
        Saves scores to specified file.
        :param filename: str, name of file to save scores to.
        :param path: str, path to folder to save scores. If none specified, scores are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        
        score_file = os.path.join(path, filename)
        results = self.scores
        results = Series(results.values(), name =  self.filename, index = results.keys())
        results.to_json(score_file)
        assert os.path.exists(score_file), "Score file not saved"
        return None
    
    def save_adv_scores(self, filename:str = "adversarial_scores.json", path:str = ".") -> None:
        """
        Saves adversarial scores to specified file.
        :param filename: str, name of file to save adversarial scores to.
        :param path: str, path to folder to save adversarial scores. If none specified, scores are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        
        adv_score_file = os.path.join(path, filename)
        results = self.adv_scores
        results = Series(results.values(), name =  self.filename, index = results.keys())
        results.to_json(adv_score_file)
        assert os.path.exists(adv_score_file), "Adversarial score file not saved."
        return None
    
    def save_adversarial_samples(self, filename:str = "adversarial_examples.json", path:str = "."):
        """
        Saves adversarial examples to specified file.
        :param filename: str, name of file to save adversarial examples to.
        :param path: str, path to folder to save adversarial examples. If none specified, examples are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        assert hasattr(self, "adv_samples"), "No adversarial samples to save"
        adv_file = os.path.join(path, filename)
        adv_results = DataFrame(self.adv_samples.reshape(self.adv_samples.shape[0], -1))
        adv_results.to_json(adv_file)
        assert os.path.exists(adv_file), "Adversarial example file not saved"
        return None

    def save_time_dict(self, filename:str = "time_dict.json", path:str = "."):
        """
        Saves time dictionary to specified file.
        :param filename: str, name of file to save time dictionary to.
        :param path: str, path to folder to save time dictionary. If none specified, time dictionary is saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        assert hasattr(self, "time_dict"), "No time dictionary to save"
        time_file = os.path.join(path, filename)
        time_results = Series(self.time_dict)
        time_results.to_json(time_file)
        assert os.path.exists(time_file), "Time dictionary file not saved"
        return None

    def save_attack_params(self, filename:str = "attack_params.json", path:str = ".") -> None:
        """
        Saves attack params to specified file.
        :param filename: str, name of file to save attack params to.
        :param path: str, path to folder to save attack params. If none specified, attack params are saved in current working directory. Must exist.
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        assert os.path.isdir(path), "Path to experiment does not exist"
        attack_file = os.path.join(path, filename)
        results = self.params['Attack']
        results = Series(results.values(), index = results.keys())
        results.to_json(attack_file)
        assert os.path.exists(attack_file), "Attack file not saved."
        return None

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

    def save_results(self, path:str = ".") -> None:
        """
        Saves all data to specified folder, using default filenames.
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        # self.save_model(path = path)
        self.save_scores(path = path)
        self.save_predictions(path = path)
        if hasattr(self.model.model, 'cv_results_'):
            self.save_cv_scores(path = path)
            self.save_adversarial_samples(path = path)
        if hasattr(self, 'time_dict'):
            self.save_time_dict(path = path)
        return None

    def save_attack_results(self, path:str = ".") -> None:
        """
        Saves all data to specified folder, using default filenames.
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        # self.save_model(path = path)
        self.save_scores(path = path)
        self.save_predictions(path = path)
        if hasattr(self, "adv_scores"):
            self.save_adv_scores(path = path)
        if hasattr(self, "adv"):
            self.save_adv_predictions(path = path)
        if hasattr(self, "adv_samples"):
            self.save_adversarial_samples(path = path)
        if hasattr(self, 'time_dict'):
            self.save_time_dict(path = path)
        return None
        

