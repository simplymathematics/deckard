from ftplib import all_errors
from tkinter import E
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
from os import path, mkdir
logger = logging.getLogger(__name__)
from art.defences.postprocessor import Postprocessor
from art.defences.preprocessor import Preprocessor
from art.defences.trainer import Trainer
from art.defences.transformer import Transformer
from json import dumps
from pickle import dump
DEFENCE_TYPES = [Preprocessor, Trainer, Transformer, Postprocessor]


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
        self.params = {**self.data.params, **self.model.params, **params}
        self.name = data.dataset +"_"+ model.name if name is None else name
        self.verbose = verbose
        self.is_fitted = is_fitted
        self.predictions = None
        self.time_dict = None
        self.attack = None
        self.defense = None
        self.scores = None
        if not scorers:
            self.scorers = REGRESSOR_SCORERS if is_regressor(model.model) else CLASSIFIER_SCORERS
        else:
            self.scorers = scorers
        self.refit = list(self.scorers.keys())[0]
        for key in params.keys():
            assert key not in self.__dict__.keys(), "Keyword-- {} --is reserved".format(key)
            setattr(self, key, params[key])
        self.filename = str(my_hash(dumps(str(self.params)).encode('utf-8')).hexdigest()) if filename is None else filename


    def __hash__(self):
        """
        Returns a hash of the experiment object
        """
        
        new_string = str(self.params)
        return int(my_hash(dumps(new_string).encode('utf-8')).hexdigest(), 36)
    
    def __eq__(self, other) -> bool:
        """
        Returns true if two experiments are equal
        """
        return self.__hash__() == other.__hash__()
    
    def _build_supervised_model(self) -> tuple:
        """
        Builds a supervised model. Returns predictions as an np.ndarray and time taken to fit and predict as dual-value tuple. 
        """
        # assert self.is_supervised()
        if hasattr(self.model, 'fit_flag') or self.is_fitted == True or hasattr(self, 'y_pred'):
            logger.info("Model is already fitted")
            self.is_fitted = True
            start = process_time_ns()
            y_pred = self.model.model.predict(self.data.X_test)
            end = process_time_ns()
            fit_time = np.nan
            pred_time = end - start
        else:
            logger.info("Fitting model")
            logger.info("X_train shape: {}".format(self.data.X_train.shape))
            logger.info("y_train shape: {}".format(self.data.y_train.shape))
            start = process_time_ns()
            if  hasattr(self.model.model, 'fit'):
                start = process_time_ns()
                self.model.model.fit(self.data.X_train, self.data.y_train)
                end = process_time_ns()
            elif hasattr(self.model.model.model, 'fit'):
                start = process_time_ns()
                self.model.model.model.fit(self.data.X_train, self.data.y_train)
                end = process_time_ns()
            else:
                raise TypeError("Model has no fit method")
            self.is_fitted = True
            fit_time = end - start
            logger.info("Model training complete.")
            start = process_time_ns()
            y_pred = self.model.model.predict(self.data.X_test)
            end = process_time_ns()
            pred_time = end - start
        logger.info("Length of predictions: {}".format(len(y_pred)))
        logger.info("Made predictions")
        y_pred = np.array(y_pred)
        return y_pred, (fit_time, pred_time)

    def _build_unsupervised_model(self) -> tuple:
        """
        Builds unsupervised model. Returns predictions as an np.ndarray and time taken to fit/predict as single-value tuple.
        """
        assert not self.is_supervised()
        if hasattr(self.model, 'fit_flag') or self.is_fitted == True or hasattr(self, 'y_pred'):
            logger.warning("Model is already fitted")
            start = process_time_ns()
            y_pred = self.model.model.predict(self.data.y_test)
            end = process_time_ns()
            fit_pred_time = end - start
            self.is_fitted = True
            assert self.data.y_pred.shape == self.data.y_test.shape, "model appears to be fitted, but something went wrong."
        else:
            logger.info("Fitting and predicting model")
            logger.info("X_train shape: {}".format(self.data.X_train.shape))
            logger.info("y_train shape: {}".format(self.data.y_train.shape))
            start = process_time_ns()
            self.model.model.fit_predict(X = self.data.X_train)
            end = process_time_ns()
            fit_pred_time = end - start
            y_pred = self.model.model.predict(self.data.X_test)
            logger.info("Made predictions")
        y_pred = np.array(y_pred)
        return y_pred, fit_pred_time
    
    
    def _build_model(self) -> None:
        """
        Builds model and returns self with added time_dict and predictions attributes.
        """
        logger.debug("Model type: {}".format(type(self.model.model)))
        if self.is_supervised() == False:
            self.predictions, time = self._build_unsupervised_model()
            self.time_dict = {'fit_pred_time': time}
        elif self.is_supervised() == True:
            self.predictions, time = self._build_supervised_model()
            self.time_dict = {'fit_time': time[0], 'pred_time': time[1]}
        else:
            type_string = str(type(self.model.model))
            raise ValueError(f"Model, {type_string}, is not a supported estimator")
        return None
    
    def _build_attack(self, targeted: bool = False) -> None:
        """
        Runs the attack on the model
        """
        assert hasattr(self, 'attack'), "Attack not set"
        start = process_time_ns()
        if targeted == False:
            adv_samples = self.attack.generate(self.data.X_test)
        else:
            adv_samples = self.attack.generate(self.data.X_test, self.data.y_test)
        end = process_time_ns()
        self.time_dict['adv_fit_time'] = end - start
        start = process_time_ns()
        adv = self.model.model.predict(adv_samples)
        self.adv = adv
        end = process_time_ns()
        self.time_dict['adv_pred_time'] = end - start
        return None
    

    def is_supervised(self)-> bool:
        """
        Returns true if supervised, false if unsupervised. 
        """
        if hasattr(self.model.model, 'fit_predict'):
            result = False
            logger.info("Model is unsupervised")
        elif hasattr(self.model.model, 'fit') and not hasattr(self.model.model, 'fit_predict'):
            result = True
            logger.info("Model is supervised")
        else:
            raise ValueError("Model is not regressor or classifier")
        return result

    

    def run(self) -> None:
        """
        Sets metric scorer. Builds model. Runs evaluation. Updates scores dictionary with results. Returns self with added scores, predictions, and time_dict attributes.
        """
        self._build_model()
        self.evaluate()

    def run_attack(self):
        """
        Runs attack.
        """
        assert hasattr(self, 'attack')
        self._build_attack()
        self.evaluate_attack()

        

    def set_filename(self, filename: str) -> None:
        """
        Sets filename attribute.
        """
        self.filename = filename
        return None
    def set_attack(self, attack:object) -> None:
        """
        Adds an attack to an experiment
        :param experiment: experiment to add attack to
        :param attack: attack to add
        """
        attack_params = {}
        for key, value in attack.__dict__.items():
            if not key.startswith('_'):
                attack_params[key] = value
            else:
                attack_params[key] = str(type(value))
        assert isinstance(attack, object)
        self.params['Attack'] = {'name': str(type(attack)), 'params': attack_params}
        self.attack = attack
        self.filename = str(hash(self))
        return None
    def set_filename(self, filename:str = None) -> None:
        """
        Sets filename attribute.
        """
        if filename is None:
            self.filename = str(hash(self))
        else:
            self.filename = filename
        return None
    def set_defense(self, defense:object) -> None:
        """
        Adds a defense to an experiment
        :param experiment: experiment to add defense to
        :param defense: defense to add
        """
        def_params = {}
        assert isinstance(defense, object)
        for key, value in defense.__dict__.items():
            if not key.startswith('_'):
                def_params[key] = value
            else:
                def_params[key] = str(type(value))
        self.params['Defense'] = {'name': str(type(defense)), 'params': def_params}
        self.defense = defense
        if 'preprocessor' in str(type(defense)):
            self.model.model.preprocessing_defences = [defense]
        elif 'postprocessor' in str(type(defense)):
            self.model.model.postprocessing_defences = [defense]
        elif 'transformer' in str(type(defense)):
            logging.error("Transformer defense not yet supported")
            raise NotImplementedError
        elif 'trainer' in str(type(defense)):
            logging.error("Trainer defense not yet supported")
            raise NotImplementedError
        elif 'detector' in str(type(defense)):
            logging.error("Detector defense not yet supported")
            raise NotImplementedError
        self.filename = str(hash(self))
        return None


    def get_attack(self):
        """
        Returns the attack from an experiment
        :param experiment: experiment to get attack from
        """
        return self.attack

    def get_defense(self):
        """
        Returns the defense from an experiment
        :param experiment: experiment to get defense from
        """
        return self.defense
        



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
        self.adv_scores = scores
        return None

    def save_data(self, filename:str = "data.pkl", path:str = ".") -> None:
        """
        Saves data to specified file.
        :param filename: str, name of file to save data to.
        :param path: str, path to folder to save data to. If none specified, data is saved in current working directory.
        """
        assert path is not None, "Path to save data must be specified."
        with open(path.join(path, filename), 'wb') as f:
            dump(self.data, f)
        assert path.exists(path.join(path, filename)), "Data not saved."
        return None

    def save_model(self, filename:str = None, path:str = None) -> None:
        assert path.isdir(path), "Path to experiment does not exist"
        assert filename is not None, "Filename must be specified."
        model_file = path.join(path, filename)
        if 'art' in type(self.model.model):
            self.model.model.save(filename = model_file)
        elif 'sklearn' in type(self.model.model):
            with open(model_file, 'wb') as f:
                dump(self.model.model, f)
        else: 
            raise IOError("Model not saved. Must be 'art' or 'sklearn'")
        assert path.exists(model_file), "Model file not saved"
        return None


    def save_adv_examples(self, filename:str = None, path:str = None) -> None:
        assert path.isdir(path), "Path to experiment does not exist"
        assert filename is not None, "Filename must be specified"
        adv_file = path.join(path, filename)
        adv_results = DataFrame(self.adv)
        adv_results.to_json(adv_file)
        assert path.exists(adv_file), "Adversarial example file not saved"
        return None
    
    def save_cv_scores(self, filename:str = None, path:str = None) -> None:
        assert path.isdir(path), "Path to experiment does not exist"
        assert filename is not None, "Filename must be specified"
        cv_file = path.join(path, filename)
        cv_results = Series(self.model.model.cv_results_, name = self.filename)
        cv_results.to_json(cv_file)
        assert path.exists(cv_file), "CV results file not saved"

    def save_model(self, filename:str = None, path:str = None) -> None:
        assert path.isdir(path), "Path to experiment does not exist"
        assert filename is not None, "Filename must be specified."
        model_file = path.join(path, filename)
        if 'art' in type(self.model.model):
            self.model.model.save(filename = model_file)
        elif 'sklearn' in type(self.model.model):
            with open(model_file, 'wb') as f:
                dump(self.model.model, f)
        else: 
            raise IOError("Model not saved. Must be 'art' or 'sklearn'")
        assert path.exists(model_file), "Model file not saved"
        return None
        

    def save_predictions(self, filename:str = None, path:str = None) -> None:
        assert path.isdir(path), "Path to experiment does not exist"
        assert filename is not None, "Filename must be specified."
        prediction_file = path.join(path, filename)
        results = self.predictions
        results = Series(results.values(), name =  self.filename, index = results.keys())
        results.to_json(prediction_file)
        assert path.exists(prediction_file), "Prediction file not saved"
        return None

    def save_scores(self, filename:str = None, path:str = ".") -> None:
        assert path.isdir(path), "Path to experiment does not exist"
        assert filename is not None, "Filename must be specified."
        score_file = path.join(path, filename)
        results = self.scores
        results = Series(results.values(), name =  self.filename, index = results.keys())
        results.to_json(score_file)
        assert path.exists(score_file), "Score file not saved"
        return None
    
    def save_adv_scores(self, filename:str = None, path:str = ".") -> None:
        assert path.isdir(path), "Path to experiment does not exist"
        assert filename is not None, "Filename must be specified."
        adv_score_file = path.join(path, filename)
        results = self.adv_scores
        results = Series(results.values(), name =  self.filename, index = results.keys())
        results.to_json(adv_score_file)
        assert path.exists(adv_score_file), "Adversarial score file not saved."
        return None
    
    def save_attack(self, filename:str = None, path:str = None) -> None:
        assert path.isdir(path), "Path to experiment does not exist"
        assert filename is not None, "Filename must be specified."
        attack_file = path.join(path, filename)
        results = self.params['Attack']
        results = Series(results.values(), name =  self.filename, index = results.keys())
        results.to_json(attack_file)
        assert path.exists(attack_file), "Attack file not saved."
        return None

    def save_defense(self, filename:str = None, path:str = None) -> None:
        assert path.isdir(path), "Path to experiment does not exist"
        assert filename is not None, "Filename must be specified."
        defense_file = path.join(path, filename)
        results = self.params['Defense']
        results = Series(results.values(), name =  self.filename, index = results.keys())
        results.to_json(defense_file)
        assert path.exists(defense_file), "Defense file not saved."
        return None

    def save(self, filename:str = None, path:str = None) -> None:
        self.save_model(filename = filename, path = path)
        self.save_scores(filename = filename, path = path)
        self.save_predictions(filename = filename, path = path)
        if isinstance(self.attack, object):
            self.save_attack(filename = filename, path = path)
        if isinstance(self.defense, object):
            self.save_defense(filename = filename, path = path)
        if hasattr(self, "adv"):
            self.save_adv_scores(filename = filename, path = path)
        if hasattr(self.model.model, 'cv_results_'):
            self.save_cv_scores(filename = filename, path = path)
        if hasattr(self, "adv"):
            self.save_adv_examples(filename = filename, path = path)
        if hasattr(self, "indices"):
            self.save_indices(filename = filename, path = path)
        return None
        

