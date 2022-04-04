from ftplib import all_errors
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, \
    mean_squared_error, r2_score, mean_absolute_percentage_error, explained_variance_score
import logging
from sklearn.base import is_regressor
import uuid
from collections.abc import Callable
from torch import get_file_path
from hashlib import md5 as my_hash
from deckard.base.model import Model
from deckard.base.data import Data, validate_data
from time import process_time_ns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, balanced_accuracy_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.base import is_regressor
from pandas import Series, DataFrame
from os import path, mkdir
from pickle import dump
logger = logging.getLogger(__name__)
from art.defences.postprocessor import Postprocessor
from art.defences.preprocessor import Preprocessor
from art.defences.trainer import Trainer
from art.defences.transformer import Transformer
from json import dump
DEFENCE_TYPES = [Preprocessor, Trainer, Transformer, Postprocessor]

# Create experiment object


class Experiment(object):
    """
    Creates an experiment object
    """
    def __init__(self, data:Data, model:Model, params:dict = {}, verbose : int = 1, scorers:dict = None, refit = None, name:str = None, is_fitted:bool = False, filename = None):
        """
        Creates an experiment object
        :param data: Data object
        :param model: Model object
        :param params: Dictionary of other parameters you want to add to this object. Obviously everything in self.__dict__.keys() should be treated as a reserved keyword, however.
        :param verbose: Verbosity level
        :param scorers: Dictionary of scorers
        :param refit: Name of scorer to refit. If none specified, first scorer is used.
        :param name: Name of experiment
        """
        assert isinstance(model, Model)
        assert isinstance(data, Data)
        assert isinstance(scorers, dict) or scorers is None
        self.time_series = data.time_series
        self.model = model
        self.data = data
        if name is None:
            self.name = data.dataset +"_"+ model.name
        else:
            self.name = name
        for key in params.keys():
            assert key not in self.__dict__.keys(), "Keyword-- {} --is reserved".format(key)
            setattr(self, key, params[key])
        self.verbose = verbose
        self.refit = refit
        self.params = {"Model params" : model.params, "Data params" : data.params, "Experiment params" : params}
        if filename is None:
            self.filename = str(uuid.uuid4().hex)
        else:
            self.filename = filename
        if scorers is not None:
            assert isinstance(scorers, dict)
            assert [isinstance(score_name, str) for score_name in scorers.keys()]
            assert [isinstance(scorer, callable) for scorer in scorers.values()]
            self.scorers = scorers
        else:
           self = self.set_metric_scorer()
        self.is_fitted = is_fitted
        self.predictions = None
        self.time_dict = None
        self.score_dict = None
        self.attack = None
        self.defense = None


    def __hash__(self):
        """
        Returns a hash of the experiment object
        """
        from json import dumps
        new_string = str(self.params)
        return int(my_hash(dumps(new_string).encode('utf-8')).hexdigest(), 36)
    
    def __eq__(self, other) -> bool:
        """
        Returns true if two experiments are equal
        """
        return self.__hash__() == other.__hash__()
    
    def _build_supervised_model(self) -> dict:
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
            self.model.model.fit(self.data.X_train, self.data.y_train)
            end = process_time_ns()
            self.is_fitted = True
            fit_time = end - start
            logger.info("Model training complete.")
            start = process_time_ns()
            y_pred = self.model.model.predict(self.data.X_test)
            end = process_time_ns()
            pred_time = end - start
        logger.info("Length of predictions: {}".format(len(y_pred)))
        logger.info("Made predictions")
        return y_pred, (fit_time, pred_time)

    def _build_unsupervised_model(self) -> dict:
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
        return y_pred (fit_pred_time)
    
    def _build_time_series_model(self) -> dict:
        """
        Builds time series model. Returns predictions as an np.ndarray and time taken to fit and predict as -value tuple.
        """
        from sktime.forecasting.base import ForecastingHorizon
        fh = ForecastingHorizon(self.data.y_test.index, is_relative=False)
        forecaster = self.model.model        
        if hasattr( self.model, 'fit_flag') or self.is_fitted == True or hasattr(self, 'y_pred'):
            logger.info("Model is already fitted")
            self.is_fitted = True
            start = process_time_ns()
            y_pred = forecaster.predict(fh = fh)
            end = process_time_ns()
            fit_time = np.nan
            pred_time = end - start
        else:
            logger.info("Fitting model")
            logger.info("X_train shape: {}".format(self.data.X_train.shape))
            logger.info("y_train shape: {}".format(self.data.y_train.shape))
            start = process_time_ns()
            forecaster.fit(y = self.data.y_train, X = self.data.X_train, fh = fh)
            end = process_time_ns()
            self.is_fitted = True
            fit_time = end - start
            logger.info("Model training complete.")
            start = process_time_ns()
            y_pred = forecaster.predict(fh = fh)
            end = process_time_ns()
            pred_time = end - start
        logger.info("Length of predictions: {}".format(len(y_pred)))
        logger.info("Made predictions")
        return y_pred, (fit_time, pred_time)
    
    
    def build_model(self) -> dict:
        """
        Builds model and returns self with added time_dict and predictions attributes.
        """
        logger.debug("Model type: {}".format(type(self.model.model)))
        if self.is_supervised() == False and self.time_series == False:
            self.predictions, time = self._build_unsupervised_model()
            self.time_dict = {'fit_pred_time': time[0]}
        elif self.is_supervised() == True and self.time_series == False:
            self.predictions, time = self._build_supervised_model()
            self.time_dict = {'fit_time': time[0], 'pred_time': time[1]}
        elif self.time_series == True:
            # TODO: fix time-series pipeline compatibility, ensure that time_series data != time_series model
            self.predictions, time = self._build_supervised_model()
            self.time_dict = {'fit_time': time[0], 'pred_time': time[1]}
        else:
            type_string = str(type(self.model.model))
            raise ValueError(f"Model, {type_string}, is not a supported estimator")
        return self
    
    

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

    

    def run(self) -> dict:
        """
        Sets metric scorer. Builds model. Runs evaluation. Updates scores dictionary with results. Returns self with added scores, predictions, and time_dict attributes.
        """
        self.set_metric_scorer()
        self.build_model()
        scores = self.evaluate()
        if self.attack is not None:
            self = self.run_attack()
            self.adv_scores = self.evaluate_attack()
        scores.update(self.time_dict)
        scores.update({'Name': self.name})
        scores.update({'id_': self.filename})
        self.scores = scores

    def set_filename(self, filename: str) -> None:
        """
        Sets filename attribute.
        """
        self.filename = filename

    def set_attack(self, attack:object):
        """
        Adds an attack to an experiment
        :param experiment: experiment to add attack to
        :param attack: attack to add
        """
        assert isinstance(attack, object)
        self.params['Attack'] = {'name': str(type(attack)), 'params': attack.__dict__}
        self.attack = attack
        self.filename = str(hash(self))
        return self

    def set_defense(self, defense:object):
        """
        Adds a defense to an experiment
        :param experiment: experiment to add defense to
        :param defense: defense to add
        """
        assert isinstance(defense, object)
        self.params['Defense'] = {'name': str(type(defense)), 'params': defense.__dict__}
        self.defense = defense
        self.filename = str(hash(self))
        return self

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

    def run_attack(self):
        """
        Runs the attack on the model
        """
        assert hasattr(self, 'attack'), "Attack not set"
        adv_pred = self.attack.generate(self.data.X_test)
        self.adv = self.model.model.predict(adv_pred)
        return self


    def set_metric_scorer(self) -> dict:
        """
        Sets metric scorer from dictionary passed during initialization. If no scorers are passed, default is used. If no refit is specified, first scorer is used.
        """
        if not hasattr(self, 'scorers'):
            if is_regressor(self.model.model) == True or 'sktime' in str(type(self.model.model)):
                logger.info("Model is regressor.")
                new_scorers = {'MAPE': mean_absolute_percentage_error, "MSE" : mean_squared_error, 'MAE': mean_absolute_error,  "R2" : r2_score, "EXVAR" : explained_variance_score}
            elif is_regressor(self.model.model) == False:
                logger.info("Model is classifier.")
                self.data.y_test = self.data.y_test.astype(int)
                new_scorers = {'F1' : f1_score, 'BALACC' : balanced_accuracy_score, 'ACC' : accuracy_score, 'PREC' : precision_score, 'REC' : recall_score,'AUC': roc_auc_score}
            else:
                raise ValueError("Model is not estimator")
        elif len(list(self.scorers)) == 1: # If there's only one scorer, we verify and use it
            assert isinstance(list(self.scorers)[0], str)
            assert isinstance(list(self.values)[0], callable)
            new_scorers = self.scorers
        else:
            new_scorers = {}
            for score_name, scorer in self.scorers.items():
                assert isinstance(score_name, str)
                new_scorers[score_name] = scorer
        if self.refit is not None:
            assert self.refit in new_scorers.keys()
        else:
            self.refit = list(new_scorers.keys())[0]
        self.params.update({'Optimization Scorer' : self.refit})
        self.scorers = new_scorers
        return self

    def evaluate(self, scorers:dict = None) -> dict:
        """
        Sets scorers for evalauation if specified, returns a dict of scores in general.
        """
        if scorers is None:
            self.set_metric_scorer()
        scores = {}
        for scorer in self.scorers:
            logger.info("Scoring with {}".format(scorer))
            if scorer in ['F1', 'REC', 'PREC']:
                average = 'weighted'
                if not len(self.predictions.shape) == 1:
                    predictions = np.argmax(self.predictions, axis=1)
                    y_test = np.argmax(self.data.y_test, axis=1)
                scores[scorer] = self.scorers[scorer](y_test.astype(int), predictions, average = average)
            elif scorer in ['BALACC', 'ACC']:
                if not len(self.predictions.shape) == 1:
                    predictions = np.argmax(self.predictions, axis=1)
                    y_test = np.argmax(self.data.y_test, axis=1)
                scores[scorer] = self.scorers[scorer](y_test, predictions)
            elif scorer in ['AUC', 'ROC']:
                scores[scorer] = self.scorers[scorer](self.data.y_test, self.predictions, multi_class = 'ovr')
            else:
                try:
                    scores[scorer] = self.scorers[scorer](self.data.y_test, self.predictions)
                except ValueError as e:
                    logger.error(e)
                    logger.error("Scorer {} unable to score".format(scorer))
                    scores[scorer] = np.nan
                else:
                    scores[scorer] = np.nan
        return scores

    def evaluate_attack(self, scorers:dict = None) -> dict:
        """
        Sets scorers for evalauation if specified, returns a dict of scores in general.
        """
        assert hasattr(self, 'adv'), "Attack not run"
        if scorers is None:
            self.set_metric_scorer()
        scores = {}
        for scorer in self.scorers:
            logger.info("Scoring with {}".format(scorer))
            if scorer in ['F1', 'REC', 'PREC']:
                average = 'weighted'
                if not len(self.adv.shape) == 1:
                    adv = np.argmax(self.adv, axis=1)
                if not len(self.data.y_test.shape) == 1:
                    y_test = np.argmax(self.data.y_test, axis=1)
                scores[scorer] = self.scorers[scorer](y_test.astype(int), adv, average = average)
            elif scorer in ['BALACC', 'ACC']:
                if not len(self.adv.shape) == 1:
                    adv = np.argmax(self.adv, axis=1)
                if not len(self.data.y_test.shape) == 1:
                    y_test = np.argmax(self.data.y_test, axis=1)
                scores[scorer] = self.scorers[scorer](y_test, adv)
            elif scorer in ['AUC', 'ROC']:
                scores[scorer] = self.scorers[scorer](self.data.y_test, self.adv, multi_class = 'ovr')
            else:
                try:
                    scores[scorer] = self.scorers[scorer](self.data.y_test, self.adv)
                except ValueError as e:
                    logger.error(e)
                    logger.error("Scorer {} unable to score".format(scorer))
                    scores[scorer] = np.nan
                else:
                    scores[scorer] = np.nan
        return scores

    def save_results(self, score_file:str = "results.json", data_file:str = "data_params.json", filename:str = "model_params.json", predictions_file:str = "predictions.json", indices_file = "indices.json", folder:str = ".") -> None:
        """
        Saves results as json files in specified folder.
        :param score_file: name of score file
        :param data_file: name of data file
        :param filename: name of model file
        :param folder: str, path to folder to save results in. Defaults to "." (current working directory).
        """
        if not path.isdir(folder):
            mkdir(folder)
            # set permissions to read/write for all
            # TODO: fix permissions
            from os import chmod
            chmod(folder, 0o770)
        logger.debug("Saving results")
        adv_score_file = "adv_" + score_file
        score_file = path.join(folder, score_file)
        data_file = path.join(folder, data_file)
        filename = path.join(folder, filename)
        adv_score_file = path.join(folder, adv_score_file)
        if hasattr(self, 'scores'):
            results = self.scores
            results = Series(results.values(), name =  self.filename, index = results.keys())
            results.to_json(score_file)
        if hasattr(self, 'adv_scores'):
            results = self.adv_scores
            results = Series(results.values(), name =  self.filename, index = results.keys())
            adv_score_file = adv_score_file.replace(".json", "_adv.json")
            results.to_json(adv_score_file)
        if hasattr(self.model, 'params') and isinstance(self.model.params, DataFrame):
            self.model.params.to_json(filename)
        elif hasattr(self.model, 'params') and isinstance(self.model.params, dict):
            new_dict = {}
            for key, value in self.model.params.items():
                new_dict[str(key)] = str(value)
            with open(filename, 'w') as f:
                dump(new_dict, f)
        else:
            raise IOError("Model params not saved")
        if "Attack" in self.params:
            attack_file = path.join(folder, "attack_params.json")
            new_dict = {}
            for key, value in self.params.items():
                new_dict[str(key)] = str(value)
            with open(attack_file, 'w') as f:
                dump(new_dict, f, indent=4)
        if "Defense" in self.params:  
            defense_file = path.join(folder, "defense_params.json")
            new_dict = {}
            for key, value in self.params['Defense'].items():
                new_dict[str(key)] = str(value)
            with open(defense_file, 'w') as f:
                dump(new_dict, f, indent=4)
        if hasattr(self.model.model, "cv_results_"):
            cv_file = path.join(folder, f"{self.filename}_cv_results.json")
            cv_results = Series(self.model.model.cv_results_, name = self.filename)
            cv_results.to_json(cv_file)
        if hasattr(self, "predictions"):
            predictions_file = path.join(folder, "predictions.json")
            try:
                predictions = Series(self.predictions, name = self.filename)
            except:
                predictions = DataFrame(self.predictions)
                predictions['id_'] = self.filename
            predictions.to_json(predictions_file)
        if hasattr(self, "adv"):
            adv_file = path.join(folder, "predictions_adv.json")
            try:
                adv = Series(self.adv, name = self.filename)
            except:
                adv = DataFrame(self.adv)
                adv['id_'] = self.filename
            adv.to_json(adv_file)
        if hasattr(self, "indices"):
            indices_file = path.join(folder, "indices.json")
            indices = Series(self.indices, name = self.filename)
            indices.to_json(indices_file)
        logger.info("Results:{}".format(results))
        logger.info("Data Params: {}".format(self.data.params))
        logger.debug("Model Params: {}".format(self.model.params))
        results.to_json(score_file)
        logger.info("Saved results.")
        return None

    def save(self, filename:str = None, path:str = None, save_model:bool=True, save_results:bool=True) -> None:
        """
        :param filename: str, name of file to save model to. If none specified, model is not saved.
        :param path: str, path to folder to save model to. If none specified, model is saved in current working directory.
        """
        assert path is not None, "Path to save experiment must be specified."
        if filename is None:
            filename = self.filename
        if save_model:
            self.model.model.save(filename = filename, path = path)
        if save_results:
            self.save_results(folder = path)
    # def save_model(self, model_name:str = "model",  folder:str = ".") -> None:
    #     """
    #     Saves experiment as specified name to specified folder.
    #     :param moel_name: str, name of file to save experiment as. Defaults to "best.model".
    #     :move_from: str, path to move experiment from. If none specified, saves in current working directory using pickle.
    #     :param folder: str, path to folder to save results in. Defaults to "." (current working directory).
    #     :param control: bool, whether to treat the model as a control model or as an instanced model with a uuid.
    #     """
    #     if not path.isdir(folder):
    #         mkdir(folder)
    #     filename = path.join(folder, model_name)
    #     logger.info("Model file: {}".format(filename))
    #     if self.model.model_type == None:
    #         try:
    #             dump(self.model, open(filename+".pkl", 'wb'))
    #         except TypeError as e:
    #             logger.warning("Model not saved. Check that model is pickleable. This is not the case for TF1 models. Use model_type = 'tf1' in that case.")
    #             raise e
    #     elif self.model.model_type == 'tf1' or 'tensorflow1':
    #         print(f"Saving TF1 model to {filename}")
    #         input("Press Enter to continue...")
    #         self.model.model.save(filename)
    #         assert path.exists(filename), "Saving experiment unsuccessful"
    #     else:
    #         dump(self.model, open(filename+".pkl", 'wb'))
    #         assert path.exists(filename+".pkl"), "Saving experiment unsuccessful"
            
    #     logger.info("Saved experiment to {}".format(filename))
    #     return None

