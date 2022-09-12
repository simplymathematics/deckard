from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
import os, logging, json, yaml
from typing import Union, Callable
from pathlib import Path

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, \
    mean_squared_error, r2_score, mean_absolute_percentage_error, explained_variance_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, roc_curve,\
    balanced_accuracy_score, accuracy_score, precision_score, recall_score, roc_auc_score
# Default scorers
REGRESSOR_SCORERS = {'MAPE': mean_absolute_percentage_error, "MSE" : mean_squared_error, 'MAE': mean_absolute_error,  "R2" : r2_score, "EXVAR" : explained_variance_score}
CLASSIFIER_SCORERS = {'F1' : f1_score, 'ACC' : accuracy_score, 'PREC' : precision_score, 'REC' : recall_score,'AUC': roc_auc_score}


logger = logging.getLogger(__name__)
class Scorer():
    def __init__(self, config:dict = None, is_regressor:bool = None):
        """
        Initialize the scorer.
        :param config: dict, configuration for the scorer.
        :param is_regressor: bool, whether the scorer is a regressor or not.
        :param score_function: Function that takes predictions and ground truth and returns a score.
        """
        assert config or is_regressor is not None, "Must specify either config or is_regressor."
        if config is None:
            assert is_regressor is not None, "If no config is provided, is_regressor must be specified."
            if self.is_regressor:
                scorers = list(REGRESSOR_SCORERS.values())
                names = list(REGRESSOR_SCORERS.keys())
            else:
                scorers = list(CLASSIFIER_SCORERS.values())
                names = list(CLASSIFIER_SCORERS.keys())
        elif isinstance(config, dict):
            scorers = config.values()
            names = config.keys()       
        elif isinstance(str, Path):
            with open(config, 'r') as f:
                config_ = yaml.load(f, Loader=yaml.FullLoader)
            scorers = config_.values()
            names = config_.keys()
        self.scorers = scorers
        self.names = names
        self.scores = {}
        logger.info("Scorer {} initialized.".format(self.names))
        
    
    def read_data_from_json(self, json_file:str):
        """ Read data from json file. """
        assert os.path.isfile(json_file), "File {} does not exist.".format(json_file)
        data = pd.read_json(json_file)
        return data
    
    def read_score_from_json(self, name: str, score_file:str):
        """ Read score from score file. """
        assert hasattr(self, 'names'), 'Scorer must be initialized with a name.'
        with open(score_file, 'r') as f:
            score_dict = json.load(f)
        logger.info("Score read from score file {}.".format(score_file))
        assert name in score_dict, 'Scorer name, {}, not found in json file: {}.'.format(self.names, score_file)
        return score_dict[name]
    
    def score(self, ground_truth:pd.DataFrame, predictions:pd.DataFrame) -> None:
        """
        Sets scorers for evalauation if specified, returns a dict of scores in general.
        """
        scores = {}
        if predictions.shape != ground_truth.shape:
            raise ValueError("Predictions and ground truth must have the same shape.")
        for name, scorer in zip(self.names, self.scorers):
            try:
                scores[name] = scorer(ground_truth, predictions)
            except ValueError as e:
                # logger.warning(e)
                if "Target is multilabel-indicator but average='binary'" in str(e):
                    logger.warning("Average binary not supported for multilabel-indicator. Using micro.")
                    scores[name] = scorer(ground_truth, predictions, average='weighted')
                elif "multilabel-indicator" and "not supported" in str(e):
                    try:
                        pred = np.argmax(pd.DataFrame(predictions).reset_index(), axis = 1)
                        test = np.argmax(pd.DataFrame(ground_truth).reset_index(), axis = 1)
                        scores[name] = scorer(test, pred)
                    except ValueError as e:
                        logger.warning("Pred shape before/after argmax: ", predictions.shape)
                        logger.warning("Test shape before/after argmax: ", ground_truth.shape)
                        logger.warning("Score {} not supported for multilabel-indicator".format(name))
                        logger.warning("Error Type: {}".format(type(e)))
                        logger.warning("Error: {}".format(e))
                        continue
                else:
                    raise e
                
        self.scores = pd.Series(scores)
        return self.scores
    
    
    def get_name(self):
        """ Return the names of the scorer. """
        logger.info("Returning names {}.".format(self.names))
        return self.names
        
    def get_scorers(self):
        """
        Sets the scorer for an experiment
        :param experiment: experiment to set scorer for
        :param scorer: scorer to set
        """
        return str(self)

    def set_scorers(self, scorers:Union[Callable, list], names:Union[str, list]) -> None:
        """
        Sets the scorer for an experiment
        :param experiment: experiment to set scorer for
        :param scorer: scorer to set
        """
        for scorer in scorers:
            assert isinstance(scorer, Callable), "Scorer must be callable"
        for name in names:
            assert isinstance(name, str), "Names must be a string"
        if isinstance(scorers, list) or isinstance(names, list):
            assert len(scorers) == len(names), 'If a list of scorers is provided, the list of names must be the same length.'
        self.scorers = scorers if isinstance(scorers, list) else [scorers]
        self.names = names if isinstance(names, list) else [names]
        return None

    def save_score(self, results, filename:str = "scores.json", prefix = None, path:str = ".") -> None:
        """
        Saves scores to specified file.
        :param filename: str, names of file to save scores to.
        :param path: str, path to folder to save scores. If none specified, scores are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        score_file = os.path.join(path, filename)
        if not isinstance(results, pd.Series):
            results = pd.Series(results.values(), name = filename, index = results.keys())
        results.to_json(score_file)
        assert os.path.exists(score_file), "Score file not saved"
        return results
    
    def save_list_score(self, results, filename:str = "scores.json", prefix = None, path:str = ".") -> None:
        """
        Saves scores to specified file.
        :param filename: str, names of file to save scores to.
        :param path: str, path to folder to save scores. If none specified, scores are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        filetype = filename.split('.')[-1]
        if prefix is not None:
            score_file = os.path.join(path, prefix + "_" + filename)
        else:
            score_file = os.path.join(path, filename)
        try:
            results = pd.DataFrame(results.values(), names =  score_file, index = results.keys())
        except TypeError as e:
            if "unexpected keyword argument 'name'" in str(e):
                results = pd.DataFrame(results.values(), index = results.keys())
            else:
                raise e
        if filetype == 'json':
            results.to_json(score_file)
        elif filetype == 'csv':
            results.to_csv(score_file)
        else:
            raise NotImplementedError("Filetype {} not implemented.".format(filetype))
        assert os.path.exists(score_file), "Score file not saved"
        return results
    
    def save_results(self, prefix = None, path:str = ".",  filetype = '.json') -> None:
        """
        Saves all data to specified folder, using default filenames.
        """
        if not os.path.isdir(path):
            try:
                os.mkdir(path)
            except FileExistsError:
                logger.warning("Path {} already exists. Overwriting".format(path))
        save_names = []
        save_scores = []
        results = {}
        for name, score in zip(self.names, self.scores):
            results[name] = score
            if isinstance(score, (list, tuple)):
                filename = name + filetype
                result = self.save_list_score({name: score}, filename=filename, prefix=prefix, path=path)
                results[name] = result
            else:
                save_names.append(name)
                save_scores.append(score)
            dict_ = {save_name:save_scores for save_name, save_scores in zip(save_names, save_scores)}
            
        final_result = self.save_score(dict_, filename = 'scores'+filetype, prefix=prefix, path=path)
        self.scores = results.update(final_result)
        return self

    def __call__(self, ground_truth_file:str, predictions_file:str, path:str = ".", prefix:str =None, filetype = '.json'):
        """ Score the predictions from the file and updates best score. """
        logger.info("Reading from {} and {}.".format(ground_truth_file, predictions_file))
        try:
            predictions = self.read_data_from_json(predictions_file)
            ground_truth = self.read_data_from_json(ground_truth_file)
            self.scores = self.score(ground_truth, predictions)
        except Exception as e:
            raise e
        self.save_results(prefix = prefix, path = path, filetype = filetype)
        return self
