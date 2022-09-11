import numpy as np
from numpy import AxisError
import pandas as pd
import os, logging, json, yaml
from typing import Union, Callable


from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, \
    mean_squared_error, r2_score, mean_absolute_percentage_error, explained_variance_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, roc_curve,\
    balanced_accuracy_score, accuracy_score, precision_score, recall_score, roc_auc_score
# Default scorers
REGRESSOR_SCORERS = {'MAPE': mean_absolute_percentage_error, "MSE" : mean_squared_error, 'MAE': mean_absolute_error,  "R2" : r2_score, "EXVAR" : explained_variance_score}
CLASSIFIER_SCORERS = {'F1' : f1_score, 'ACC' : accuracy_score, 'PREC' : precision_score, 'REC' : recall_score,'AUC': roc_auc_score, 'ROC': roc_curve, "BALACC" : balanced_accuracy_score}


logger = logging.getLogger(__name__)
class Scorer():
    def __init__(self, name:Union[str, list] = None, scorers:Union[Callable, list] = None, best:Union[str, int, float] = None,  is_regressor:bool = False):
        """
        Initialize the scorer.
        :param name: Name of the scorer.
        :param score_function: Function that takes predictions and ground truth and returns a score.
        """
        if name is None and scorers is None:
            if is_regressor:
                scorers = list(REGRESSOR_SCORERS.values())
                name = list(REGRESSOR_SCORERS.keys())
            else:
                scorers = list(CLASSIFIER_SCORERS.values())
                name = list(CLASSIFIER_SCORERS.keys())
        if isinstance(name, list) or isinstance(scorers, list):
            assert len(name) == len(scorers), 'If a list of scorers is provided, the list of names must be the same length.'
        self.name = name if isinstance(name, list) else [name]
        logger.info("Scorer {} initialized.".format(self.name))
        self.scorers = scorers if isinstance(scorers, list) else [scorers]
        self.scores = []
    
    def __repr__(self) -> str:
        score_dict = {}
        for name, score in zip(self.name, self.scores):
            score_dict[name] = score
        return str(score_dict)
    
    def read_data_from_json(self, json_file:str):
        """ Read data from json file. """
        try:
            data = pd.read_json(json_file)
            logger.info("Data read from json file {}.".format(json_file))
        except ValueError as e:
            # Below catches the case in which the json is only key-value pairs, without an index.
            if 'you must pass an index' in str(e):
                with open(json_file, 'r') as f:
                    new_dict = {'0': json.load(f)}
                data = pd.DataFrame(new_dict)
                logger.info("Data read from json file {}. Attempting to set index due to a failure.".format(json_file))
            else:
                raise e
        return data
    
    def read_score_from_json(self, name: str, score_file:str):
        """ Read score from score file. """
        assert hasattr(self, 'name'), 'Scorer must be initialized with a name.'
        with open(score_file, 'r') as f:
            score_dict = json.load(f)
        logger.info("Score read from score file {}.".format(score_file))
        assert name in score_dict, 'Scorer name, {}, not found in json file: {}.'.format(self.name, score_file)
        return score_dict[name]
    
    def score(self, ground_truth:pd.DataFrame, predictions:pd.DataFrame) -> None:
        """
        Sets scorers for evalauation if specified, returns a dict of scores in general.
        """
        scores = {}
        if len(ground_truth.shape) > 1:
            ground_truth = np.argmax(ground_truth, axis=1)
        if len(predictions.shape) > 1:
            predictions = np.argmax(predictions, axis=1)
        for scorer, name in zip(self.scorers, self.name):
            try:
                scores[name] = scorer(ground_truth, predictions)
            except AxisError as e:
                y_test = LabelBinarizer().fit(ground_truth).transform(ground_truth)
                predictions = LabelBinarizer().fit(ground_truth).transform(predictions)
                scores[name] = scorer(y_test, predictions, multi_class='ovr')
            except ValueError as e:
                if "average=" in str(e):
                    scores[name] = scorer(ground_truth, predictions, average='weighted')
                elif 'multi_class must be in' in str(e):
                    y_test = LabelBinarizer().fit(ground_truth).transform(ground_truth)
                    predictions = LabelBinarizer().fit(ground_truth).transform(predictions)
                    scores[name] = scorer(y_test, predictions, multi_class='ovr')
                elif 'Only one class present in y_true' in str(e):
                    pass
                else:
                    raise e
        self.scores = scores
        return self.scores
    
    
    def get_name(self):
        """ Return the name of the scorer. """
        logger.info("Returning name {}.".format(self.name))
        return self.name
        
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
            assert isinstance(name, str), "Name must be a string"
        if isinstance(scorers, list) or isinstance(names, list):
            assert len(scorers) == len(names), 'If a list of scorers is provided, the list of names must be the same length.'
        self.scorers = scorers if isinstance(scorers, list) else [scorers]
        self.name = names if isinstance(names, list) else [names]
        return None

    def save_score(self, results, filename:str = "scores.json", prefix = None, path:str = ".") -> None:
        """
        Saves scores to specified file.
        :param filename: str, name of file to save scores to.
        :param path: str, path to folder to save scores. If none specified, scores are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        score_file = os.path.join(path, filename)

        results = pd.Series(results.values(), name = filename, index = results.keys())
        results.to_json(score_file)
        assert os.path.exists(score_file), "Score file not saved"
        return results
    
    def save_list_score(self, results, filename:str = "scores.json", prefix = None, path:str = ".") -> None:
        """
        Saves scores to specified file.
        :param filename: str, name of file to save scores to.
        :param path: str, path to folder to save scores. If none specified, scores are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        filetype = filename.split('.')[-1]
        score_file = os.path.join(path, filename)
        try:
            results = pd.DataFrame(results.values(), name =  score_file, index = results.keys())
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
        return None
    
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
        for name, score in zip(self.name, self.scores):
            if isinstance(score, list):
                filename = name + filetype
                self.save_list_score({name: score}, filename=filename, prefix=prefix, path=path)
            else:
                save_names.append(name)
                save_scores.append(score)
        dict_ = {zip(save_names, save_scores)}
        self.save_score(dict_, filename = 'scores'+filetype, prefix=prefix, path=path)
        return None

    def __call__(self, ground_truth_file:str, predictions_file:str, path:str = ".", prefix:str =None, filetype = '.json'):
        """ Score the predictions from the file and updates best score. """
        logger.info("Reading from {} and {}.".format(ground_truth_file, predictions_file))
        predictions = self.read_data_from_json(predictions_file)
        ground_truth = self.read_data_from_json(ground_truth_file)
        self.scores = self.score(ground_truth, predictions)
        self.save_results(prefix = prefix, path = path, filetype = filetype)
        return self.scores
