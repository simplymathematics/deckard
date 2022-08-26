import numpy as np
import pandas as pd
import os, logging, json, yaml
from typing import Union
logger = logging.getLogger(__name__)
class Scorer():
    def __init__(self, name:str = None, score_function:callable = None, best:Union[str, int, float] = None, smaller_is_better:bool = False):
        """
        Initialize the scorer.
        :param name: Name of the scorer.
        :param score_function: Function that takes predictions and ground truth and returns a score.
        """
        assert name is not None or score_function is not None, 'Scorer must be initialized with a name or a score function.'
        self.name = name
        logger.info("Scorer {} initialized.".format(self.name))
        self.score_function = score_function
        logger.info("Scorer {} score function initialized.".format(self.name))
        self.smaller_is_better = smaller_is_better
        logger.info("Scorer {} smaller is better initialized.".format(self.name))
        if best is None:
            self.best = 1e9 if self.smaller_is_better else -1e9
        elif isinstance(self.best, str) and os.path.isfile(self.best):
            self.best = self.read_score_from_json(self.best)
        elif isinstance(self.best, (int, float)):
            self.best = float(self.best)
        else:
            raise ValueError('Best score must be a file or a number. It is {}.'.format(type(self.best)))
        logger.info("Scorer {} best score initialized.".format(self.name))
    
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
    
    def read_score_from_json(self, json_file:str):
        """ Read score from json file. """
        assert hasattr(self, 'name'), 'Scorer must be initialized with a name.'
        with open(json_file, 'r') as f:
            score_dict = json.load(f)
        logger.info("Score read from json file {}.".format(json_file))
        assert self.name in score_dict, 'Scorer name, {}, not found in json file.'.format(self.name)
        return score_dict[self.name]

    def score(self, ground_truth:pd.DataFrame, predictions:pd.DataFrame):
        """
        Score the predictions.
        :param predictions: Predictions.
        :param ground_truth: Ground truth.
        """
        logger.info("Scoring predictions with scorer {} and function {}.".format(self.name, self.score_function))
        return self.score_function(ground_truth, predictions)
    
    def update_best(self, score:Union[str, int, float]):
        """
        Update the best score.
        :param score: Score to update with.
        """
        if isinstance(score, str) and os.path.isfile(score):
            logger.info("Reading score from json file {}.".format(score))
            score = self.read_score_from_json(score)
        elif isinstance(score, (int, float)):
            logger.info("Score is a number {}.".format(score))
            score = float(score)
        else:
            raise ValueError('Score must be a file or a number. It is {}.'.format(type(score)))
        if self.smaller_is_better:
            if score < self.best:
                self.best = score
        else:
            if score > self.best:
                self.best = score
        return self.best
    
    def get_best(self):
        """ Return the best score. """
        logger.info("Returning best score {}.".format(self.best))
        return self.best
    
    def get_name(self):
        """ Return the name of the scorer. """
        logger.info("Returning name {}.".format(self.name))
        return self.name
    
    def evaluate_function(self, ground_truth_file:str, predictions_file:str):
        """ Score the predictions from the file and updates best score. """
        logger.info("Reading from {} and {}.".format(ground_truth_file, predictions_file))
        predictions = self.read_data_from_json(predictions_file)
        ground_truth = self.read_data_from_json(ground_truth_file)
        score = self.score(ground_truth, predictions)
        logger.info("Score is {}. Current best score is {}.".format(score, self.best))
        self.update_best(score)
        return self.best
    
    def evaluate_score_from_json(self, score_file:str):
        """ Score the predictions from the file and updates best score. """
        logger.info("Reading score from json file {}.".format(score_file))
        score = self.update_best(score_file)
        logger.info("Score is {}. Current best score is {}.".format(score, self.best))
        return score
    
