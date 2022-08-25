import numpy as np
import pandas as pd
import os, logging, json, yaml
from deckard.base.experiment import Experiment, Model, Data
from typing import Union

class Scorer():
    def __init__(self, name:str = None, score_function:callable = None, best:Union[str, int, float] = None, smaller_is_better:bool = False):
        """
        Initialize the scorer.
        :param name: Name of the scorer.
        :param score_function: Function that takes predictions and ground truth and returns a score.
        """
        assert name is not None or score_function is not None, 'Scorer must be initialized with a name or a score function.'
        self.name = name
        self.score_function = score_function
        self.smaller_is_better = smaller_is_better
        if best is None:
            self.best = 1e9 if self.smaller_is_better else -1e9
        elif isinstance(self.best, str) and os.path.isfile(self.best):
            self.best = self.read_score_from_json(self.best)
    
    def read_data_from_json(self, json_file:str):
        """ Read data from json file. """
        try:
            data = pd.read_json(json_file)
        except ValueError as e:
            # Below catches the case in which the json is only key-value pairs, without an index.
            if 'you must pass an index' in str(e):
                pd.read_json(json_file, index = 0)
            else:
                raise e
        return data
    
    def read_score_from_json(self, json_file:str):
        """ Read score from json file. """
        assert hasattr(self, 'name'), 'Scorer must be initialized with a name.'
        with open(json_file, 'r') as f:
            score_dict = json.load(f)
        assert self.name in score_dict, 'Scorer name, {}, not found in json file.'.format(self.name)
        return score_dict[self.name]

    def score(self, ground_truth:pd.DataFrame, predictions:pd.DataFrame):
        """
        Score the predictions.
        :param predictions: Predictions.
        :param ground_truth: Ground truth.
        """
        return self.score_function(ground_truth, predictions)
    
    def update_best(self, score:Union[str, int, float]):
        """
        Update the best score.
        :param score: Score to update with.
        """
        if isinstance(score, str) and os.path.isfile(score):
            score = self.read_score_from_json(score)
        elif isinstance(score, (int, float)) and not os.path.isfile(score):
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
        return self.best
    
    def get_name(self):
        """ Return the name of the scorer. """
        return self.name
    
    def evaluate_function(self, ground_truth_file, predictions_file):
        """ Score the predictions from the file and updates best score. """
        predictions = self.read_data_from_json(predictions_file)
        ground_truth = self.read_data_from_json(ground_truth_file)
        score = self.score(ground_truth, predictions)
        self.update_best(score)
        return self.best
    
    def evaluate_score_from_json(self, score_file):
        """ Score the predictions from the file and updates best score. """
        score = self.update_best(score_file)
        return score
    
