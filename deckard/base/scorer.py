import json
import logging
import os
from pathlib import Path
from copy import deepcopy
import numpy as np
from .hashable import BaseHashable, my_hash
import pandas as pd
import collections
from .utils import factory

logger = logging.getLogger(__name__)


class Scorer(
    collections.namedtuple(
        typename="Scorer",
        field_names="data, scorers, files, attack, model,  plots",
        defaults=({}, {}, {}, {}),
        rename=True,
    ),
    BaseHashable,
):
    def __new__(cls, loader, node):
        return super().__new__(cls, **loader.construct_mapping(node))

    def read_data_from_json(self, json_file: str):
        """Read data from json file."""
        with open(json_file, "r") as f:
            data = json.load(f)
        data = pd.Series(data)
        return data

    def read_score_from_json(self, name: str, score_file: str):
        """Read score from score file."""
        with open(score_file, "r") as f:
            score_dict = json.load(f)
        logger.info("Score read from score file {}.".format(score_file))
        return score_dict[name]

    def score_from_memory(
        self,
        ground_truth: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> None:
        """
        Sets scorers for evalauation if specified, returns a dict of scores in general.
        """
        scores = {}
        # if predictions.shape != ground_truth.shape:
        #     raise ValueError("Predictions and ground truth must have the same shape.")
        names = deepcopy(self.scorers).keys()
        scorers = [deepcopy(self.scorers[name]) for name in names]
        for name, scorer in zip(names, scorers):
            obj_name = scorer.pop("name")
            try:
                score = factory(
                    obj_name, **scorer, y_pred=predictions, y_true=ground_truth
                )
            except ValueError as e:
                if len(predictions.shape) > 1:
                    predictions = np.argmax(predictions, axis=1)
                if len(ground_truth.shape) > 1:
                    ground_truth = np.argmax(ground_truth, axis=1)
                score = factory(
                    obj_name, **scorer, y_pred=predictions, y_true=ground_truth
                )
            scores[name] = score
        scores = pd.Series(scores).T
        return scores

    def save(
        self,
        results,
    ) -> None:
        """
        Saves scores to specified file.
        :param filename: str, names of file to save scores to.
        :param path: str, path to folder to save scores. If none specified, scores are saved in current working directory. Must exist.
        """
        files = deepcopy(self._asdict())["files"]
        score_file = Path(
            files.pop("path"),
            my_hash(self),
            files.pop("score_dict_file"),
        )
        score_file.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(results, pd.Series):
            results = pd.Series(results.values(), name=score_file, index=results.keys())
        results.to_json(score_file)
        assert os.path.exists(score_file), "Score file not saved"
        return str(score_file.as_posix())

    def score(self):
        """Score the predictions from the file and updates best score.
        :param self.files.prediction_file: str, path to file containing predictions.
        :param self.files.ground_truth_file: str, path to file containing ground truth.
        :param self.scorers: dict, dict of scorers to use (set during init).
        :return scores: dict, scores for predictions.
        """
        filenames = deepcopy(self._asdict()["files"])
        path = filenames.pop("path")
        pred_file = filenames.pop("predictions_file")
        true_file = filenames.pop("ground_truth_file")
        pred_file = Path(path, my_hash(self), pred_file)
        true_file = Path(path, my_hash(self), true_file)
        test = self.read_data_from_json(pred_file)
        true = self.read_data_from_json(true_file)
        scores = self.score(true, test)
        return scores

    def score_attack(self):
        """
        Runs the attack on the model
        """
        files = deepcopy(self._asdict())["attack"]["files"]
        pred_file = files.pop("adv_predictions_file")
        pred_file = Path(files.pop["path"], pred_file)
        true_file = files.pop("ground_truth_file")
        true_file = Path(files.pop["path"], true_file)
        test = self.read_data_from_json(pred_file)
        true = self.read_data_from_json(true_file)
        scores = self.score(true, test)
        path = self.save_score(results=scores)
        return str(path.as_posix())
