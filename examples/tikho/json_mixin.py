import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import factory


class JSONMixin:
    def score(self, data: dict, predictions: np.ndarray) -> dict:
        """Scores predictions according to self.scorers.
        :param data: dict, data to score predictions on.
        :param predictions: np.ndarray, predictions to score.
        :returns: dict, dictionary of scores.
        """
        score_dict = {}
        for scorer in self.scorers:
            class_name = self.scorers[scorer].pop("name")
            params = self.scorers[scorer]
            params.update({"y_true": data.y_test, "y_pred": predictions})
            score = factory(class_name, **params)
            score_dict[scorer] = score
        return score_dict

    def save_params(self, filename: str, params: dict) -> Path:
        """Saves parameters to specified file.
        :param filename: str, name of file to save parameters to.
        :returns: Path, path to saved parameters.
        """
        path = Path(filename).parent
        path.mkdir(parents=True, exist_ok=True)
        pd.Series(params).to_json(filename)
        return Path(filename).resolve()

    def save_predictions(
        self,
        predictions: np.ndarray,
        filename: str = "predictions.json",
    ) -> Path:
        """Saves predictions to specified file.
        :param filename: str, name of file to save predictions to.
        :param predictions: np.ndarray, predictions to save.
        :returns: Path, path to saved predictions.
        """
        path = Path(filename).parent
        path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(predictions).to_json(filename)
        return Path(filename).resolve()

    def save_ground_truth(
        self,
        ground_truth: np.ndarray,
        filename: str = "ground_truth.json",
    ) -> Path:
        """
        :param filename: str, name of file to save predictions to.
        :param ground_truth: np.ndarray, ground truth to save.
        :returns: Path, path to saved ground truth.
        """
        path = Path(filename).parent
        path.mkdir(parents=True, exist_ok=True)
        pd.Series(ground_truth).to_json(filename)
        return Path(filename).resolve()

    def save_time_dict(self, time_dict: dict, filename: str = "time_dict.json") -> Path:
        """
        :param filename: str, name of file to save predictions to.
        :param time_dict: dict, time dictionary to save.
        :returns: Path, path to saved time dictionary.
        """
        path = Path(filename).parent
        path.mkdir(parents=True, exist_ok=True)
        pd.Series(time_dict).to_json(filename)
        return Path(filename).resolve()

    def save_scores(self, scores: dict, filename: str = "scores.json") -> Path:
        """
        :param filename: str, name of file to save predictions to.
        :param scores: dict, scores to save.
        :returns: Path, path to saved scores.
        """
        path = Path(filename).parent
        path.mkdir(parents=True, exist_ok=True)
        pd.Series(scores).to_json(
            filename,
        )
        return Path(filename).resolve()

    def save(
        self,
        data: dict = None,
        data_file=None,
        ground_truth: np.ndarray = None,
        model: object = None,
        model_file: str = None,
        params: dict = None,
        params_file: str = None,
        score_dict: dict = None,
        time_dict: dict = None,
        predictions: np.ndarray = None,
        time_dict_file: str = None,
        predictions_file: str = None,
        ground_truth_file: str = None,
        score_dict_file: str = None,
        path: Path = ".",
    ) -> Path:
        files = []
        Path(path).mkdir(parents=True, exist_ok=True)
        if score_dict_file is not None:
            score_dict_file = Path(path) / score_dict_file
            files.append(self.save_scores(score_dict, filename=score_dict_file))
        if data_file is not None:
            files.append(self.save_data(data, data_file))
        if model is not None:
            Path(model_file).parent.mkdir(parents=True, exist_ok=True)
            if hasattr(model, "save"):
                files.append(self.save_model(model, model_file))
            else:
                with open(model_file, "wb") as f:
                    pickle.dump(model, f)
                files.append(Path(model_file).resolve())
        if params_file is not None:
            assert (
                params is not None
            ), "params must be specified if params_file is specified"
            params_file = Path(path, params_file)
            files.append(self.save_params(params_file, params))
        if ground_truth is not None:
            ground_truth_file = Path(path, ground_truth_file)
            assert (
                ground_truth_file is not None
            ), "Ground truth must be passed to function call if specified in config file."
            files.append(
                self.save_ground_truth(ground_truth, filename=ground_truth_file),
            )
        if predictions is not None:
            predictions_file = Path(path, predictions_file)
            assert (
                predictions_file is not None
            ), "Predictions must be passed to function call if specified in config file."
            files.append(self.save_predictions(predictions, filename=predictions_file))
        if time_dict is not None:
            assert (
                time_dict_file is not None
            ), "Time dictionary must be passed to function call if specified in config file."
            time_dict_file = Path(path, time_dict_file)
            assert (
                time_dict_file is not None
            ), "Time dictionary must be passed to function call if specified in config file."
            files.append(self.save_time_dict(time_dict, filename=time_dict_file))
        return files
