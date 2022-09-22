import logging, os, pickle


# Operating System
import yaml
from pathlib import Path
from typing import Union
from pandas import DataFrame, Series

# Math Stuff
import numpy as np
from pandas import Series


from deckard.base.model import Model
from deckard.base.data import Data
from deckard.base.hashable import BaseHashable

from art.defences.postprocessor import Postprocessor
from art.defences.preprocessor import Preprocessor
from art.defences.trainer import Trainer
from art.defences.transformer import Transformer

DEFENCE_TYPES = [Preprocessor, Trainer, Transformer, Postprocessor]

logger = logging.getLogger(__name__)


# Create experiment object
class Experiment(BaseHashable):
    """
    Creates an experiment object
    """

    def __init__(
        self,
        data: Data,
        model: Model,
        verbose: int = 1,
        is_fitted: bool = False,
        fit_params: dict = None,
        predict_params: dict = None,
    ):
        """
        Creates an experiment object
        :param data: Data object
        :param model: Model object
        :param params: Dictionary of other parameters you want to add to this object. Obviously everything in self.__dict__.keys() should be treated as a reserved keyword, however.
        :param verbose: Verbosity level
        :param scorers: Dictionary of scorers
        :param name: Name of experiment
        """

        self.verbose = verbose
        self.is_fitted = is_fitted
        self.params = {}
        self.params["Model"] = model.params
        self.params["Data"] = model.params
        self.data = data
        self.model = model
        self.hash = hash(self)
        self.params["Experiment"] = {
            "verbose": self.verbose,
            "is_fitted": self.is_fitted,
            "id": hash(self),
            "model": hash(model),
            "data": hash(data),
        }
        self.time_dict = None
        self.predictions = None
        self.ground_truth = None

    def fit(self) -> None:
        """
        Builds model.
        """
        self.model.is_fitted = self.is_fitted
        if not self.is_fitted:
            self.model.fit(self.data.X_train, self.data.y_train)
        else:
            logger.info("Model already fitted. Skipping fit.")
        self.predictions = self.model.predict(self.data.X_test)
        self.time_dict = self.model.time_dict
        self.params["Experiment"]["if_fitted"] = True
        self.hash = hash(self)

    def __call__(
        self,
        path,
        model_file: Union[str, Path] = "model",
        prefix=None,
        predictions_file: Union[str, Path] = "predictions.json",
        ground_truth_file: Union[str, Path] = "ground_truth.json",
        time_dict_file: Union[str, Path] = "time_dict.json",
        params_file: Union[str, Path] = "params.json",
    ) -> list:
        """
        Sets metric scorer. Builds model. Runs evaluation. Updates scores dictionary with results.
        Returns self with added scores, predictions, and time_dict attributes.
        """

        files = self.save_params(
            filename=params_file,
            path=path,
            prefix=prefix,
        )
        if not hasattr(self.data, "X_train"):
            logger.debug("Data not initialized. Initializing.")
            self.data()
        if isinstance(self.model.model, (Path, str)):
            logger.debug("Model not initialized. Initializing.")
            self.model()
        self.ground_truth = self.data.y_test
        if not os.path.isdir(path):
            os.mkdir(path)
        self.fit()
        preds_file = self.save_predictions(
            filename=predictions_file,
            path=path,
            prefix=prefix,
        )
        truth_File = self.save_ground_truth(
            filename=ground_truth_file,
            path=path,
            prefix=prefix,
        )
        time_file = self.save_time_dict(
            filename=time_dict_file,
            path=path,
            prefix=prefix,
        )
        model_file = os.path.join(path, model_file)
        model_name = str(hash(self.model)) if model_file is None else model_file
        model_file = self.save_model(filename=Path(model_name).name, path=path)
        files.extend([preds_file, truth_File, time_file, model_file])
        # TODO: Fix scoring
        return files

    def save_data(
        self,
        filename: str = "data.pkl",
        prefix=None,
        path: str = ".",
    ) -> None:
        """
        Saves data to specified file.
        :param filename: str, name of file to save data to.
        :param path: str, path to folder to save data to. If none specified, data is saved in current working directory. Must exist.
        """
        assert path is not None, "Path to save data must be specified."
        if prefix is not None:
            filename = os.path.join(path, prefix + "_" + filename)
        else:
            filename = os.path.join(path, filename)
        with open(filename, "wb") as f:
            pickle.dump(self.data, f)
        assert os.path.exists(os.path.join(path, filename)), "Data not saved."
        return None

    def save_params(self, filename="params.yaml", prefix=None, path: str = ".") -> None:
        """
        Saves data to specified file.
        :param data_params_file: str, name of file to save data parameters to.
        :param model_params_file: str, name of file to save model parameters to.
        :param path: str, path to folder to save data to. If none specified, data is saved in current working directory. Must exist.
        """
        assert path is not None, "Path to save data must be specified."
        if not os.path.isdir(path) and not os.path.exists(path):
            os.mkdir(path)
        filenames = []
        newname = Path(filename).name
        for key, value in self.params.items():
            filename = newname
            if prefix is not None:
                filename = prefix + key.lower() + "_" + newname
            else:
                filename = key.lower() + "_" + newname
            filename = os.path.join(path, filename)
            print("Saving params to {}".format(filename))
            with open(filename, "w") as f:
                yaml.dump(value, f, indent=4)
            ###################################
            # Enable for debugging:           #
            ###################################
            # with open(filename, "r") as f:  #
            #     print(f.read())             #
            ###################################
            filenames.append(os.path.join(path, filename))
        return filenames

    def save_model(self, filename: str = "model", prefix=None, path: str = ".") -> str:
        """
        Saves model to specified file (or subfolder).
        :param filename: str, name of file to save model to.
        :param path: str, path to folder to save model. If none specified, model is saved in current working directory. Must exist.
        :return: str, path to saved model.
        """
        if prefix is not None:
            filename = prefix + "_" + filename
        assert os.path.isdir(path), "Path {} to experiment does not exist".format(path)
        logger.info("Saving model to {}".format(os.path.join(path, filename)))
        filename = Path(filename).name
        self.model.save_model(filename=filename, path=path)
        return os.path.join(path, filename)

    def save_predictions(
        self,
        filename: str = "predictions.json",
        prefix=None,
        path: str = ".",
    ) -> None:
        """
        Saves predictions to specified file.
        :param filename: str, name of file to save predictions to.
        :param path: str, path to folder to save predictions. If none specified, predictions are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        if prefix is not None:
            filename = prefix + "_" + filename
        prediction_file = os.path.join(path, filename)
        results = self.predictions
        results = DataFrame(results)
        results.to_json(prediction_file, orient="records")
        assert os.path.exists(prediction_file), "Prediction file not saved"
        return prediction_file

    def save_ground_truth(
        self,
        filename: str = "ground_truth.json",
        prefix=None,
        path: str = ".",
    ) -> None:
        """
        Saves ground_truth to specified file.
        :param filename: str, name of file to save ground_truth to.
        :param path: str, path to folder to save ground_truth. If none specified, ground_truth are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        if prefix is not None:
            filename = prefix + "_" + filename
        prediction_file = os.path.join(path, filename)
        results = self.ground_truth
        results = DataFrame(results)
        results.to_json(prediction_file, orient="records")
        assert os.path.exists(prediction_file), "Prediction file not saved"
        return prediction_file

    def save_cv_scores(
        self,
        filename: str = "cv_scores.json",
        prefix=None,
        path: str = ".",
    ) -> None:
        """
        Saves crossvalidation scores to specified file.
        :param filename: str, name of file to save crossvalidation scores to.
        :param path: str, path to folder to save crossvalidation scores. If none specified, scores are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        assert filename is not None, "Filename must be specified"
        if prefix is not None:
            filename = prefix + "_" + filename
        cv_file = os.path.join(path, filename)
        try:
            cv_results = Series(self.model.model.model.cv_results_, name=str(self.hash))
        except:
            cv_results = Series(self.model.model.cv_results_, name=str(self.hash))
        cv_results.to_json(cv_file, orient="records")
        assert os.path.exists(cv_file), "CV results file not saved"
        return cv_file

    def save_time_dict(
        self,
        filename: str = "time_dict.json",
        prefix=None,
        path: str = ".",
    ):
        """
        Saves time dictionary to specified file.
        :param filename: str, name of file to save time dictionary to.
        :param path: str, path to folder to save time dictionary. If none specified, time dictionary is saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        assert hasattr(self, "time_dict"), "No time dictionary to save"
        if prefix is not None:
            filename = prefix + "_" + filename
        time_file = os.path.join(path, filename)
        time_results = Series(self.time_dict, name=str(self.hash))
        time_results.to_json(time_file, orient="records")
        assert os.path.exists(time_file), "Time dictionary file not saved"
        return time_file
