import collections
from pathlib import Path
from time import process_time
from typing import List
import numpy as np
import yaml
import pickle
import logging
import json
from copy import deepcopy
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import is_regressor, is_classifier
from argparse import Namespace

# from dvc.api import params_show
# from dvclive import Live
# from tqdm import tqdm
from .utils import factory
from .data import Data
from .model import Model
from .hashable import BaseHashable, my_hash
from .scorer import Scorer
from .attack import Attack
from .visualise import Yellowbrick_Visualiser

logger = logging.getLogger(__name__)


class Experiment(
    collections.namedtuple(
        typename="Experiment",
        field_names="data, model, attack, scorers, plots, files",
        defaults=({}, {}, {}, {}, {}),
    ),
    BaseHashable,
):
    def __new__(cls, loader, node):
        return super().__new__(cls, **loader.construct_mapping(node))

    def load(self) -> tuple:
        """
        Loads data, model from the config file.
        :param config: str, path to config file.
        :returns: tuple(dict, dict, dict, dict, YellowbrickVisualizer), (data, model, attack, files, vis).
        """
        params = deepcopy(self._asdict())
        if len(params["data"]) > 0:
            yaml.add_constructor("!Data:", Data)
            data_document = """!Data:\n""" + str(dict(params["data"]))
            data = yaml.load(data_document, Loader=yaml.Loader)
            assert isinstance(
                data,
                Data,
            ), "Data initialization failed. Check config file."
        else:
            raise ValueError("Data not specified in config file")
        if len(params["model"]) > 0:
            yaml.add_constructor("!Model:", Model)
            model_document = """!Model:\n""" + str(dict(params["model"]))
            model = yaml.load(model_document, Loader=yaml.Loader)
            assert isinstance(
                model,
                Model,
            ), "Model initialization failed. Check config file."
        else:
            model = {}
        files = deepcopy(params["files"]) if "files" in params else {}
        return (data, model, files)

    def save_data(self, data: dict) -> Path:
        """Saves data to specified file.
        :data data: dict, data to save.
        :returns: Path, path to saved data.
        """
        assert "files" in self.data, "Data must have files attribute"
        assert "data_path" in self.data["files"], "Data must have data_path attribute"
        assert (
            "data_filetype" in self.data["files"]
        ), "Data must have data_filetype attribute"
        filename = Path(
            self.data["files"]["data_path"],
            my_hash(self.data) + "." + self.data["files"]["data_filetype"],
        )
        path = Path(filename).parent
        path.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        return str(Path(filename).as_posix())

    def save_params(self) -> Path:
        """Saves parameters to specified file.
        :returns: Path, path to saved parameters.
        """
        filename = Path(self.files["path"], self.files["params_file"])
        filename.parent.mkdir(parents=True, exist_ok=True)
        params = deepcopy(self._asdict())
        files = params.pop("files", None)
        new_files = {}
        if files is not None:
            path = files.pop("path", ".")
            path = Path(path)
            for x in files:
                x = Path(path, files[x])
                new_files[x] = str(x.relative_to(path.parent).as_posix())
        files = new_files
        data_files = params["data"].pop("files", {})
        data_path = data_files.pop("data_path", "")
        data_filetype = data_files.pop("data_filetype", "")
        data_file = Path(data_path, my_hash(self.data) + "." + data_filetype)
        model_files = params["model"].pop("files", {})
        model_path = model_files.pop("path", "")
        model_filetype = model_files.pop("model_filetype", "")
        model_file = Path(model_path, my_hash(self.model) + "." + model_filetype)
        files["data"] = str(data_file)
        files["model"] = str(model_file)
        params["files"] = files
        params["scorer"] = (
            list(params.pop("scorers", {}).keys())[0]
            if len(params["scorers"]) > 0
            else None
        )
        params.pop("plots", None)
        path.mkdir(parents=True, exist_ok=True)
        pd.Series(params).to_json(filename)
        return str(Path(filename).as_posix())

    def save_model(self, model: object) -> Path:
        """Saves model to specified file.
        :model model: object, model to save.
        :returns: Path, path to saved model.
        """
        assert "files" in self.model, "Model must have files attribute"
        assert (
            "model_path" in self.model["files"]
        ), "Model must have model_path attribute"
        assert (
            "model_filetype" in self.model["files"]
        ), "Model must have model_filetype attribute"
        filename = Path(
            self.model["files"]["model_path"],
            my_hash(self.model) + "." + self.model["files"]["model_filetype"],
        )
        path = Path(filename).parent
        file = Path(filename).name
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save"):
            if file.endswith(".pickle") or file.endswith(".pkl"):
                model.save(Path(filename).stem, path=path)
            else:
                model.save(filename=file, path=path)
        else:
            with open(filename, "wb") as f:
                pickle.dump(model, f)
        return str(Path(filename).as_posix())

    def save_predictions(
        self,
        predictions: np.ndarray,
    ) -> Path:
        """Saves predictions to specified file.
        :param predictions: np.ndarray, predictions to save.
        :returns: Path, path to saved predictions.
        """
        filename = Path(
            self.files["path"],
            self.files["predictions_file"],
        )
        path = Path(filename).parent
        path.mkdir(parents=True, exist_ok=True)
        try:
            pd.Series(predictions).to_json(filename)
        except ValueError as e:
            if "1-d" in str(e):
                predictions = np.argmax(predictions, axis=1)
                pd.Series(predictions).to_json(filename)
            else:
                raise
        return str(Path(filename).as_posix())

    def save_ground_truth(
        self,
        ground_truth: np.ndarray,
    ) -> Path:
        """
        :param filename: str, name of file to save predictions to.
        :param ground_truth: np.ndarray, ground truth to save.
        :returns: Path, path to saved ground truth.
        """
        filename = Path(
            self.files["path"],
            self.files["ground_truth_file"],
        )
        path = Path(filename).parent
        path.mkdir(parents=True, exist_ok=True)
        try:
            pd.Series(ground_truth).to_json(filename)
        except ValueError as e:
            if "1-d" in str(e):
                ground_truth = np.argmax(ground_truth, axis=1)
                pd.Series(ground_truth).to_json(filename)
            else:
                raise
        return str(Path(filename).as_posix())

    def save_time_dict(self, time_dict: dict) -> Path:
        """
        :param filename: str, name of file to save predictions to.
        :param time_dict: dict, time dictionary to save.
        :returns: Path, path to saved time dictionary.
        """
        filename = Path(
            self.files["path"],
            self.files["time_dict_file"],
        )
        path = Path(filename).parent
        path.mkdir(parents=True, exist_ok=True)
        if filename.exists():
            with open(filename, "r") as f:
                old_dict = json.load(f)
            time_dict = old_dict.update(time_dict)
        pd.Series(time_dict).to_json(filename)
        return str(Path(filename).as_posix())

    def save_scores(self, scores: dict) -> Path:
        """
        :param filename: str, name of file to save predictions to.
        :param scores: dict, scores to save.
        :returns: Path, path to saved scores.
        """
        filename = Path(
            self.files["path"],
            self.files["score_dict_file"],
        )
        path = Path(filename).parent
        path.mkdir(parents=True, exist_ok=True)
        pd.Series(scores).to_json(
            filename,
        )
        return str(Path(filename).as_posix())

    def fit(self, data: Namespace, model: object, art_bool=None) -> tuple:
        """
        Fits model to data.
        :param data: dict, data to fit model to.
        :param model: object, model to fit.
        :returns: tuple, (model, data, fit_time).
        """
        fit_params = deepcopy(self.model["fit"]) if "fit" in self.model else {}
        if isinstance(self.attack, dict) and self.attack is not {}:
            art_bool = True
        else:
            art_bool = False
        if isinstance(data, Data):
            loaded_data = data.load()
        if isinstance(model, Model):
            loaded_model = model.load(art_bool)
        start = process_time()
        try:
            loaded_model.fit(loaded_data.X_train, loaded_data.y_train, **fit_params)
        except np.AxisError as e:
            loaded_data.y_train = LabelBinarizer().fit_transform(loaded_data.y_train)
            loaded_data.y_test = (
                LabelBinarizer().fit(loaded_data.y_train).transform(loaded_data.y_test)
            )
            try:
                loaded_model.fit(loaded_data.X_train, loaded_data.y_train, **fit_params)
            except ValueError as e:
                if "number of classes" in str(e):
                    loaded_model.fit(
                        loaded_data.X_train, loaded_data.y_train, **fit_params
                    )
                else:
                    raise e
        result = process_time() - start
        return (loaded_data, loaded_model, result / len(loaded_data.X_train))

    def predict(self, data: dict, model: object, art=False) -> tuple:
        """
        Predicts data with model.
        :param data: dict, data to predict.
        :param model: object, model to predict with.
        :returns: tuple, (predictions, predict_time).
        """
        start = process_time()
        if isinstance(data, Data):
            data = data.load()
        if isinstance(model, Model):
            model = model.load(art=art)
        predictions = model.predict(data.X_test)
        if len(predictions.shape) > 1:
            predictions = predictions.argmax(axis=1)
        result = process_time() - start
        return predictions, result / len(data.X_test)

    def save(
        self,
        data: dict = None,
        ground_truth: np.ndarray = None,
        model: object = None,
        time_dict: dict = None,
        score_dict: dict = None,
        predictions: np.ndarray = None,
    ) -> dict:
        """
        Saves data, model, parameters, predictions, scores, and time dictionary.
        :param data: dict, data to save.
        :param ground_truth: np.ndarray, ground truth to save.
        :param model: object, model to save.
        :param params: dict, parameters to save.
        :param time_dict: dict, time dictionary to save.
        :param predictions: np.ndarray, predictions to save.
        """
        files = {}
        path = Path(self.files["path"])
        path.mkdir(parents=True, exist_ok=True)
        if "files" in self.data:
            files.update({"data": self.save_data(data)})
        if "files" in self.model:
            files.update({"model": self.save_model(model)})
        if "params_file" in self.files:
            files.update({"params": self.save_params()})
        if "ground_truth_file" in self.files:
            files.update({"ground_truth": self.save_ground_truth(ground_truth)})
        if "predictions_file" in self.files:
            files.update({"predictions": self.save_predictions(predictions)})
        if time_dict is not None:
            files.update({"time": self.save_time_dict(time_dict)})
        if score_dict is not None:
            files.update({"scores": self.save_scores(score_dict)})
        return files

    def run(
        self,
        fit=True,
        predict=True,
        score=True,
        visualise=True,
        art=False,
        attack=True,
        mtype="classifier",
    ) -> dict:
        """
        Runs experiment and saves results according to config file.
        """
        #######################################################################
        logger.info("Parsing Config File")
        data, model, files = self.load()
        params = deepcopy(self._asdict())
        time_dict = {}
        outs = {}
        path = Path(params["files"]["path"], str(my_hash(self._asdict())))
        self.files.update({"path": str(path.as_posix())})
        if path.exists():
            logger.warning(
                f"Path {path} already exists. Will overwrite any files specified in the config.",
            )
        else:
            path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving to {path}")
        assert isinstance(data, Data)
        assert isinstance(model, (Model, dict))
        #######################################################################
        # Fit model, if applicable
        if fit is True:
            logger.info("Fitting model")
            loaded_data, fitted_model, fit_time = self.fit(data, model)
            time_dict.update({"fit_time": fit_time})
            results = {
                "data": loaded_data,
                "model": fitted_model,
                "time_dict": {"fit_time": fit_time},
                "ground_truth": loaded_data.y_test,
            }
        elif isinstance(model, Model):
            loaded_data = data.load()
            fitted_model = model.load(art=art)
            time_dict = {}
            results = {
                "data": loaded_data,
                "model": fitted_model,
                "ground_truth": loaded_data.y_test,
            }
            results["time_dict"] = time_dict
        elif isinstance(model, dict):
            loaded_data = data.load()
            results = {"data": loaded_data, "ground_truth": loaded_data.y_test}
        #######################################################################
        if predict is True:
            logger.info("Predicting")
            predictions, predict_time = self.predict(loaded_data, fitted_model)
            time_dict.update({"predict_time": predict_time})
            results.update({"predictions": predictions})
            results["time_dict"].update(time_dict)
        #######################################################################
        if score is True:
            logger.info("Scoring")
            score_dict = self.score(loaded_data.y_test, predictions)
            results.update({"score_dict": score_dict})
        #######################################################################
        if visualise is True:
            if "art" in str(type(fitted_model)):
                art = True
            else:
                art = False
            self.files["path"] = str(path.as_posix())
            plots = self.visualise(data=loaded_data, model=fitted_model, art=art)
            outs.update({"plots": plots})
            logger.info("Visualising")
        #######################################################################
        attack_keys = len(self.attack)
        if attack is True and attack_keys > 0:
            attack = "!Attack:\n" + str(self._asdict())
            yaml.add_constructor("!Attack:", Attack)
            attack = yaml.load(attack, Loader=yaml.FullLoader)
            self.files["attack_path"] = str(path.as_posix())
            targeted = (
                attack.attack["init"]["targeted"]
                if "targeted" in attack.attack["init"]
                else False
            )
            attack_results = attack.run_attack(
                data=loaded_data,
                model=fitted_model,
                mtype=mtype,
                targeted=targeted,
            )
        saved_files = self.save(**results)
        outs.update(saved_files)
        return outs

    def visualise(self, data, model, mtype=None, art: bool = False) -> List[Path]:
        """
        Visualises data and model according to config file.

        Args:
            data (Namespace): _description_
            model (object): _description_
            path (_type_, optional): _description_. Defaults to path.

        Returns:
            List[Path]: _description_
        """
        plots = []

        yaml.add_constructor("!YellowBrick_Visualiser:", Yellowbrick_Visualiser)
        vis = yaml.load(
            "!YellowBrick_Visualiser:\n" + str(self._asdict()),
            Loader=yaml.FullLoader,
        )
        plot_dict = vis.visualise(data=data, model=model, mtype=mtype, art=art)
        plots.extend(plot_dict)
        return plots

    def score(self, ground_truth, predictions) -> List[Path]:
        """
        :param self: specified in the config file.
        """
        yaml.add_constructor("!Scorer:", Scorer)
        scorer = yaml.load("!Scorer:\n" + str(self._asdict()), Loader=yaml.FullLoader)
        score_paths = scorer.score_from_memory(ground_truth, predictions)
        return score_paths


config = """
    model:
        init:
            loss: "hinge"
            name: sklearn.linear_model.SGDClassifier
            alpha: 0.0001
        files:
            model_path : model
            model_filetype : pickle
        # fit:
        #     epochs: 1000
        #     learning_rate: 1.0e-08
        #     log_interval: 10
        art_pipeline:
            preprocessor:
                name: art.defences.preprocessor.FeatureSqueezing
                bit_depth: 32
        sklearn_pipeline:
            feature_selection:
                name: sklearn.feature_selection.SelectKBest
                k : 2

    data:
        sample:
            shuffle : True
            random_state : 42
            train_size : 800
            stratify : True
        add_noise:
            train_noise : 1
        name: classification
        files:
            data_path : data
            data_filetype : pickle
        generate:
            n_samples: 1000
            n_features: 2
            n_informative: 2
            n_redundant : 0
            n_classes: 3
            n_clusters_per_class: 1
        sklearn_pipeline:
            scaling :
                name : sklearn.preprocessing.StandardScaler
                with_mean : true
                with_std : true
    attack:
        init:
            name: art.attacks.evasion.HopSkipJump
            max_iter : 10
            init_eval : 10
            init_size : 10
    plots:
        balance: balance
        classification: classification
        confusion: confusion
        correlation: correlation
        radviz: radviz
        rank: rank
    scorers:
        accuracy:
            name: sklearn.metrics.accuracy_score
            normalize: true
        f1-macro:
            average: macro
            name: sklearn.metrics.f1_score
        f1-micro:
            average: micro
            name: sklearn.metrics.f1_score
        f1-weighted:
            average: weighted
            name: sklearn.metrics.f1_score
        precision:
            average: weighted
            name: sklearn.metrics.precision_score
        recall:
            average: weighted
            name: sklearn.metrics.recall_score
    files:
        ground_truth_file: ground_truth.json
        predictions_file: predictions.json
        time_dict_file: time_dict.json
        params_file: params.json
        score_dict_file: scores.json
        path: reports
        attack_samples_file: adv_samples.json
        attack_predictions_file : adv_predictions.json
        attack_time_dict_file : adv_time_dict.json
        attack_params_file : attack_params.json
        attack_path : attack
    """
