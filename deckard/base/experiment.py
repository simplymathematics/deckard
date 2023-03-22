import collections
import json
import logging
import pickle
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from time import process_time
from typing import List

import numpy as np
import pandas as pd
import yaml
from art.utils import to_categorical
from sklearn.exceptions import NotFittedError

from .attack import Attack
from .data import Data
from .hashable import BaseHashable, my_hash
from .model import Model
from .scorer import Scorer
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
        if "data_file" in params["files"] and Path(params["files"]["data_file"]).exists():
            data = data.load(params["files"]["data_file"])
        
        if len(params["model"]) > 0 :
            yaml.add_constructor("!Model:", Model)
            model_document = """!Model:\n""" + str(dict(params["model"]))
            model = yaml.load(model_document, Loader=yaml.Loader)
            assert isinstance(
                model,
                Model,
            ), "Model initialization failed. Check config file."

        else:
            model = {}
        if "model_file" in params["files"] and Path(params["files"]["model_file"]).exists():
            model = model.load(params["files"]["model_file"])
        files = deepcopy(params["files"]) if "files" in params else {}
        path = Path(files["path"]) if "path" in files else Path.cwd()
        for k, v in files.items():
            if isinstance(v, str) and k not in ["reports", "path"]:
                full_path = Path(path, v)
                if full_path.exists():
                    if full_path.suffix == ".json":
                        with open(full_path, "r") as f:
                            try:
                                files[k] = json.load(f)
                            except json.decoder.JSONDecodeError:
                                files[k] = yaml.load(f, Loader=yaml.Loader)
                    elif full_path.suffix == ".yaml":
                        with open(full_path, "r") as f:
                            files[k] = yaml.load(f, Loader=yaml.Loader)
                    elif full_path.suffix == ".pkl" or full_path.suffix == ".pickle":
                        with open(full_path, "rb") as f:
                            files[k] = pickle.load(f)
                    elif full_path.suffix == ".csv":
                        files[k] = pd.read_csv(full_path)
                    else:
                        raise NotImplementedError(f"File type, {full_path.suffix}, not supported")
                else:
                    files[k] = v
        return (data, model, files)

    def save_data(self, data: dict) -> Path:
        """Saves data to specified file.
        :data data: dict, data to save.
        :returns: Path, path to saved data.
        """
        filename = Path(self.files["data_file"])
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
        with open(filename, "w") as f:
            yaml.dump(params, f, default_flow_style=False)
        assert Path(filename).exists(), "Parameters not saved."
        return str(Path(filename))

    def save_model(self, model: object) -> Path:
        """Saves model to specified file.
        :model model: object, model to save.
        :returns: Path, path to saved model.
        """
        filename = Path(self.files["model_file"])
        path = Path(filename).parent
        path.mkdir(parents=True, exist_ok=True)
        if Path(filename).suffix == ".pkl":
            filename = filename.with_suffix(".pickle")
        if hasattr(model, "save"):
            model.save(Path(filename).stem, path=path)
        else:
            if hasattr("model", "model"):
                model = model.model
            with open(filename, "wb") as f:
                pickle.dump(model, f)
        return str(Path(filename))

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
                time_dict = {**json.load(f), **time_dict}
        with open(filename, "w") as f:
            json.dump(time_dict, f)
        return str(Path(filename).as_posix())

    def save_probabilities(self, probabilities: np.ndarray) -> Path:
        """
        :param filename: str, name of file to save predictions to.
        :param probabilities: np.ndarray, probabilities to save.
        :returns: Path, path to saved probabilities.
        """
        filename = Path(
            self.files["path"],
            self.files["probabilities_file"],
        )
        path = Path(filename).parent
        path.mkdir(parents=True, exist_ok=True)
        pd.Series(probabilities).to_json(filename)
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

    def fit(self, data: Namespace, model: object, art=False) -> tuple:
        """
        Fits model to data.
        :param data: dict, data to fit model to.
        :param model: object, model to fit.
        :returns: tuple, (model, data, fit_time).
        """
        
        fit_params = deepcopy(self.model["fit"]) if "fit" in self.model else {}
        if isinstance(data, Data):
            loaded_data = data.load(self.files["data_file"])
        else:
            loaded_data = data        
        if isinstance(model, Model):
            loaded_model = model.load(self.files["model_file"], art)
        else:
            loaded_model = model
        try:
            start = process_time()
            loaded_model.fit(loaded_data.X_train, loaded_data.y_train, **fit_params)
        except Exception as e:
            if "number of classes" in str(e):
                if len(loaded_data.y_train.shape) == 1:
                    loaded_data.y_train =  to_categorical(loaded_data.y_train)
                if len(loaded_data.y_test.shape) == 1:
                    loaded_data.y_test = to_categorical(loaded_data.y_test)
                start = process_time()
                loaded_model.fit(loaded_data.X_train, loaded_data.y_train, **fit_params)
            elif "1d" in str(e):
                if len(loaded_data.y_train.shape) > 1:
                    loaded_data.y_train = loaded_data.y_train.argmax(axis=1)
                if len(loaded_data.y_test.shape) > 1:
                    loaded_data.y_test = loaded_data.y_test.argmax(axis=1)
                start = process_time()
                loaded_model.fit(loaded_data.X_train, loaded_data.y_train, **fit_params)
            elif "out of bounds" in str(e):
                if len(loaded_data.y_train.shape) == 1:
                    loaded_data.y_train =  to_categorical(loaded_data.y_train)
                if len(loaded_data.y_test.shape) == 1:
                    loaded_data.y_test = to_categorical(loaded_data.y_test)
                start = process_time()
                loaded_model.fit(loaded_data.X_train, loaded_data.y_train, **fit_params)
            elif "Scikitlearn" and  "loss_gradient" in str(e):
                from art.estimators.classification.scikitlearn import \
                    ScikitlearnSVC
                if hasattr(loaded_model, "steps"):
                    loaded_model = loaded_model.steps[-1][-1]
                if hasattr(loaded_model, "model"):
                    loaded_model = ScikitlearnSVC(model=loaded_model.model, clip_values=(0, 1))
                else:
                    loaded_model = ScikitlearnSVC(model=model, clip_values=(0, 1))
                start = process_time()
                loaded_model.fit(loaded_data.X_train, loaded_data.y_train, **fit_params)
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
            data = data.load(self.files["data_file"])
        if isinstance(model, Model):
            model = model.load(self.files["model_file"], art=art)
        predictions = model.predict(data.X_test)
        if len(predictions.shape) > 1:
            predictions = predictions.argmax(axis=1)
        result = process_time() - start
        return predictions, result / len(data.X_test)

    def predict_proba(self, data: dict, model: object, art) -> tuple:
        """
        Predicts data with model.
        :param data: dict, data to predict.
        :param model: object, model to predict with.
        :returns: tuple, (predictions, predict_time).
        """
        start = process_time()
        if isinstance(data, Data):
            data = data.load(self.files["data_file"])
        if isinstance(model, Model):
            model = model.load(self.files["model_file"], art=art)
        try:
            predictions = model.predict_proba(data.X_test)
        except:  # noqa E722
            predictions = model.predict(data.X_test)
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
        probabilities: np.ndarray = None,
        save_data = True,
        save_model = False,
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
        if "data_file" in self.files and data is not None and save_data is True:
            logger.info("Saving data...")
            files.update({"data": self.save_data(data)})
            assert Path(files['data']).exists(), f"Data file {files['data']} does not exist."
        if "model_file" in self.files and model is not None and save_model is True:
            logger.info("Saving model...")
            files.update({"model": self.save_model(model)})
            assert Path(files['model']).exists(), f"Model file {files['model']} does not exist."
        if "params_file" in self.files:
            logger.info("Saving parameters...")
            files.update({"params": self.save_params()})
            assert Path(files['params']).exists(), f"Params file {files['params']} does not exist."
        if "ground_truth_file" in self.files and ground_truth is not None:
            logger.info("Saving ground truth...")
            files.update({"ground_truth": self.save_ground_truth(ground_truth)})
            assert Path(files['ground_truth']).exists(), f"Ground truth file {files['ground_truth']} does not exist."
        if "predictions_file" in self.files and predictions is not None:
            logger.info("Saving predictions...")
            files.update({"predictions": self.save_predictions(predictions)})
            assert Path(files['predictions']).exists(), f"Predictions file {files['predictions']} does not exist."
        if "probabilities_file" in self.files and probabilities is not None:
            logger.info("Saving probabilities...")
            files.update({"probabilities": self.save_probabilities(predictions)})
            assert Path(files['probabilities']).exists(), f"Probabilities file {files['probabilities']} does not exist."
        if "time_dict_file" in self.files and time_dict is not None:
            logger.info("Saving time dictionary...")
            time_file = self.save_time_dict(time_dict)
            files.update({"time": time_file})
            assert Path(time_file).exists(), f"Time file {time_file} does not exist."
        if "score_dict_file" in self.files and score_dict is not None:
            logger.info("Saving score dictionary...")
            files.update({"scores": self.save_scores(score_dict)})
            assert Path(files['scores']).exists(), f"Scores file {files['scores']} does not exist."
        return files

    def run(
        self,
        art=True,
        save_data = False,
        save_model = False,
    ) -> dict:
        """
        Runs experiment and saves results according to config file.
        """
        #######################################################################
        logger.info("Parsing Config File")
        data, model, files = self.load()
        results = {}
        time_dict = {}
        outs = {}
        #######################################################################
        # Fit model, if applicable
        if isinstance(data, Namespace) :
            loaded_data = data
        else:
            loaded_data = data.load(self.files["data_file"])
        if isinstance(model, Model) and len(self.model) > 0:
            logger.info("Fitting model")
            loaded_data, fitted_model, fit_time = self.fit(data, model, art=art)
            time_dict.update({"fit_time": fit_time})
            results = {
                "data": loaded_data,
                "model": fitted_model,
                "ground_truth": loaded_data.y_test,
            }
            results["time_dict"] = time_dict
        elif hasattr(model, "fit") and "ground_truth_file" in files and isinstance(files['ground_truth_file'], str) and len(self.model) > 0:
            logger.info("Fitting model")
            fitted_model = model
            results = {
                "data": loaded_data,
                "model": fitted_model,
                "ground_truth": loaded_data.y_test,
            }
            results["time_dict"] = time_dict
        elif hasattr(model, "fit") and "ground_truth_file" in files and not isinstance(files['ground_truth_file'], str) and len(self.model) > 0:
            logger.info("Model already fitted.")
            time_dict = files['time_dict_file'] if "time_dict_file" in files else {}
            fitted_model = model
            results = {
                "data": loaded_data,
                "model": fitted_model,
                "ground_truth": files['ground_truth_file'],
                "time_dict": time_dict,
            }
        # only true if model not specificed in config
        elif isinstance(model, dict) and len(self.model) == 0:
            logger.info("No model specified. Skipping model fitting.")
            results = {"data": loaded_data, "ground_truth": loaded_data.y_test}
            results["time_dict"] = time_dict
        #######################################################################
        if "predictions_file" in files and isinstance(files['predictions_file'], str) and len(self.model) > 0:
            logger.info("Predicting")
            try:
                predictions, predict_time = self.predict(loaded_data, fitted_model, art=art)
            except NotFittedError or ValueError:
                logger.warning("Model not fitted. Fitting model.")
                loaded_data, fitted_model, fit_time = self.fit(data, model, art=art)
                predictions, predict_time = self.predict(loaded_data, fitted_model, art=art)
                time_dict.update({"fit_time": fit_time})
            time_dict.update({"predict_time": predict_time})
            results.update({"predictions": predictions})
            results["time_dict"].update(time_dict)
        elif "predictions_file" in files and not isinstance(files['predictions_file'], str) and len(self.model) > 0:
            logger.info("Predictions already made.")
            predictions = files['predictions_file']
            results.update({"predictions": files['predictions_file']})
            time_dict = files['time_dict_file'] if "time_dict_file" in files else {}
            results["time_dict"].update(time_dict)
        #######################################################################
        if "probabilities_file" in files and isinstance(files['probabilities_file'], str) and len(self.model) > 0:
            logger.info("Predicting probabilities")
            probabilities, proba_time = self.predict_proba(
                loaded_data,
                fitted_model,
                art=art,
            )
            results.update({"probabilities": probabilities})
            time_dict.update({"proba_time": proba_time})
        elif "probabilities_file" in files and not isinstance(files['probabilities_file'], str):
            logger.info("Probabilities already made.")
            results.update({"probabilities": files['probabilities_file']})
            probabilities = files['probabilities_file']
            time_dict = files['time_dict_file'] if "time_dict_file" in files else {}
            results["time_dict"].update(time_dict)
        #######################################################################
        if "score_dict_file" in files and isinstance(files['score_dict_file'], str) and len(self.model) > 0:
            logger.info("Scoring")
            score_dict = self.score(loaded_data.y_test, predictions)
            results.update({"score_dict": score_dict})
        elif "score_dict_file" in files and not isinstance(files['score_dict_file'], str):
            logger.info("Scores already made.")
            results.update({"score_dict": files['score_dict_file']})
            score_dict = files['score_dict_file']
            time_dict = files['time_dict_file'] if "time_dict_file" in files else {}
            results['time_dict'].update(time_dict)
        #######################################################################
        if "attack_samples_file" in files and isinstance(files['attack_samples_file'], str) and len(self.attack) > 0:
            attack = "!Attack:\n" + str(self._asdict())
            yaml.add_constructor("!Attack:", Attack)
            attack = yaml.load(attack, Loader=yaml.FullLoader)
            targeted = (
                attack._asdict()['attack']["init"]["targeted"]
                if "targeted" in attack._asdict()['attack']["init"]
                else False
            )
            attack_results = attack.run_attack(
                data=loaded_data,
                model=fitted_model,
                targeted=targeted,
            )
            outs.update(attack_results)
            loaded_data.X_test = pd.read_json(attack_results["attack_samples"])
        elif "attack_samples_file" in files and not isinstance(files['attack_samples_file'], str):
            attack_kwargs = [k for k in files.keys() if "attack_" in k]
            attack_results = {}
            rerun = False
            logger.info("Attack samples already made. Checking if attack needs to be rerun.")
            for kwarg in attack_kwargs:
                if isinstance(files[kwarg], str) and Path(files[kwarg]).exists():
                    if Path(files[kwarg]).suffix == ".json":
                        attack_results.update({kwarg: pd.read_json(files[kwarg])})
                    elif Path(files[kwarg]).suffix == ".csv":
                        attack_results.update({kwarg: pd.read_csv(files[kwarg])})
                    elif Path(files[kwarg]).suffix == ".pkl":
                        with open(Path(files[kwarg]), "rb") as f:
                            attack_results.update({kwarg: pickle.load(f)})
                    elif Path(files[kwarg]).suffix == '.yaml':
                        attack_results.update({kwarg: yaml.load(files[kwarg], Loader=yaml.FullLoader)})
                    else:
                        raise NotImplementedError(f"File type {Path(files[kwarg]).suffix} not implemented.")
                elif isinstance(files[kwarg], str) and kwarg not in ['data_file', 'model_file', 'attack_file']:
                    rerun = True
                    break
                elif isinstance(files[kwarg], str) and kwarg in ['data_file', 'model_file', 'attack_file']:
                    pass          
            if rerun:
                logger.info("Rerunning attack.")
                attack = "!Attack:\n" + str(self._asdict())
                yaml.add_constructor("!Attack:", Attack)
                attack = yaml.load(attack, Loader=yaml.FullLoader)
                targeted = (
                    attack._asdict()['attack']["init"]["targeted"]
                    if "targeted" in attack._asdict()['attack']["init"]
                    else False
                )
                attack_results = attack.run_attack(
                    data=loaded_data,
                    model=fitted_model,
                    targeted=targeted,
                )
                outs.update(attack_results)
                loaded_data.X_test = pd.read_json(attack_results["attack_samples"])
                self['files']['data_file']  = self['files']['attack_file']
        
        saved_files = self.save(**results, save_data=save_data, save_model=save_model)
        outs.update(saved_files)
        #######################################################################
        if len(self.plots) > 0 and len(self.model) > 0: # and not isinstance(files['plots_folder'], str):
            logger.info("Visualising")
            if "art" in str(type(fitted_model)):
                art = True
            else:
                art = False
            plots = self.visualise(data=loaded_data, model=fitted_model, art=art)
            outs.update({"plots": plots})
            logger.info("Visualising")
        logger.info(f"Experiment {self.files['path']} finished.")
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

    def score(self, ground_truth = None, predictions = None) -> List[Path]:
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
        attack_file: /tmp/attack.pkl
        attack_predictions_file: attack_predictions.json
        attack_probabilities_file: attack_probabilities.json
        attack_samples_file: samples.json
        attack_score_dict_file: attack_scores.json
        attack_time_dict_file: attack_time_dict.json
        data_file: /tmp/data.pkl
        model_file: /tmp/model.pkl
        params_file: params.yaml
        path : /tmp/reports/deckard_test
        predictions_file: predictions.json
        probabilities_file: probabilities.json
        reports: /tmp/reports
        score_dict_file: scores.json
        time_dict_file: time_dict.json
        ground_truth_file: ground_truth.json
        attack_params_file: attack_params.json
    """
