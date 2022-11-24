import collections
from pathlib import Path
from time import process_time
import numpy as np
import yaml

# from dvc.api import params_show
# from dvclive import Live
# from tqdm import tqdm
from copy import deepcopy
from data import Data
from model import Model
import pandas as pd
import pickle
from utils import factory

from hashable import BaseHashable, my_hash
import logging
from sklearn.preprocessing import LabelBinarizer

logger = logging.getLogger(__name__)


class Experiment(
    collections.namedtuple(
        typename="Experiment",
        field_names="data, model, scorers, plots, files",
        defaults=({}, {}, {}, {}, {}),
        rename=True,
    ),
    BaseHashable,
):
    def __new__(cls, loader, node):
        return super().__new__(cls, **loader.construct_mapping(node))

    def fit(self, data: dict, model: object) -> tuple:
        """
        Fits model to data.
        :param data: dict, data to fit model to.
        :param model: object, model to fit.
        :returns: tuple, (model, data, fit_time).
        """
        if isinstance(data, Data):
            loaded_data = data.load()
        if isinstance(model, Model):
            loaded_model = model.load()
        start = process_time()
        try:
            result = loaded_model.fit(loaded_data.X_train, loaded_data.y_train)
        except:

            result = loaded_model.fit()
        result = process_time() - start
        return loaded_model, loaded_data, result / len(loaded_data.X_train)

    def predict(self, data: dict, model: object) -> tuple:
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
            model = model.load()
        predictions = model.predict(data.X_test)
        result = process_time() - start
        return predictions, result / len(data.X_test)

    def load(self) -> tuple:
        """
        Loads data, model from the config file.
        :param config: str, path to config file.
        :returns: tuple(dict, object), (data, model).
        """
        params = deepcopy(self._asdict())
        if params["data"] is not {}:
            yaml.add_constructor("!Data", Data)
            data_document = """!Data\n""" + str(dict(params["data"]))
            data = yaml.load(data_document, Loader=yaml.Loader)

        else:
            raise ValueError("Data not specified in config file")
        if params["model"] is not {}:
            yaml.add_constructor("!Model", Model)
            model_document = """!Model\n""" + str(dict(params["model"]))
            model = yaml.load(model_document, Loader=yaml.Loader)

        else:
            model = {}
        if params["plots"] is not {}:
            from visualise import Yellowbrick_Visualiser
            yaml.add_constructor("!Yellowbrick_Visualiser:", Yellowbrick_Visualiser)
            plots_document = "!Yellowbrick_Visualiser:\n" + str(params)
            vis = yaml.load(plots_document, Loader=yaml.Loader)
        else:
            vis = None
        # print("Inside load")
        # print("Self hash: ", my_hash(params))
        # print("Vis hash: ", my_hash(vis._asdict()))
        # input("Press enter to continue")
        params.pop("data", None)
        params.pop("model", None)
        params.pop("plots", None)
        files = params.pop("files", None)
        return (data, model, files, vis)

    def score(self, ground_truth: np.ndarray, predictions: np.ndarray) -> dict:
        """Scores predictions according to self.scorers.
        :param ground_truth: np.ndarray, ground truth.
        :param predictions: np.ndarray, predictions to score.
        :returns: dict, dictionary of scores.
        """
        score_dict = {}
        scorers = deepcopy(self.scorers)
        for scorer in self.scorers:
            class_name = scorers[scorer].pop("name")
            params = scorers[scorer]
            params.update({"y_true": ground_truth, "y_pred": predictions})
            score = factory(class_name, **params)
            score_dict[scorer] = score
        return score_dict

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
        return Path(filename).resolve()

    def save_params(self, filename: str) -> Path:
        """Saves parameters to specified file.
        :param filename: str, name of file to save parameters to.
        :returns: Path, path to saved parameters.
        """
        params = deepcopy(self._asdict())
        files = params.pop("files", None)
        new_files = {}
        if files is not None:
            path = files.pop("path", ".")
            path = Path(path)
            for x in files:
                x = Path(path, my_hash(self._asdict()), files[x])
                new_files[x] = x.relative_to(path.parent).as_posix()

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
        params["scorer"] = list(params.pop("scorers").keys())[0]
        params.pop("plots", None)
        path.mkdir(parents=True, exist_ok=True)
        pd.Series(params).to_json(filename)
        return Path(filename).resolve()

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
            if file.endswith(".pickle"):
                model.save(Path(filename).stem, path=path)
            model.save(filename=file, path=path)
        else:
            with open(filename, "wb") as f:
                pickle.dump(model, f)
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
        pd.Series(predictions).to_json(filename)
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
        ground_truth: np.ndarray = None,
        model: object = None,
        params_file: str = None,
        score_dict: dict = None,
        time_dict: dict = None,
        predictions: np.ndarray = None,
        time_dict_file: str = None,
        predictions_file: str = None,
        ground_truth_file: str = None,
        score_dict_file: str = None,
        path: Path = None,
    ) -> Path:
        """
        Saves data, model, parameters, predictions, scores, and time dictionary.
        :param data: dict, data to save.
        :param ground_truth: np.ndarray, ground truth to save.
        :param model: object, model to save.
        :param params: dict, parameters to save.
        :param params_file: str, name of file to save parameters to.
        :param score_dict: dict, scores to save.
        :param time_dict: dict, time dictionary to save.
        :param predictions: np.ndarray, predictions to save.
        :param time_dict_file: str, name of file to save time dictionary to.
        :param predictions_file: str, name of file to save predictions to.
        :param ground_truth_file: str, name of file to save ground truth to.
        :param score_dict_file: str, name of file to save scores to.
        :param path: Path, path to save files to. Defaults to current directory.
        """
        files = []
        path = Path(path, str(my_hash(self._asdict())))
        path.mkdir(parents=True, exist_ok=True)
        if score_dict_file is not None:
            score_dict_file = path / score_dict_file
            files.append(self.save_scores(score_dict, filename=score_dict_file))
        if "files" in self.data:
            files.append(self.save_data(data))
        if "files" in self.model:
            files.append(self.save_model(model))
        if params_file is not None:
            params_file = Path(path, params_file)
            files.append(self.save_params(params_file))
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

    def run(self, is_fitted=False):
        """
        Runs experiment and saves results according to config file.
        """
        logger.info("Parsing Config File")
        
    
        data, model, files, vis = self.load()
        params = deepcopy(self._asdict())
        path = Path(params["files"]["path"], str(my_hash(self._asdict())))
        if path.exists():
            logger.warning(
                f"Path {path} already exists. Will overwrite any files specified in the config.",
            )
        else:
            path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving to {path}")
        assert isinstance(data, Data)
        assert isinstance(model, Model)
        if is_fitted is False:
            logger.info("Fitting Model")
            fitted_model, loaded_data, fit_time = self.fit(data, model)
        else:
            logger.info("Model already fitted")
            fitted_model = model.load()
            loaded_data = data.load()
        logger.info("Scoring Model")
        predictions, predict_time = self.predict(loaded_data, fitted_model)
        ground_truth = loaded_data.y_test
        score_dict = self.score(predictions=predictions, ground_truth=ground_truth)
        results = {
            "data": loaded_data,
            "model": fitted_model,
            "predictions": predictions,
            "time_dict": {"fit_time": fit_time, "predict_time": predict_time},
            "ground_truth": ground_truth,
            "score_dict": score_dict,
        }
        logger.info("Saving Results")
        outs = self.save(**results, **files)
        if vis is not None:
            from visualise import Yellowbrick_Visualiser
            pass
            plot_dict = vis.visualise(path=path)
            templating_string = params["plots"].pop(
                "templating_string",
                "{{plot_divs}}",
            )
            output_html = params["plots"].pop("output_html", "report.html")
            template = params["plots"].pop("template", "template.html")
            output_html = Path(path, output_html)
            # outs.append(
            #     vis.render(
            #         plot_dict=plot_dict,
            #         templating_string=templating_string,
            #         output_html=output_html,
            #         template=template,
            #     ),
            # )
        for file in outs:
            assert file.exists(), f"File {file} does not exist."
        return outs


if "__main__" == __name__:

    config = """
    model:
        init:
            loss: "hinge"
            name: sklearn.linear_model.SGDClassifier
        files:
            model_path : model
            model_filetype : pickle
        fit:
            epochs: 1000
            learning_rate: 1.0e-08
            log_interval: 10
    data:
        sample:
            shuffle : True
            random_state : 42
            train_size : 800
            stratify : True
        add_noise:
            train_noise : 1
            time_series : True
        name: classification
        files:
            data_path : data
            data_filetype : pickle
        generate:
            n_samples: 1000
            n_features: 2
            n_informative: 2
            n_redundant : 0
            n_classes: 2
        sklearn_pipeline:
            - sklearn.preprocessing.StandardScaler
        transform:
            sklearn.preprocessing.StandardScaler:
                with_mean : true
                with_std : true
                X_train : true
                X_test : true
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

    """
    from visualise import Yellowbrick_Visualiser
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    yaml.add_constructor("!Experiment:", Experiment)
    experiment = yaml.load("!Experiment:\n" + str(config), Loader=yaml.Loader)
    yaml.add_constructor("!Visualizer:", Yellowbrick_Visualiser)
    visualizer = yaml.load("!Visualizer:\n" + str(config), Loader=yaml.Loader)
    # print("Before: ")
    # print(my_hash(experiment._asdict()))
    # print(my_hash(visualizer._asdict()))
    
    experiment.run()
    # print("After: ")
    # print(my_hash(experiment._asdict()))
    # print(my_hash(visualizer._asdict()))
