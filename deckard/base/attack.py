import collections
import logging
import os
from pathlib import Path
from time import process_time
from typing import Union

from data import Data
from experiment import Experiment
from hashable import BaseHashable, my_hash
from model import Model
from pandas import DataFrame
from utils import factory
import numpy as np 
import yaml
from copy import deepcopy 

ART_NUMPY_DTYPE = "float32"

logger = logging.getLogger(__name__)


class Attack(
    collections.namedtuple(
        typename="Attack",
        field_names="data, model, attack, scorers, plots, files",
        defaults=( {}, {}, {}),
        rename=True,
    ),
    BaseHashable,
):
    def __new__(cls, loader, node):
        return super().__new__(cls, **loader.construct_mapping(node))

    # def __init__(
    #     self,
    #     data: Data,
    #     model: Model,
    #     attack: Union[str, Path, dict],
    #     is_fitted: bool = False,
    #     fit_params: dict = None,
    #     predict_params: dict = None,
    # ):
    #     """
    #     Creates an experiment object
    #     :param data: Data object
    #     :param model: Model object
    #     :param params: Dictionary of other parameters you want to add to this object. Obviously everything in self.__dict__.keys() should be treated as a reserved keyword, however.
    #     :param verbose: Verbosity level
    #     :param scorers: Dictionary of scorers
    #     :param name: Name of experiment
    #     """
    #     assert isinstance(
    #         attack,
    #         (dict, str, Path),
    #     ), "Attack must be a dictionary, str, or path. It is type {}".format(
    #         type(attack),
    #     )
    #     assert "name" and "params" in attack
    #     super().__init__(
    #         data=data,
    #         model=model,
    #         is_fitted=is_fitted,
    #         fit_params=fit_params,
    #         predict_params=predict_params,
    #     )
    #     config_tuple = generate_tuple_from_yml(attack)
    #     if "Attack" not in self.params:
    #         self.params["Attack"] = {}
    #     id_ = (
    #         my_hash(config_tuple)
    #         if isinstance(attack, dict)
    #         else Path(attack).name.split(".")[0]
    #     )
    #     try:
    #         attack = generate_object_from_tuple(config_tuple)
    #     except TypeError as e:
    #         if "classifier" or "estimator" in str(e):
    #             attack = generate_object_from_tuple(config_tuple, self.model.model)
    #     id_ = my_hash(config_tuple)
    #     self.params["Attack"][id_] = {
    #         "name": config_tuple[0],
    #         "params": config_tuple[1],
    #     }
    #     self.attack = attack

    # def run(
    #     self,
    #     predictions_file: Union[str, Path] = "predictions.json",
    #     time_dict_file: Union[str, Path] = "time_dict.json",
    #     attack_samples_file: Union[str, Path] = "attack_samples.json",
    #     generate_params: dict = {},
    # ) -> list:
    #     """
    #     Runs attack.
    #     """
    #     data, model, attack, files, vis = self.load()
    #     data = data.load()
    #     model = model.load()
    #     path = Path(self.files['path'], my_hash(self._asdict()))
    #     adv_pred, adv_samples, time_dict = self.run_attack(data, model, attack, **generate_params)
    #     predictions = Path(path, predictions_file)
    #     samples = Path(path, attack_samples_file)
    #     times = Path(path, time_dict_file)
    #     pred_file = self.save_attack_predictions(predictions = adv_pred, filename = predictions)
    #     sampl_file = self.save_attack_samples(samples = adv_samples, filename = samples)
    #     time_file = self.save_time_dict(time_dict = time_dict, filename = times)
    #     files.extend([pred_file, sampl_file])
    #     if vis is not None:
    #         from visualise import Yellowbrick_Visualiser
    #         plot_dict = vis.visualise(path=path)
    #         templating_string = params["plots"].pop(
    #             "templating_string",
    #             "{{plot_divs}}",
    #         )
    #         output_html = params["plots"].pop("output_html", "report.html")
    #         template = params["plots"].pop("template", "template.html")
    #         output_html = Path(path, output_html)
    #         # outs.append(
    #         #     vis.render(
    #         #         plot_dict=plot_dict,
    #         #         templating_string=templating_string,
    #         #         output_html=output_html,
    #         #         template=template,
    #         #     ),
    #         # )
    #     for file in outs:
    #         assert file.exists(), f"File {file} does not exist."
    #     return outs
    #     return files

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
            raise ValueError("Model not specified in config file")
        data = data.load()
        model = model.load(art = True)
        if params["attack"] is not {}:
            name = params['attack']['init'].pop("name")
            params = params['attack']['init']
            try:
                params.update({"classifier" : model})
                attack = factory(name, **params)
            except ValueError as e:
                params.update({"estimator" : model})
                attack = factory(name, **params)
            except Exception as e:
        else:
            raise ValueError("Attack not specified correctly in config file.")
        params.pop("data", None)
        params.pop("model", None)
        params.pop("plots", None)
        params.pop("attack", None)
        files = params.pop("files", None)
        return (data, model, attack, files)


    def run_attack(self, data, model, attack, targeted: bool = False, **kwargs) -> None:
        """
        Runs the attack on the model
        """
        time_dict = {}
        assert hasattr(self, "attack"), "Attack not set"
        start = process_time()
        if "AdversarialPatch" in str(type(self.attack)):
            patches, masks = self.attack.generate(
                data.X_test,data.y_test, **kwargs
            )
            adv_samples = attack.apply_patch(
                data.X_test,
                scale=attack._attack.scale_max,
            )
        elif targeted is False:
            adv_samples = attack.generate(data.X_test, **kwargs)
        else:
            adv_samples = attack.generate(
                data.X_test, data.y_test, **kwargs
            )
        end = process_time()
        time_dict.update({"adv_fit_time:": end - start})
        start = process_time()
        adv = model.model.predict(adv_samples)
        end = process_time()
        adv = adv
        adv_samples = adv_samples
        time_dict.update({"adv_pred_time": end - start})
        return adv_pred, adv_samples, time_dict


    def save_attack_samples(
        self,
        samples: np.ndarray, 
        filename: str = "examples.json",
    ) -> Path:
        """
        Saves adversarial examples to specified file.
        :param filename: str, name of file to save adversarial examples to.
        :param path: str, path to folder to save adversarial examples. If none specified, examples are saved in current working directory. Must exist.
        """
        adv_results = DataFrame(samples.reshape(samples.shape[0], -1))
        adv_results.to_json(filename)
        assert os.path.exists(filename), "Adversarial example file not saved"
        return filename

    def save_attack_predictions(
        self,
        predictions: np.ndarray,
        filename: str = "predictions.json",
    ) -> Path:
        """
        Saves adversarial predictions to specified file.
        :param filename: str, name of file to save adversarial predictions to.
        :param path: str, path to folder to save adversarial predictions. If none specified, predictions are saved in current working directory. Must exist.
        """
        adv_results = DataFrame(predictions)
        adv_results.to_json(filename)
        assert os.path.exists(filename), "Adversarial example file not saved"
        return filename

if "__main__" == __name__:

    config = """
    model:
        init:
            n_estimators : 100
            name: sklearn.ensemble.RandomForestClassifier
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
    attack:
        init:
            name: art.attacks.evasion.HopSkipJump
            max_iter : 1000
            init_eval : 1000
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

    """
    from visualise import Yellowbrick_Visualiser
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    yaml.add_constructor("!Attack:", Attack)
    experiment = yaml.load("!Attack:\n" + str(config), Loader=yaml.Loader)
    data, model, attack, files = experiment.load()