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
from parse import generate_object_from_tuple
import numpy as np 
import yaml
from copy import deepcopy 

ART_NUMPY_DTYPE = "float32"

logger = logging.getLogger(__name__)


class Attack(
    collections.namedtuple(
        typename="Attack",
        field_names="init, generate, files ",
        defaults=({},{}, {}),
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
    #     path = Path(self.files['path'], my_hash(self._asdict()))
    #     adv_pred, adv_samples, time_dict = self.run_attack(data, model, attack, **generate_params)
    #     predictions = Path(path, predictions_file)
    #     samples = Path(path, attack_samples_file)
    #     times = Path(path, time_dict_file)
    #     pred_file = self.save_attack_predictions(predictions = adv_pred, filename = predictions)
    #     sampl_file = self.save_attack_samples(samples = adv_samples, filename = samples)
    #     time_file = self.save_time_dict(time_dict = time_dict, filename = times)
    #     files.extend([pred_file, sampl_file, time_file])
    #     
    #     return files

    def load(self, model) -> tuple:
        """
        Loads data, model from the config file.
        :param config: str, path to config file.
        :returns: tuple(dict, object), (data, model).
        """
        params = deepcopy(self._asdict())

        if params["init"] is not {}:
            name = params['init'].pop("name")
            params = params['init']
            try:
                attack = generate_object_from_tuple((name, params), model)
            except ValueError as e:
                attack = generate_object_from_tuple((name, params))
            except Exception as e:
                raise e
        else:
            raise ValueError("Attack not specified correctly in config file.")
        generate = params.pop("generate", {})
        params.pop("attack", None)
        files = params.pop("files", None)
        return attack, generate, files


    def run_attack(self, data, model, attack, targeted: bool = False, **kwargs) -> None:
        """
        Runs the attack on the model
        """
        time_dict = {}
        assert hasattr(self, "attack"), "Attack not set"
        start = process_time()
        if targeted is False:
            adv_samples = attack.generate(data.X_test, **kwargs)
        else:
            adv_samples = attack.generate(
                data.X_test, data.y_test, **kwargs
            )
        end = process_time()
        time_dict.update({"adv_fit_time:": end - start})
        start = process_time()
        adv_pred = model.model.predict(adv_samples)
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
    import pickle
    config = """
    init:
        name: art.attacks.evasion.HopSkipJump
        max_iter : 1000
        init_eval : 1000
        init_size : 10
    files:
        adv_samples: adv_samples.json
        adv_predictions : adv_predictions.json
        adv_time_dict : adv_time_dict.json
        attack_params : attack_params.json

    """
    from visualise import Yellowbrick_Visualiser
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    yaml.add_constructor("!Attack:", Attack)
    attack = yaml.load("!Attack:\n" + str(config), Loader=yaml.Loader)
    with open("/workspaces/deckard/deckard/base/model/2db00e44d0b930b24d549ef1307f177a.pickle", "rb") as f:
        model = pickle.load(f)
    with open("/workspaces/deckard/deckard/base/data/fdf009456bdd8bc7a3db8c2785157ef3 copy.pickle", "rb") as f:
        data = pickle.load(f)
    from art.estimators.classification.scikitlearn import ScikitlearnClassifier
    model = ScikitlearnClassifier(model)
    loaded_attack, generate, files = attack.load(model)
    adv_pred, adv_samples, time_dict = attack.run_attack(data, model, loaded_attack, **generate)
