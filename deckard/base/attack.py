import collections
import logging
from pathlib import Path
from time import process_time
from typing import Callable, List
from argparse import Namespace
import json
import numpy as np
import yaml
from copy import deepcopy
from pandas import DataFrame, Series

from .hashable import BaseHashable, my_hash
from .utils import factory
from .model import Model


ART_NUMPY_DTYPE = "float32"

logger = logging.getLogger(__name__)


class Attack(
    collections.namedtuple(
        typename="Attack",
        field_names="data, model, attack, scorers, files, plots",
        defaults=({},),
    ),
    BaseHashable,
):
    def __new__(cls, loader, node):
        return super().__new__(cls, **loader.construct_mapping(node))

    def load(self, model) -> tuple:
        """
        Loads data, model from the config file.
        :param config: str, path to config file.
        :returns: tuple(object, dict), (attck, generate).
        """
        params = deepcopy(dict(self._asdict()))
        name = params["attack"]["init"].pop("name")
        try:
            attack = factory(name, args=[model], **params["attack"]["init"])
        except ValueError as e:
            attack = factory(name, **params)
        except Exception as e:
            raise e
        generate = params.pop("generate", {})
        return (attack, generate)

    def fit(self, data, model, targeted=False):
        time_dict = {}
        start = process_time()
        if "X_test" not in vars(data):
            data = data.load()
        if isinstance(model, BaseHashable):
            model = model.load(art=True)
        assert hasattr(model, "fit"), "Model must have a fit method."
        attack, gen = self.load(model)
        if targeted is False:
            start = process_time()
            attack_samples = attack.generate(data.X_test, **gen)
        else:
            start = process_time()
            attack_samples = attack.generate(data.X_test, data.y_test, **gen)
        attack_pred = model.model.predict(attack_samples)
        end = process_time()
        time_dict.update({"attack_pred_time": end - start})
        return attack_samples, attack_pred, time_dict

    def run_attack(self, data, model, mtype: str = None, targeted=False):
        attack_samples, attack_pred, time_dict = self.fit(data, model, targeted)
        results = {
            "samples": attack_samples,
            "predictions": attack_pred,
            "time_dict": time_dict,
        }
        outs = self.save(**results)
        return outs

    def save_attack_time(
        self,
        time_dict: dict,
    ) -> Path:
        """
        Saves the time dictionary to a json file.
        """
        filename = Path(self.files["path"], self.files["attack_time_dict_file"])
        filename.parent.mkdir(parents=True, exist_ok=True)
        Series(time_dict).to_json(filename)
        assert Path(filename).exists(), f"File {filename} not saved."
        return Path(filename).resolve()

    def save_attack_params(self) -> Path:
        """
        Saves the attack parameters to a json file.
        """
        filename = Path(self.files["path"], self.files["attack_params_file"])
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as f:
            json.dump(self._asdict(), f)
        assert Path(filename).exists(), f"File {filename} not saved."
        return Path(filename).resolve()

    def save_attack_samples(
        self,
        samples: np.ndarray,
    ) -> Path:
        """
        Saves adversarial examples to specified file.
        """
        filename = Path(self.files["path"], self.files["attack_samples_file"])
        filename.parent.mkdir(parents=True, exist_ok=True)
        attack_results = DataFrame(samples.reshape(samples.shape[0], -1))
        attack_results.to_json(filename)
        assert Path(filename).exists(), "Adversarial example file not saved"
        return Path(filename).resolve()

    def save_attack_predictions(
        self,
        predictions: np.ndarray,
    ) -> Path:
        """
        Saves adversarial predictions to specified file.
        """
        filename = Path(self.files["path"], self.files["attack_predictions_file"])
        filename.parent.mkdir(parents=True, exist_ok=True)
        attack_results = DataFrame(predictions)
        attack_results.to_json(filename)
        assert Path(filename).exists(), "Adversarial example file not saved"
        return Path(filename).resolve()

    def save_attack_scores(
        self,
        score_dict: dict,
    ) -> Path:
        """Saves adversarial results to specified file.

        :param score_dict (dict): Dictionary of scores.

        :return path: Path to saved file.
        """
        filename = Path(self.files["path"], self.files["attack_scores_file"])
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as f:
            json.dump(score_dict, f)
        assert Path(filename).exists(), "Adversarial example file not saved"
        return Path(filename).resolve()

    def save(
        self,
        data: Namespace = None,
        model: Callable = None,
        score_dict: dict = None,
        predictions: dict = None,
        samples: dict = None,
        time_dict: dict = None,
    ) -> List[Path]:
        """
        Saves the attack result as specified in the config file.
        :param data: Namespace, data object.
        :param model: object, model object.
        :param score_dict: dict, dictionary of scores.
        :param predictions: dict, dictionary of predictions.
        :param samples: dict, dictionary of adversarial samples.
        :param time_dict: dict, dictionary of times.
        """
        outs = {}
        if data is not None:
            raise NotImplementedError("Saving data not implemented yet.")
        if model is not None:
            raise NotImplementedError("Saving model not implemented yet.")
        if score_dict is not None:
            file = self.save_attack_scores(score_dict)
            outs.update({"scores": file})
        if predictions is not None:
            file = self.save_attack_predictions(predictions)
            outs.update({"attack_predictions": file})
        if samples is not None:
            file = self.save_attack_samples(samples)
            outs.update({"attack_samples": file})
        if time_dict is not None:
            file = self.save_attack_time(time_dict)
            outs.update({"attack_time": file})
        return outs
