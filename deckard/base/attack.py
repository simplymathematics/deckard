import collections
import logging
from pathlib import Path
from time import process_time
from typing import Union, Callable
import json
import numpy as np 
import yaml
from copy import deepcopy 
from pandas import DataFrame, Series

from .data import Data
from .experiment import Experiment
from .hashable import BaseHashable, my_hash
from .model import Model
from .utils import factory





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
                attack = factory(name, args = [model], **params)    
            except ValueError as e:
                attack = factory(name, **params)
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
        start = process_time()
        if not isinstance(attack.generate, Callable):
            attack, _, _ = attack.load(model)
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
        adv_samples = adv_samples
        time_dict.update({"adv_pred_time": end - start})
        return adv_pred, adv_samples, time_dict

    def save_attack_time(
        self, time_dict: dict, filename: Union[str, Path] = "time_dict.json"
    )-> Path:
        """
        Saves the time dictionary to a json file.
        """
        path = Path(filename).parent
        file = Path(filename).name
        filename = Path(
            path,
            my_hash(self._asdict()),
            file,
        )
        filename.parent.mkdir(parents=True, exist_ok=True)
        Series(time_dict).to_json(filename)
        assert Path(filename).exists(), f"File {filename} not saved."
        return Path(filename).resolve()
    
    def save_attack_params(self, filename:Union[str, Path])-> Path:
        """
        Saves the attack parameters to a json file.
        """
        path = Path(filename).parent
        file = Path(filename).name
        filename = Path(
            path,
            my_hash(self._asdict()),
            file,
        )
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as f:
            json.dump(self._asdict(), f)
        assert Path(filename).exists(), f"File {filename} not saved."
        return Path(filename).resolve()
                
    def save_attack_samples(
        self,
        samples: np.ndarray, 
        filename: str,
    ) -> Path:
        """
        Saves adversarial examples to specified file.
        :param filename: str, name of file to save adversarial examples to.
        :param path: str, path to folder to save adversarial examples. If none specified, examples are saved in current working directory. Must exist.
        """
       
        path = Path(filename).parent
        file = Path(filename).name
        filename = Path(
            path,
            my_hash(self._asdict()),
            file,
        )
        filename.parent.mkdir(parents=True, exist_ok=True)
        adv_results = DataFrame(samples.reshape(samples.shape[0], -1))
        adv_results.to_json(filename)
        assert Path(filename).exists(), "Adversarial example file not saved"
        return Path(filename).resolve()

    def save_attack_predictions(
        self,
        predictions: np.ndarray,
        filename: str,
    ) -> Path:
        """
        Saves adversarial predictions to specified file.
        :param filename: str, name of file to save adversarial predictions to.
        :param path: str, path to folder to save adversarial predictions. If none specified, predictions are saved in current working directory. Must exist.
        """
        path = Path(filename).parent
        file = Path(filename).name
        filename = Path(
            path,
            my_hash(self._asdict()),
            file,
        )
        filename.parent.mkdir(parents=True, exist_ok=True)
        adv_results = DataFrame(predictions)
        adv_results.to_json(filename)
        assert Path(filename).exists(), "Adversarial example file not saved"
        return Path(filename).resolve()
