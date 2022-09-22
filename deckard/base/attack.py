import logging
import os
from pathlib import Path
from time import process_time
from typing import Union

from .parse import generate_tuple_from_yml, generate_object_from_tuple
from pandas import DataFrame

from .data import Data
from .experiment import Experiment
from .hashable import my_hash
from .model import Model

ART_NUMPY_DTYPE = "float32"

logger = logging.getLogger(__name__)


class AttackExperiment(Experiment):
    """ """

    def __init__(
        self,
        data: Data,
        model: Model,
        attack: Union[str, Path, dict],
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
        assert isinstance(
            attack,
            (dict, str, Path),
        ), "Attack must be a dictionary, str, or path. It is type {}".format(
            type(attack),
        )
        assert "name" and "params" in attack
        super().__init__(
            data=data,
            model=model,
            is_fitted=is_fitted,
            fit_params=fit_params,
            predict_params=predict_params,
        )
        config_tuple = generate_tuple_from_yml(attack)
        if "Attack" not in self.params:
            self.params["Attack"] = {}
        id_ = (
            my_hash(config_tuple)
            if isinstance(attack, dict)
            else Path(attack).name.split(".")[0]
        )
        try:
            attack = generate_object_from_tuple(config_tuple)
        except TypeError as e:
            if "classifier" or "estimator" in str(e):
                attack = generate_object_from_tuple(config_tuple, self.model.model)
        id_ = my_hash(config_tuple)
        self.params["Attack"][id_] = {
            "name": config_tuple[0],
            "params": config_tuple[1],
        }
        self.attack = attack

    def __call__(
        self,
        path,
        model_file: Union[str, Path] = "model",
        prefix=None,
        predictions_file: Union[str, Path] = "predictions.json",
        ground_truth_file: Union[str, Path] = "ground_truth.json",
        time_dict_file: Union[str, Path] = "time_dict.json",
        params_file: Union[str, Path] = "params.json",
        attack_samples_file: Union[str, Path] = "attack_samples.json",
        attack_prefix="attack",
        generate_params: dict = None,
        benign_prefix=None,
    ) -> list:
        """
        Runs attack.
        """
        prefix = attack_prefix
        files = super().__call__(
            path,
            model_file,
            benign_prefix,
            predictions_file,
            ground_truth_file,
            time_dict_file,
            params_file,
        )
        if generate_params is not None:
            self.run_attack(**generate_params)
        else:
            self.run_attack()
        assert hasattr(
            self,
            "adv",
        ), "Attack does not have attribute adv. Something went wrong."
        assert hasattr(
            self,
            "adv_samples",
        ), "Attack does not have attribute adv_samples. Something went wrong."
        assert hasattr(
            self,
            "time_dict",
        ), "Attack does not have attribute time_dict. Something went wrong."

        pred_file = self.save_attack_predictions(prefix=prefix, path=path)
        sampl_file = self.save_attack_samples(prefix=prefix, path=path)
        files.extend([pred_file, sampl_file])
        return files

    def run_attack(self, targeted: bool = False, **kwargs) -> None:
        """
        Runs the attack on the model
        """
        if not hasattr(self, "time_dict") or self.time_dict is None:
            self.time_dict = {}
        assert hasattr(self, "attack"), "Attack not set"
        start = process_time()
        if "AdversarialPatch" in str(type(self.attack)):
            patches, masks = self.attack.generate(
                self.data.X_test, self.data.y_test, **kwargs
            )
            adv_samples = self.attack.apply_patch(
                self.data.X_test,
                scale=self.attack._attack.scale_max,
            )
        elif targeted is False:
            adv_samples = self.attack.generate(self.data.X_test, **kwargs)
        else:
            adv_samples = self.attack.generate(
                self.data.X_test, self.data.y_test, **kwargs
            )
        end = process_time()
        self.time_dict.update({"adv_fit_time:": end - start})
        start = process_time()
        adv = self.model.model.predict(adv_samples)
        end = process_time()
        self.adv = adv
        self.adv_samples = adv_samples
        self.time_dict.update({"adv_pred_time": end - start})
        return None

    def get_attack(self):
        """
        Returns the attack from an experiment
        :param experiment: experiment to get attack from
        """
        return self.attack

    def save_attack_samples(
        self,
        prefix=None,
        filename: str = "examples.json",
        path: str = ".",
    ):
        """
        Saves adversarial examples to specified file.
        :param filename: str, name of file to save adversarial examples to.
        :param path: str, path to folder to save adversarial examples. If none specified, examples are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        assert hasattr(self, "adv_samples"), "No adversarial samples to save"
        if prefix is not None:
            filename = prefix + "_" + filename
        adv_file = os.path.join(path, filename)
        adv_results = DataFrame(self.adv_samples.reshape(self.adv_samples.shape[0], -1))
        adv_results.to_json(adv_file)
        assert os.path.exists(adv_file), "Adversarial example file not saved"
        return adv_file

    def save_attack_predictions(
        self,
        prefix=None,
        filename: str = "predictions.json",
        path: str = ".",
    ) -> None:
        """
        Saves adversarial predictions to specified file.
        :param filename: str, name of file to save adversarial predictions to.
        :param path: str, path to folder to save adversarial predictions. If none specified, predictions are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        if prefix is not None:
            filename = prefix + "_" + filename
        adv_file = os.path.join(path, filename)
        adv_results = DataFrame(self.adv)
        adv_results.to_json(adv_file)
        assert os.path.exists(adv_file), "Adversarial example file not saved"
        return adv_file
