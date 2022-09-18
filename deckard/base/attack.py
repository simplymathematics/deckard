import os, logging
from pandas import Series, DataFrame
from time import process_time
from json import dumps
from hashlib import md5 as my_hash
from .data import Data
from .model import Model
from .experiment import Experiment
from .hashable import BaseHashable
from typing import Callable

ART_NUMPY_DTYPE = "float32"

logger = logging.getLogger(__name__)


class AttackExperiment(Experiment):
    """ """

    def __init__(
        self,
        data: Data,
        model: Model,
        attack: Callable,
        verbose: int = 1,
        name: str = None,
        is_fitted: bool = False,
        filename: str = None,
        model_type="sklearn",
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
        self.model = model
        self.model_type = model_type
        self.data = data
        self.name = (
            self.data.dataset + "_" + str(hash(model)) + "_" + str(hash(attack))
            if name is None
            else name
        )
        self.verbose = verbose
        self.is_fitted = is_fitted
        self.predictions = None
        self.ground_truth = None
        self.time_dict = None
        self.params = dict()
        self.params["Model"] = dict(model.params)
        self.params["Data"] = dict(data)
        self.params["Attack"] = {"name": str(type(attack)), "params": attack.__dict__}
        self.attack = attack
        self.attack = attack
        self.hash = hash(self)
        self.params["Experiment"] = {
            "name": self.hash,
            "verbose": self.verbose,
            "is_fitted": self.is_fitted,
            "id": self.hash,
            "attack": hash(str(self.params["Attack"])),
            "model": hash(model),
            "data": hash(data),
        }

    def __call__(self, path, prefix=None, **kwargs):
        """
        Runs attack.
        """
        assert hasattr(self, "attack")
        if not os.path.isdir(path):
            os.mkdir(path)
        self.run_attack(**kwargs)
        self.save_params(prefix=prefix, path=path)
        self.save_time_dict(prefix=prefix, path=path)
        self.save_predictions(prefix=prefix, path=path)
        self.save_ground_truth(prefix=prefix, path=path)
        self.save_attack_results(prefix=prefix, path=path)
        return None

    def run_attack(self, targeted: bool = False, **kwargs) -> None:
        """
        Runs the attack on the model
        """
        if not hasattr(self, "time_dict") or self.time_dict is None:
            self.time_dict = dict()
        assert hasattr(self, "attack"), "Attack not set"
        start = process_time()
        if "AdversarialPatch" in str(type(self.attack)):
            patches, masks = self.attack.generate(
                self.data.X_test, self.data.y_test, **kwargs
            )
            adv_samples = self.attack.apply_patch(
                self.data.X_test, scale=self.attack._attack.scale_max
            )
        elif targeted == False:
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

    def set_attack(self, attack: object) -> None:
        """
        Adds an attack to an experiment
        :param experiment: experiment to add attack to
        :param attack: attack to add
        """
        attack_params = {}
        for key, value in attack.__dict__.items():
            if isinstance(value, int):
                attack_params[key] = value
            elif isinstance(value, float):
                attack_params[key] = value
            elif isinstance(value, str):
                attack_params[key] = value
            elif isinstance(value, list):
                attack_params[key] = value
            elif isinstance(value, dict):
                attack_params[key] = str(value)
            elif isinstance(value, tuple):
                attack_params[key] = value
            elif isinstance(value, type(None)):
                attack_params[key] = None
            else:
                attack_params[key] = str(type(value))
        assert isinstance(attack, object)
        self.params["Attack"] = {"name": str(type(attack)), "params": attack_params}
        return attack

    def get_attack(self):
        """
        Returns the attack from an experiment
        :param experiment: experiment to get attack from
        """
        return self.attack

    def save_attack_results(self, prefix=None, path: str = ".") -> None:
        """
        Saves all data to specified folder, using default filenames.
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        # if hasattr(self, "adv_scores"):
        #     self.save_adv_scores(path = path)
        if hasattr(self, "adv"):
            self.save_attack_predictions(path=path)
        if hasattr(self, "adv_samples"):
            self.save_attack_samples(path=path)
        if hasattr(self, "time_dict"):
            self.save_time_dict(path=path, filename="time_dict.json")
        if hasattr(self.model.model, "cv_results_"):
            self.save_cv_scores(path=path)
            self.save_attack_samples(path=path)
        return None

    def save_attack_samples(self, filename: str = "examples.json", path: str = "."):
        """
        Saves adversarial examples to specified file.
        :param filename: str, name of file to save adversarial examples to.
        :param path: str, path to folder to save adversarial examples. If none specified, examples are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        assert hasattr(self, "adv_samples"), "No adversarial samples to save"
        adv_file = os.path.join(path, filename)
        adv_results = DataFrame(self.adv_samples.reshape(self.adv_samples.shape[0], -1))
        adv_results.to_json(adv_file)
        assert os.path.exists(adv_file), "Adversarial example file not saved"
        return None

    def save_attack_predictions(
        self, filename: str = "predictions.json", path: str = "."
    ) -> None:
        """
        Saves adversarial predictions to specified file.
        :param filename: str, name of file to save adversarial predictions to.
        :param path: str, path to folder to save adversarial predictions. If none specified, predictions are saved in current working directory. Must exist.
        """
        assert os.path.isdir(path), "Path to experiment does not exist"
        adv_file = os.path.join(path, filename)
        adv_results = DataFrame(self.adv)
        adv_results.to_json(adv_file)
        assert os.path.exists(adv_file), "Adversarial example file not saved"
        return None
