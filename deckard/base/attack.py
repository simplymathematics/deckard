import collections
import json
import logging
import pickle
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from time import process_time
from typing import Callable, List

import numpy as np
import yaml
from art.estimators.classification.scikitlearn import ScikitlearnSVC
from pandas import DataFrame, Series

from .hashable import BaseHashable
from .scorer import Scorer
from .utils import factory

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

    def load(self, data, model) -> tuple:
        """
        Loads data, model from the config file.
        :param config: str, path to config file.
        :returns: tuple(object, dict), (attck, generate).
        """
        logger.info("Loading attack")
        params = deepcopy(dict(self._asdict()))
        name = params["attack"]["init"].pop("name")
        if hasattr(model, "steps"):
            model = model.steps[-1][1]
        model = ScikitlearnSVC(model, clip_values=(0, 1))
        model.model.fit_status_ = 0
        try:
            attack = factory(name, model, **params["attack"]["init"])
        except ValueError as e:
            logger.warning(
                f"White-box attack failed with error: {e}. Trying black-box.",
            )
            attack = factory(name, **params)
        
        except Exception as e:
            raise e
        generate = params.pop("generate", {})
        return (attack, generate, model)

    def fit(self, data, model, targeted=False):
        logger.info("Fitting attack")
        start = process_time()
        if "X_test" not in vars(data):
            data = data.load()
        attack, gen, model = self.load(data, model)
        if hasattr(data.X_test, "values"):
            data.X_test = data.X_test.values
        if hasattr(data.y_test, "values"):
            data.y_test = data.y_test.values
        start_position = gen.pop("start_position", 0)
        attack_size = gen.pop("attack_size", 100)
        data.X_test = data.X_test[start_position : start_position + attack_size]
        data.y_test = data.y_test[start_position : start_position + attack_size]
        if targeted is False:
            start = process_time()
            attack_samples = attack.generate(data.X_test, **gen)
        else:
            start = process_time()
            attack_samples = attack.generate(data.X_test, data.y_test, **gen)
        end = process_time() - start
        
        return attack_samples, end/attack_size
    
    def predict(self, attack_samples, model):
        logger.info("Predicting attack samples")
        start = process_time()
        attack_pred = model.predict(attack_samples)
        end = process_time() - start
        return attack_pred, end/len(attack_samples)

    def predict_proba(self, attack_samples, model):
        logger.info("Predicting attack samples")
        start = process_time()
        assert hasattr(model, "predict_proba"), "Model must have predict_proba method."
        attack_pred = model.predict_proba(attack_samples)
        end = process_time() - start
        return attack_pred, end/len(attack_samples)
    
    def score(self, ground_truth = None, predictions = None) -> List[Path]:
        """
        :param self: specified in the config file.
        """
        logger.info("Scoring attack")
        yaml.add_constructor("!Scorer:", Scorer)
        scorer = yaml.load("!Scorer:\n" + str(self._asdict()), Loader=yaml.FullLoader)
        score_paths = scorer.score_from_memory(ground_truth, predictions)
        return score_paths
    
    def run_attack(self, data, model, targeted=False):
        logger.info("Running attack")
        score_dict = {}
        attack_samples, fit_time = self.fit(data, model, targeted)
        probabilities, proba_time = self.predict_proba(attack_samples, model)
        predictions, predict_time = self.predict(attack_samples, model)
        time_dict = {
            "attack_fit_time": fit_time,
            "attack_predict_time": predict_time,
            "attack_proba_time": proba_time,
        }
        if "score_dict_file" in self.files:
            logger.info("Scoring")
            score_dict = self.score(data.y_test, predictions)
        for key in score_dict.keys():
            new_key = "attack_" + key
            score_dict[new_key] = score_dict[key]
        results = {
            "attack_samples": attack_samples,
            "attack_probabilities": probabilities,
            "attack_predictions": predictions,
            "attack_time_dict": time_dict,
            "attack_score_dict": score_dict,
        }
        outs = self.save(**results)
        return outs
    
    def save_model(self, model: object) -> Path:
        """Saves model to specified file.
        :model model: object, model to save.
        :returns: Path, path to saved model.
        """
        filename = Path(self.files["model_file"])
        path = Path(filename).parent
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save"):
            model.save(Path(filename).stem, path=path)
        else:
            if hasattr("model", "model"):
                model = model.model
            with open(filename, "wb") as f:
                pickle.dump(model, f)
        return str(Path(filename).as_posix())
    
    def save_attack_time(
        self,
        time_dict: dict,
    ) -> Path:
        """
        Saves the time dictionary to a json file.
        """
        logger.info("Saving attack time")
        filename = Path(self.files["path"], self.files["attack_time_dict_file"])
        filename.parent.mkdir(parents=True, exist_ok=True)
        Series(time_dict).to_json(filename)
        assert Path(filename).exists(), f"File {filename} not saved."
        return str(Path(filename))

    def save_attack_params(self) -> Path:
        """
        Saves the attack parameters to a json file.
        """
        logger.info("Saving attack parameters")
        filename = Path(self.files["path"], self.files["attack_params_file"])
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as f:
            json.dump(self._asdict(), f)
        assert Path(filename).exists(), f"File {filename} not saved."
        return str(Path(filename))

    def save_attack_samples(
        self,
        attack_samples: np.ndarray,
    ) -> Path:
        """
        Saves adversarial examples to specified file.
        """
        logger.info("Saving attack samples")
        filename = Path(self.files["path"], self.files["attack_samples_file"])
        filename.parent.mkdir(parents=True, exist_ok=True)
        attack_results = DataFrame(attack_samples.reshape(attack_samples.shape[0], -1))
        attack_results.to_json(filename)
        assert Path(filename).exists(), "Adversarial example file not saved"
        return str(Path(filename))

    def save_attack_predictions(
        self,
        attack_predictions: np.ndarray,
    ) -> Path:
        """
        Saves adversarial predictions to specified file.
        """
        logger.info("Saving attack predictions")
        filename = Path(self.files["path"], self.files["attack_predictions_file"])
        filename.parent.mkdir(parents=True, exist_ok=True)
        attack_results = DataFrame(attack_predictions)
        attack_results.to_json(filename)
        assert Path(filename).exists(), "Adversarial example file not saved"
        return str(Path(filename))
    
    def save_attack_probabilities(
        self,
        attack_probabilities: np.ndarray,
    ) -> Path:
        """
        Saves adversarial probabilities to specified file.
        """
        logger.info("Saving attack probabilities")
        filename = Path(self.files["path"], self.files["attack_probabilities_file"])
        filename.parent.mkdir(parents=True, exist_ok=True)
        attack_results = DataFrame(attack_probabilities)
        attack_results.to_json(filename)
        assert Path(filename).exists(), "Adversarial example file not saved"
        return str(Path(filename))

    def save_attack_scores(
        self,
        attack_score_dict: dict,
    ) -> Path:
        """Saves adversarial results to specified file.

        :param score_dict (dict): Dictionary of scores.

        :return path: Path to saved file.
        """
        logger.info("Saving attack scores")
        filename = Path(self.files["path"], self.files["attack_score_dict_file"])
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as f:
            try:
                json.dump(attack_score_dict, f)
            except TypeError as e:
                if isinstance(attack_score_dict, Series):
                    attack_score_dict.to_json(f)
                else:
                    raise e
        assert Path(filename).exists(), "Adversarial example file not saved"
        return str(Path(filename))

    def save(
        self,
        data: Namespace = None,
        model: Callable = None,
        attack_score_dict: dict = None,
        attack_predictions: dict = None,
        attack_probabilities: dict = None,
        attack_samples: dict = None,
        attack_time_dict: dict = None,
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
        logger.info("Saving attack results")
        outs = {}
        if data is not None:
            logger.warning("Saving ttack data is not implemented yet.")
        if model is not None:
            self.save_model(model)
        if attack_score_dict is not None:
            file = self.save_attack_scores(attack_score_dict)
            outs.update({"attack_score_dict": file})
        if attack_predictions is not None:
            file = self.save_attack_predictions(attack_predictions)
            outs.update({"attack_predictions": file})
        if attack_probabilities is not None:
            file = self.save_attack_probabilities(attack_probabilities)
            outs.update({"attack_probabilities": file})
        if attack_samples is not None:
            file = self.save_attack_samples(attack_samples)
            outs.update({"attack_samples": file})
        if attack_time_dict is not None:
            file = self.save_attack_time(attack_time_dict)
            outs.update({"attack_time_dict": file})
        return outs
