import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ..attack import Attack
from ..data import Data
from ..files import FileConfig
from ..model import Model
from ..scorer import ScorerDict
from ..utils.hashing import my_hash

__all__ = ["Experiment"]
logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    data: Data = field(default_factory=Data)
    model: Union[Model, None] = field(default_factory=Model)
    attack: Union[Attack, None] = field(default_factory=Attack)
    scorers: Union[ScorerDict, None] = field(default_factory=ScorerDict)
    files: Union[FileConfig, None] = field(default_factory=FileConfig)
    name: Union[str, None] = field(default_factory=str)
    stage: Union[str, None] = field(default_factory=str)
    optimizers: Union[list, None] = field(default_factory=list)
    device_id: str = "cpu"
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(
        self,
        data: Data,
        model: Model,
        scorers: ScorerDict,
        files: list,
        device_id: str = "cpu",
        attack: Attack = None,
        name=None,
        stage=None,
        optimizers=None,
        **kwargs,
    ):
        # if isinstance(data, dict):
        #     self.data = Data(**data)
        self.data = Data(**OmegaConf.to_container(OmegaConf.create(data)))
        self.model = Model(**OmegaConf.to_container(OmegaConf.create(model))) if model is not None else None
        self.attack = Attack(**OmegaConf.to_container(OmegaConf.create(attack))) if attack is not None else None
        if isinstance(files, dict):
            self.files = FileConfig(**files)
        elif isinstance(files, DictConfig):
            file_dict = OmegaConf.to_container(files, resolve=True)
            self.files = FileConfig(**file_dict, files=files)
        elif isinstance(files, FileConfig):
            self.files = files
        else:  # pragma: no cover
            raise ValueError("files must be a dict, DictConfig, or FileConfig object.")
        assert isinstance(self.files, FileConfig)
        self.scorers = ScorerDict(**OmegaConf.to_container(OmegaConf.create(scorers)))
        self.device_id = device_id
        self.stage = stage
        self.optimizers = optimizers
        self.kwargs = kwargs
        self.name = name if name is not None else self._set_name()

    def __hash__(self):
        name = str(self.name).encode("utf-8")
        return int.from_bytes(name, "little")

    def __call__(self, **kwargs):
        """Runs the experiment. If the experiment has already been run, it will load the results from disk. If scorer is not None, it will return the score for the specified scorer. If scorer is None, it will return the score for the first scorer in the ScorerDict.
        :param scorer: The scorer to return the score for. If None, the score for the first scorer in the ScorerDict will be returned.
        :type scorer: str
        :return: The score for the specified scorer or the status of the experiment if scorer=None (default).
        """
        logger.debug("Running experiment with id: {}".format(self.get_name()))
        # Setup files, data, and model
        files = deepcopy(self.files).get_filenames()

        # Check status of files
        assert (
            "score_dict_file" in files
        ), f"score_dict_file must be in files. Got {files.keys()}"
        if (
            files["score_dict_file"] is not None
            and Path(files["score_dict_file"]).exists()
        ):
            score_dict = self.data.load(files["score_dict_file"])
            score_dict = dict(score_dict)
            assert isinstance(score_dict, dict), f"score_dict is {type(score_dict)}"
        else:
            score_dict = {}
        results = {}
        results["score_dict"] = score_dict
        files.update(**results)
        # TODO verify this works
        # data = self.data(**files)
        # files['data'] = data
        #########################################################################
        # Load or train model
        #########################################################################
        if self.model is not None:
            model_results = self.model(**files)
            score_dict.update(**model_results.pop("time_dict", {}))
            score_dict.update(**model_results.pop("score_dict", {}))
            files.update(**model_results)
            data=files['data']
            # Prefer probabilities, then loss_files, then predictions
            if (
                "probabilities" in model_results
                and model_results["probabilities"] is not None
            ):
                probs = model_results["probabilities"]
                logger.debug(f"probs shape: {probs.shape}")
            if (
                "predictions" in model_results
                and model_results["predictions"] is not None
            ):
                preds = model_results["predictions"]
                if not hasattr(preds, "shape"):
                    preds = np.array(preds)
                logger.debug(f"preds shape: {preds.shape}")
            if "losses" in model_results and model_results["losses"] is not None:
                losses = model_results["losses"]
                if not hasattr(losses, "shape"):
                    losses = np.array(losses)
                logger.debug(f"losses shape: {losses.shape}")
        else:  
            #########################################################################
            # Load or generate data
            # For experiments without models, e.g Mutual Information experiments on datasets
            #########################################################################
            data = self.data(**files)
            files['data'] = data
            preds = data[2]
        
        ##########################################################################
        # Load or run attack
        ##########################################################################
        if self.attack is not None:
            adv_results = self.attack(
                **files,
            )
            if "adv_predictions" in adv_results:
                adv_preds = adv_results["adv_predictions"]
                if not hasattr(adv_preds, "shape"):
                    adv_preds = np.array(adv_preds)
                logger.debug(f"adv_preds shape: {adv_preds.shape}")
            if "adv_losses" in adv_results:
                adv_preds = adv_results["adv_losses"]
                logger.debug(f"adv_losses shape: {adv_preds.shape}")
            if "adv_probabilities" in adv_results:
                adv_preds = adv_results["adv_probabilities"]
                logger.debug(f"adv_probabilities shape: {adv_preds.shape}")
            if "time_dict" in adv_results:
                adv_time_dict = adv_results["time_dict"]
                score_dict.update(**adv_time_dict)
            if "adv_success" in adv_results:
                adv_success = adv_results["adv_success"]
                score_dict.update({"adv_success": adv_success})
            files.update(**adv_results)
        ##########################################################################
        # Score results
        #########################################################################
        if self.scorers is not None:
            if "probs" in locals() and "preds" not in locals():
                preds = probs
            elif "losses" in locals() and "preds" not in locals():
                preds = losses
            if "preds" in locals() and self.model is not None:
                ground_truth = data[3][: len(preds)]
                logger.debug(f"preds shape: {preds.shape}")
                logger.debug(f" len(preds) : {len(preds)}")
                if self.model is not None:
                    preds = preds[: len(ground_truth)]
                    ground_truth = ground_truth[: len(preds)]
                else:  # For dexperiments without models
                    preds = data[0][: len(ground_truth)]
                    ground_truth = data[1][: len(preds)]
                logger.debug(f" len(preds) : {len(preds)}")
                new_score_dict = self.scorers(ground_truth, preds)
                score_dict.update(**new_score_dict)
                results["score_dict"] = score_dict
            if "adv_preds" in locals():
                ground_truth = data[3][: len(adv_preds)]
                adv_preds = adv_preds[: len(ground_truth)]
                adv_score_dict = self.scorers(ground_truth, adv_preds)
                adv_score_dict = {
                    f"adv_{key}": value for key, value in adv_score_dict.items()
                }
                score_dict.update(**adv_score_dict)
                results["score_dict"] = score_dict
            # # Save results
            if "score_dict_file" in files and files["score_dict_file"] is not None:
                if Path(files["score_dict_file"]).exists():
                    old_score_dict = self.data.load(files["score_dict_file"])
                    old_score_dict.update(**score_dict)
                    score_dict = old_score_dict
                score_dict.update({"device_id": self.device_id})
                self.data.save(score_dict, files["score_dict_file"])
        else:  # pragma: no cover
            raise ValueError("Scorer is None. Please specify a scorer.")
        logger.debug(f"Score for id : {self.get_name()}: {score_dict}")
        logger.debug("Finished running experiment with id: {}".format(self.get_name()))
        return score_dict

    def _set_name(self):
        if self.files.name is not None:
            name = self.files.name
        else:
            name = my_hash(vars(self))
        return name

    def get_name(self):
        return self.name
