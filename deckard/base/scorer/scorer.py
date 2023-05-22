from dataclasses import dataclass, field
from typing import Literal, Dict, List
from hydra.utils import call
from hydra.errors import InstantiationException
from omegaconf import DictConfig, OmegaConf, ListConfig
import numpy as np
import json
from pathlib import Path
import pandas as pd
import logging
from copy import deepcopy
import pickle
from art.utils import to_categorical

from ..data import Data
from ..model import Model
from ..attack import Attack
from ..utils import my_hash

logger = logging.getLogger(__name__)

__all__ = ["ScorerConfig", "ScorerDict"]


@dataclass
class ScorerConfig:
    # _target_: str = "scorer.ScorerConfig"
    name: str = "sklearn.metrics.accuracy_score"
    alias: str = "accuracy"
    args: List[str] = field(default_factory=list)
    params: Dict[str, str] = field(default_factory=dict)
    direction: str = "maximize"

    def __init__(
        self,
        name: str,
        direction: Literal["maximize", "minimize"],
        alias=None,
        args=["y_true", "y_pred"],
        params: dict = {},
        **kwargs,
    ):
        # self._target_ = "scorer.ScorerConfig"
        self.name = kwargs.pop("_target_", name)
        self.alias = alias if alias is not None else name.split(".")[-1].lower()
        args = args if args is not None else ["y_pred", "y_true"]
        params.update(**kwargs.pop("params", {}))
        self.direction = direction
        if "args" in kwargs:
            new_args = kwargs.pop("args")
            if isinstance(new_args, dict):
                for v in new_args.values():
                    if v not in args:
                        args.append(v)
            elif isinstance(new_args, DictConfig):
                new_args = OmegaConf.to_container(new_args, resolve=True)
                for v in new_args.values():
                    if v not in args:
                        args.append(v)
            elif isinstance(new_args, list):
                for v in new_args:
                    if v not in args:
                        args.append(v)
            elif isinstance(new_args, ListConfig):
                new_args = OmegaConf.to_container(new_args, resolve=True)
                for v in new_args:
                    if v not in args:
                        args.append(v)
        self.args = args
        self.params = kwargs
        self.direction = direction
        logger.debug(
            f"Initializing scorer {self.name} with args {self.args} and params {self.params}",
        )

    def __hash__(self):
        return int(my_hash(self), 16)

    def score(self, ind, dep) -> float:
        args = deepcopy(self.args)
        kwargs = deepcopy(self.params)
        new_args = []
        i = 0
        for arg in args:
            i += 1
            if arg in ["y_pred", "y_train", "y_test"]:
                new_args.append(dep)
            elif arg in ["labels", "y_true", "ground_truth"]:
                new_args.append(ind)
            elif isinstance(arg, str) and Path(arg).exists():
                arg = self.load(arg)
                new_args.append(arg)
            elif isinstance(arg, list):
                new_args.append(np.array(arg))
            elif isinstance(arg, ListConfig):
                arg = OmegaConf.to_container(arg, resolve=True)
                new_args.extend(arg)
            elif isinstance(arg, dict):
                arg = OmegaConf.to_container(arg, resolve=True)
                new_args.append(arg)
            elif isinstance(arg, DictConfig):
                arg = OmegaConf.to_container(arg, resolve=True)
                new_args.append(arg)
            else:
                raise TypeError(f"Unknown type {type(arg)} for arg {arg}")
        args = new_args
        config = {"_target_": self.name}
        config.update(kwargs)
        while "params" in config:
            config.update(**config.pop("params"))
        while "kwargs" in config:
            config.update(**config.pop("kwargs"))
        new_args = []
        for arg in args:
            if isinstance(arg, str) and Path(arg).exists():
                arg = self.load(arg)
            new_args.append(arg)
        args = new_args
        try:
            result = call(config, *args, **kwargs)

        except InstantiationException as e:
            if "continuous-multioutput" in str(e):
                new_args = []
                for arg in args:
                    if hasattr(arg, "shape"):
                        if len(np.squeeze(arg).shape) == 2:
                            arg = np.argmax(np.squeeze(arg), axis=1)
                        else:
                            pass
                    elif isinstance(arg, (ListConfig, DictConfig)):
                        arg = OmegaConf.to_container(arg, resolve=True)
                    else:
                        pass
                    new_args.append(arg)
                args = new_args
                result = call(config, *args, **kwargs)
            elif "binary" in str(e):
                new_args = []
                for arg in args:
                    if hasattr(arg, "shape"):
                        if len(np.squeeze(arg).shape) == 1:
                            arg = to_categorical(arg)
                        else:
                            pass
                    elif isinstance(arg, (ListConfig, DictConfig)):
                        arg = OmegaConf.to_container(arg, resolve=True)
                    else:
                        pass
                    new_args.append(arg)
                args = new_args
                result = call(config, *args, **kwargs)
            else:
                raise e
        return result

    def __call__(self, *args, **kwargs) -> float:
        new_args = []
        for arg in args:
            if isinstance(arg, str) and Path(arg).exists():
                arg = self.load(arg)
            elif isinstance(arg, np.ndarray):
                pass
            elif isinstance(arg, pd.DataFrame):
                arg = arg.values
            elif isinstance(arg, pd.Series):
                arg = arg.values
            elif isinstance(arg, list):
                arg = np.array(arg)
            elif isinstance(DictConfig):
                arg = OmegaConf.to_container(arg, resolve=True)
                kwargs.update(**arg)
            elif isinstance(arg, dict):
                kwargs.update(**arg)
            elif isinstance(arg, ListConfig):
                arg = OmegaConf.to_container(arg, resolve=True)
            elif isinstance(arg, Data):
                arg = arg()
            elif isinstance(arg, Model):
                _, arg = arg.initialize()
            elif isinstance(arg, Attack):
                raise NotImplementedError("Attacks are not supported yet")
            new_args.append(arg)
        args = new_args
        score = self.score(*args, **kwargs)
        return score

    def read(self, filename):
        suffix = Path(filename).suffix
        if suffix in [".json"]:
            with open(filename, "r") as f:
                data = json.load(f)
        elif suffix in [".csv"]:
            data = pd.read_csv(filename)
        elif suffix in [".xlsx"]:
            data = pd.read_excel(filename)
        elif suffix in [".pkl", ".pickle"]:
            with open(filename, "rb") as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unknown file type {suffix}")
        return data


@dataclass
class ScorerDict:
    scorers: Dict[str, ScorerConfig] = field(default_factory=dict)

    def __init__(self, scorers: Dict[str, ScorerConfig] = {}, **kwargs):
        self.scorers = {}
        scorers.update(kwargs)
        for k, v in scorers.items():
            if isinstance(v, DictConfig):
                v = OmegaConf.to_container(v, resolve=True)
                v = ScorerConfig(**v)
            elif isinstance(v, dict):
                v = ScorerConfig(**v)
            elif isinstance(v, ScorerConfig):
                pass
            else:
                raise TypeError(f"Unknown type {type(v)} for scorer {k}")
            self.scorers[k] = v

    def __iter__(self):
        for k, v in self.scorers.items():
            yield k, v

    def __hash__(self):
        return int(my_hash(self), 16)

    def load(self, filename):
        filetype = Path(filename).suffix
        if Path(filename).exists():
            if filetype in [".json"]:
                with open(filename, "r") as f:
                    scores = json.load(f)
            elif filetype in [".csv"]:
                scores = pd.read_csv(filename).to_dict()
            else:
                raise NotImplementedError("Filetype not supported: {}".format(filetype))
        else:
            scores = {}
        return scores

    def __call__(
        self, *args, score_dict_file=None, labels_file=None, predictions_file=None
    ):
        new_scores = {}
        i = 0
        for arg in args:
            if hasattr(arg, "shape"):
                logger.debug(f"arg {i} has shape {arg.shape}")
            else:
                logger.debug(f"arg {i} has type {type(arg)}")
        if score_dict_file is not None and Path(score_dict_file).exists():
            scores = self.load(score_dict_file)
        else:
            scores = {}
        if labels_file is not None and Path(labels_file).exists():
            ind = self.load(labels_file)
            args[0] = ind
        else:
            pass
        if predictions_file is not None and Path(predictions_file).exists():
            dep = self.load(predictions_file)
            args[1] = dep
        else:
            pass
        for name, scorer in self:
            score = scorer.score(*args)
            new_scores[name] = score
        if score_dict_file is not None:
            scores = self.load(score_dict_file)
            scores.update(**new_scores)
            self.save(scores, score_dict_file)
            new_scores = scores
        return new_scores

    def __getitem__(self, key):
        return self.scorers[key]

    def __len__(self):
        return len(self.scorers)

    def save(self, results, filename):
        full_path = Path(filename)
        filetype = full_path.suffix
        # Prepare directory
        full_path.parent.mkdir(parents=True, exist_ok=True)
        if filetype in [".json"]:
            with open(full_path, "w") as f:
                json.dump(results, f)
        elif filetype in [".csv"]:
            df = pd.DataFrame(results)
            df.to_csv(full_path, index=False)
        else:
            raise ValueError(f"filetype {filetype} not supported for saving score_dict")
        return full_path
