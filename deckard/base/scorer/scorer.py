from dataclasses import dataclass, field
from typing import Literal, Dict, List
from hydra.utils import call
from hydra.errors import InstantiationException
from omegaconf import DictConfig, OmegaConf, ListConfig
import numpy as np
import json
from pathlib import Path
import logging
from copy import deepcopy
import pickle
from art.utils import to_categorical

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
            else:  # pragma: no cover
                raise TypeError(f"Unknown type {type(arg)} for arg {arg}")
        args = new_args
        config = {"_target_": self.name}
        config.update(kwargs)
        try:
            result = call(config, *args, **kwargs)

        except InstantiationException as e:  # pragma: no cover
            if "continuous-multioutput" in str(e) or "multiclass-multioutput" in str(e):
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
            elif "nan" in str(e).lower():
                args = [np.nan_to_num(arg) for arg in args]
                result = call(config, *args, **kwargs)
            else:
                raise e
        return result

    def __call__(self, *args) -> float:
        score = self.score(*args)
        return score


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
            else:  # pragma: no cover
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
            elif filetype in [".pkl", ".pickle"]:
                with open(filename, "rb") as f:
                    scores = pickle.load(f)
            else:  # pragma: no cover
                raise NotImplementedError("Filetype not supported: {}".format(filetype))
        else:
            scores = {}
        return scores

    def __call__(
        self,
        *args,
        score_dict_file=None,
        labels_file=None,
        predictions_file=None,
    ):
        new_scores = {}
        args = list(args)
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
        elif filetype in [".pkl", ".pickle"]:
            with open(full_path, "wb") as f:
                pickle.dump(results, f)
        else:  # pragma: no cover
            raise ValueError(f"filetype {filetype} not supported for saving score_dict")
        return full_path
