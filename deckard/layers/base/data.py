import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import hydra
import numpy as np
import yaml
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from deckard.base.hashable import my_hash
from deckard.layers.parse import parse


@dataclass
class SamplerConfig:
    random_state: int = 42
    shuffle: bool = True
    stratify: bool = True
    test_size : int = 1000
    time_series: bool = False
    train_size : int = 1000
    
    def __call__(self, X, y, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = asdict(self)
        time_series = params.pop("time_series", False)
        stratify = params.pop("stratify", False)
        if stratify is True:
            stratify = y
        else:
            stratify = None
        if time_series is False:
            for k in kwargs:
                if k in params:
                    raise ValueError(f"Parameter {k} already set.")
                else:
                    params[k] = kwargs[k]
            results = train_test_split(X, y, **params)
        else:
            raise NotImplementedError("Time series not implemented yet.")
        return results

@dataclass
class ClassificationGeneratorConfig:
    n_samples: int = 10000
    n_features: int = 100
    n_informative: int = 99
    n_redundant: int = 0
    n_repeated: int = 0
    n_classes: int = 2
    n_clusters_per_class: int = 1
    weights: Union[list, None] = None
    class_sep: float = 1.0
    hypercube: bool = True
    shift: float = 0.0
    scale: float = 1.0
    shuffle: bool = True
    random_state: int = 42
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        params = asdict(self)
        for k in kwds:
            if k in params:
                raise ValueError(f"Parameter {k} already set.")
            else:
                params[k] = kwds[k]
        return make_classification(*args, **params)

@dataclass
class DataConfig:
    name : Union[str, None] = "classification"
    generate : Union[ClassificationGeneratorConfig, None] = field(default_factory=ClassificationGeneratorConfig)
    sample : Union[SamplerConfig, None] = field(default_factory=SamplerConfig)
    filename : Union[str, None] = None
    path : Union[str, None] = "data"
    filetype : Union[str, None] = "npz"
    
    def _initialize(self, filename=None, path=None, filetype =None) -> Tuple:
        if filename is None:
            filename = self.filename if self.filename is not None else my_hash(self)
        if path is None:
            path = self.path if self.path is not None else hydra.utils.get_original_cwd()
        if filetype is None:
            filetype = self.filetype if self.filetype is not None else "npz"
        full_path =Path(path, str(filename)+ "." + filetype)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not full_path.is_file():
            if Path(self.name).is_file():
                X_train, X_test, y_train, y_test = self._load(filename=self.name, path=self.path, filetype=self.filetype)
            elif self.generate is not None:
                generator = instantiate(self.generate)
                generator.pop("__target__", None) if isinstance(generator, dict) else None
                generator = ClassificationGeneratorConfig(**generator)
                X, y = generator()
            if self.sample is not None:
                sampler = instantiate(self.sample)
                sampler.pop("__target__", None) if isinstance(sampler, dict) else None
                sampler = SamplerConfig(**sampler)
                X_train, X_test, y_train, y_test = self.sample(X = X, y = y)
            else:
                raise NotImplementedError("No data sampling method specified.")
            self._save(X_train, X_test, y_train, y_test)
        else:
            X_train, X_test, y_train, y_test = self._load(filename=filename, path=path, filetype=filetype)
        return X_train, X_test, y_train, y_test
    
    def _load(self, filename=None, path=None, filetype=None) -> Tuple:
        if path is None:
            path = self.path if self.path is not None else hydra.utils.get_original_cwd()
            path = Path(path)
        if filename is None:
            filename = self.filename if self.filename is not None else my_hash(self)
        if filetype is None:
            filetype = self.filetype if self.filetype is not None else "npz"
        path = path / Path(str(filename) + "." + filetype)
        data = np.load(str(path.resolve()))
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        return X_train, X_test, y_train, y_test
    
    def _save(self, X_train, X_test, y_train, y_test, filename=None, path=None, filetype=None) -> None:
        if path is None:
            path = self.path if self.path is not None else hydra.utils.get_original_cwd()
        Path(path).mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = self.filename if self.filename is not None else my_hash(self)
        if filetype is None:
            filetype = self.filetype if self.filetype is not None else "npz"
        path = Path(path) / Path(str(filename) + "." + filetype)
        np.savez(file=str(path.resolve()), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        return path

    def __call__(self, filename=None, path=None, filetype=None) -> Tuple:
        X_train, X_test, y_train, y_test = self._initialize(filename=filename, path=path, filetype=filetype)
        file = self._save(X_train, X_test, y_train, y_test, filename=filename, path=path, filetype=filetype)
        assert Path(file).exists(), f"File {file} does not exist."
        return X_train, X_test, y_train, y_test
    
@hydra.main(
    version_base=None,
    config_path=str(Path(os.getcwd(), "conf-tmp")),
    config_name="config",
)
def my_app(cfg) -> None:
    import tempfile
    import uuid
    yaml_config = dict(cfg)
    parsed_config = parse(yaml_config)
    instance = instantiate(parsed_config)
    generator = instantiate(instance.data.generate)
    generator.pop("__target__")
    generator = ClassificationGeneratorConfig(**generator)
    sampler = instantiate(instance.data.sample)
    sampler.pop("__target__")
    sampler = SamplerConfig(**sampler)
    data = DataConfig(name=instance.data.name, generate=generator, sample=sampler, filename =uuid.uuid4(), path=tempfile.gettempdir(), filetype="npz")
    X_train, X_test, y_train, y_test = data()
    assert isinstance(X_train, np.ndarray), f"X_train is not a numpy array."
    assert isinstance(X_test, np.ndarray), f"X_test is not a numpy array."
    assert isinstance(y_train, np.ndarray), f"y_train is not a numpy array."
    assert isinstance(y_test, np.ndarray), f"y_test is not a numpy array."
if __name__ == "__main__":
    
    parsed_config = my_app()
    