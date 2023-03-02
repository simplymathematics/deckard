import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import hydra
import numpy as np
import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from deckard.base.hashable import my_hash
from deckard.layers.base.data import DataConfig
from deckard.layers.base.model import ModelConfig
from deckard.layers.base.attack import AttackConfig
from deckard.layers.base.plots import PlotsConfig
from deckard.layers.base.experiment import ExperimentConfig
from deckard.layers.parse import parse

    
    




cs = ConfigStore.instance()
cs.store(name="config", node=ExperimentConfig)


# def parse(params:dict):
#     params = dict(params)
#     files = dict(params["files"])
#     data = dict(params["data"])
#     model = dict(params["model"])
#     files['data_path'] = params['data']['path'] if params['data']['path'] is not None else my_hash(data)
#     files['data_filetype'] = params['data']['filetype'] if params['data']['filetype'] is not None else "npz"
#     files['model_path'] = params['model']['path'] if params['model']['path'] is not None else my_hash(model)
#     files['model_filetype'] = params['model']['filetype'] if params['model']['filetype'] is not None else "pkl"
#     params['files'] = files
#     if files['path'] is None:
#         params['files']["path"] = str(my_hash(params))
#     else:
#         params['files']["path"] = str(Path(files['path']).as_posix())
#     if params['data']['filename'] is None:
#         params["data"]["filename"] = str(Path(files['data_path'], my_hash(data) + f".{files['data_filetype']}").as_posix())
#     if params['model']['filename'] is None:
#         params["model"]["filename"] = str(Path(files['model_path'], my_hash(model) + f".{files['model_filetype']}").as_posix())
#     return params



@hydra.main(
    version_base=None,
    config_path=str(Path(os.getcwd(), "conf")),
    config_name="config",
)
def my_app(cfg) -> None:
    import tempfile
    import uuid
    yaml_config = dict(cfg)
    parsed_config = parse(yaml_config)
    print(yaml.dump(parsed_config, default_flow_style=False))
    obj1 = OmegaConf.create(parsed_config)
    print(f"obj1.data is a callable True/False: {hasattr(obj1.data, '__call__')}")
    obj2 = OmegaConf.instantia
    
if __name__ == "__main__":
    
    parsed_config = my_app()
    