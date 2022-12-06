

import argparse
import logging
import os
from pathlib import Path

import dvc.api
import hydra
import yaml
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from deckard.base import Experiment
from deckard.base.hashable import my_hash

config_path =  Path(os.getcwd())

logger = logging.getLogger(__name__)
queue_path = "queue"

@hydra.main(version_base=None, config_path=Path(os.getcwd(), "conf"), config_name="config")
def parse(cfg: DictConfig, queue_path = "queue"):
    params = OmegaConf.to_object(cfg)
    files = params["files"]
    data = params["data"]
    model = params["model"]
    if "data" in params and "files" in params["data"]:
        files.update(params["data"].pop("files"))
    if "model" in params and "files" in params["model"]:
        files.update(params["model"].pop("files"))
    if "data_file" not in params and "data" in params:
        params["files"]['data_file'] = str(Path(files['data_path'], my_hash(data) + "." + files['data_filetype']).as_posix())
    if "model_file" not in params and "model" in params:
        params["files"]['model_file'] = str(Path(files['model_path'], my_hash(model) + "." + files['model_filetype']).as_posix())
    with open(Path(os.getcwd(), "params.yaml"), "w") as f:
        yaml.dump(params, f)
    logger.info(f"Wrote params to {Path(os.getcwd(), 'params.yaml')}")
    assert Path(os.getcwd(), 'params.yaml').exists(), f"File {path} does not exist. Something went wrong."
    params = dvc.api.params_show(Path(os.getcwd(), 'params.yaml'))
    if "files" in params:
        params["files"]["path"] = str( my_hash(params))
    if "data_file" in params.get("files", {}):
        params["files"]["data_file"] = str(Path(params["files"]["data_file"]))
    if "model_file" in params.get("files", {}):
        params["files"]["model_file"] = str(Path(params["files"]["model_file"]))
    if "attack" in params:
        if "files" in params['attack'] and "attack_samples_file" in params['attack']["files"]:
            attack_files = params['attack'].pop("files")
            for atk_file in attack_files:
                params["files"][atk_file] = str(Path(attack_files[atk_file]))
    Path(os.getcwd(), 'params.yaml').unlink()
    filename = Path(queue_path, my_hash(params) + ".yaml")
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(Path(filename), "w") as f:
        yaml.dump(params, f)
    return None



if __name__ == "__main__":
    _ = parse()
    assert Path(queue_path).exists(), f"Queue path {queue_path} does not exist. Something went wrong."
        
    
    
    