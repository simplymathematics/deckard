import argparse
import logging
import os
from pathlib import Path

import hydra
import yaml
import dvc.api
from mergedeep import merge
from omegaconf import DictConfig, OmegaConf

from deckard.base import Experiment
from deckard.layers.parse import parse

logger = logging.getLogger(__name__)

@hydra.main(
    version_base=None,
    config_path=str(Path(os.getcwd(), "conf")),
    config_name="config",
)
def hydra_runner(cfg: DictConfig, **kwargs):
    if "default" in cfg:
        default = cfg.default
        del cfg.default
    else:
        default = "params.yaml"
    if "queue_path" in cfg:
        queue_path = cfg.queue_path
        del cfg.queue_path
    else:
        queue_path = "queue"
    if "filename" in cfg:
        filename = cfg.filename
        del cfg.filename
    else:
        filename = "default.yaml"
    params = OmegaConf.to_object(cfg)
    kwargs.update({"default": default, "queue_path": queue_path, "filename": filename})
    params = parse(params, **kwargs)
    exp = load_dvc_experiment(params)
    results = exp.run()
    return results

def merge_params(default, params):
    for key, value in params.items():
        if key in default:
            if isinstance(default[key], dict):
                default[key] = merge_params(default[key], value)
            else:
                default[key] = value
        else:
            logger.warning(f"Key {key} not found in default params. Ignoring.")
    return default
            
def load_dvc_experiment(stage = None,  params = None):
    # Load params from dvc
    if stage is None:
        with open(Path(os.getcwd(), "dvc.yaml"), "r") as f:
            stages = yaml.load(f, Loader=yaml.FullLoader)["stages"].keys() #  Get all stages
        stage = list(stages)[-1] # Get the last stage
    if params is None:
        params = dvc.api.params_show("params.yaml", stages = [stage])
    else:
        default_params = dvc.api.params_show("params.yaml", stages = [stage])
        params = merge_params(default_params, params)
    # Update params with paths from dvc
    Path(params["files"]["data_file"]).parent.mkdir(parents=True, exist_ok=True)
    Path(params["files"]["model_file"]).parent.mkdir(parents=True, exist_ok=True)
    full_report = params["files"]["path"]
    parents = list(Path(full_report).parents)
    name = Path(full_report).name
    parents.insert(1, Path(stage))
    params["files"]["path"] = str(Path(params["files"]["reports"], *parents, name))
    # Load and run experiment from yaml
    tag = "!Experiment:"
    yaml.add_constructor(tag, Experiment)
    exp = yaml.load(f"{tag}\n" + str(params), Loader=yaml.FullLoader)
    return exp

    
if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbosity", type=str, default="INFO")
    parser.add_argument("stage", type=str, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    exp =  load_dvc_experiment(args.stage)
    results = exp.run()
    print(yaml.dump(results))
    