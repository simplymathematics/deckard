import argparse
import logging
import os
from pathlib import Path
from typing import Literal
import yaml
from omegaconf import DictConfig, OmegaConf
import dvc.api

from deckard.base.experiment import Experiment
from deckard.base.hashable import my_hash
from deckard.layers.parse import parse

config_path = Path(os.getcwd())

logger = logging.getLogger(__name__)


def runner(cfg: DictConfig, stage: str = None, save_model=True):
    cfgs = parse(cfg, stage=stage)
    big_results = {}
    for cfg in cfgs:
        params = OmegaConf.to_container(cfg, resolve=True)
        exp = load_dvc_experiment(stage=stage, params=params, parse=False)
        results = exp.run(save_model=save_model)
        big_results[my_hash(params)] = results
    return big_results


def merge_params(default, params) -> dict:
    """
    Overwrite default params with params if key is found in default.
    :param default: Default params
    :param params: Params to overwrite default params
    :return: Merged params
    """
    for key, value in params.items():
        if key in default and isinstance(default[key], dict):
            default[key] = merge_params(default[key], value)
        elif isinstance(value, (list, tuple, int, float, str, bool)):
            default[key] = value
        else:
            logger.warning(f"Key {key} not found in default params. Skipping.")
    return default


def read_subset_of_params(key_list: list, params: dict):
    """
    Read a subset of the params, denoted by the key_list
    :param key_list: The list of keys to read
    :param params: The params to read from
    :return: The subset of the params
    """
    new_params = {}
    for key in key_list:
        if key in params:
            new_params[key] = params[key]
        elif "." in key:
            first_loop = True
            dot_key = key
            total = len(key.split("."))
            i = 1
            for entry in key.split("."):
                if first_loop is True:
                    sub_params = params
                    first_loop = False
                if entry in sub_params:
                    sub_params = sub_params[entry]
                    i += 1
                else:
                    raise ValueError(f"{dot_key} not found in {params}")
                if i == total:
                    new_params[dot_key.split(".")[0]] = {**sub_params}
                else:
                    pass

        else:
            raise ValueError(f"{key} not found in {params}")
    return new_params


def parse_stage(stage: str = None, params: dict = None, path=None) -> dict:
    """
    Parse params from dvc.yaml and merge with params from hydra config
    :param stage: Stage to load params from. If None, loads the last stage in dvc.yaml
    :param params: Params to merge with params from dvc.yaml
    :return: Merged params
    """
    if path is None:
        path = Path.cwd()
    # Load params from dvc
    if stage is None:
        with open(Path(path, "dvc.yaml"), "r") as f:
            stages = yaml.load(f, Loader=yaml.FullLoader)[
                "stages"
            ].keys()  # Get all stages
        stage = list(stages)[-1]  # Get the last stage
    if params is None:
        with open(Path(path, "params.yaml"), "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        with open(Path(path, "dvc.yaml"), "r") as f:
            key_list = yaml.load(f, Loader=yaml.FullLoader)["stages"][stage]["params"]
        default_params = read_subset_of_params(key_list, params)
        params = merge_params(default_params, params)
    elif isinstance(params, str) and Path(params).is_file() and Path(params).exists():
        with open(Path(params), "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        assert isinstance(
            params,
            dict,
        ), f"Params in file {params} must be a dict. It is a {type(params)}."
        with open(Path(path, "dvc.yaml"), "r") as f:
            key_list = yaml.load(f, Loader=yaml.FullLoader)["stages"][stage]["params"]
        with open(Path(path, "params.yaml"), "r") as f:
            all_params = yaml.load(f, Loader=yaml.FullLoader)
        default_params = read_subset_of_params(key_list, all_params)
        params = merge_params(default_params, params)
    elif isinstance(params, dict):
        with open(Path(path, "dvc.yaml"), "r") as f:
            key_list = yaml.load(f, Loader=yaml.FullLoader)["stages"][stage]["params"]
        with open(Path(path, "params.yaml"), "r") as f:
            all_params = yaml.load(f, Loader=yaml.FullLoader)
        default_params = read_subset_of_params(key_list, all_params)
        params = merge_params(default_params, params)
    assert isinstance(
        params,
        dict,
    ), f"Params must be a dict. It is type {type(params)}."
    # Update params with paths from dvc
    report = params["files"]["reports"]
    name = params["files"]["path"]
    if name in str(Path(report)):
        name = str(Path(report).name)
        report = str(Path(report).parent.parent)
    else:
        name = str(Path(name).name)
    new_path = str(Path(report, stage, name))
    params["files"]["path"] = new_path
    return params


def load_dvc_experiment(
    stage=None,
    params=None,
    mode: Literal["dvc", "hydra"] = "dvc",
) -> Experiment:
    """
    Load experiment from dvc.yaml for a given stage and overwrite the default params with optional user-supplied params.
    :param stage: Stage to load params from. If None, loads the last stage in dvc.yaml
    :param params: Params to merge with params from dvc.yaml. Params can be a dict or a path to a yaml file.
    :return: Experiment
    """
    # Load and run experiment from yaml
    if mode == "hydra":
        params = parse_stage(stage, params)
    elif mode == "dvc":
        params = dvc.api.params_show("params.yaml", stages=[stage])
    tag = "!Experiment:"
    yaml.add_constructor(tag, Experiment)
    exp = yaml.load(f"{tag}\n" + str(params), Loader=yaml.FullLoader)
    return exp


if "__main__" == __name__:

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, default=Path.cwd())
    parser.add_argument("--verbosity", type=str, default="INFO")
    parser.add_argument("--stage", type=str, default=None)
    args = parser.parse_args()
    if args.exp_path == Path.cwd():
        exp = load_dvc_experiment(args.stage, mode="hydra")
    else:
        exp = load_dvc_experiment(args.stage, path=args.exp_path)
    logging.basicConfig(level=args.verbosity)
    results = exp.run(save_data=True)
    logger.info(yaml.dump(results))
