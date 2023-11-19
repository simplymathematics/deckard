import logging
import os
import traceback
from copy import deepcopy
from pathlib import Path
import time
import random
import yaml
from hydra.utils import instantiate
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, ListConfig
import hydra
from dvc.api import params_show
from deckard.base.experiment import Experiment
from dvc.lock import LockError
from ..base.utils import my_hash, unflatten_dict
from .utils import deckard_nones

logger = logging.getLogger(__name__)

__all__ = ["write_stage", "optimise", "parse_stage", "get_files"]


def get_files(
    cfg,
    stage,
):
    """
    Gets the file names from
    """
    if isinstance(cfg, dict):
        pass
    elif isinstance(cfg, list):
        cfg = unflatten_dict(cfg)
    else:
        raise TypeError(f"Expected dict or list, got {type(cfg)}")
    if "_target_" not in cfg:
        cfg.update({"_target_": "deckard.base.experiment.Experiment"})
    if (
        "attack_file" in cfg["files"]
        and cfg["files"]["attack_file"] is not None
        and "attack" in cfg
    ):
        cfg["files"]["attack_file"] = str(
            Path(cfg["files"]["attack_file"])
            .with_name(my_hash(cfg["attack"]))
            .as_posix(),
        )
    if (
        "model_file" in cfg["files"]
        and cfg["files"]["model_file"] is not None
        and "model" in cfg
    ):
        cfg["files"]["model_file"] = str(
            Path(cfg["files"]["model_file"])
            .with_name(my_hash(cfg["model"]))
            .as_posix(),
        )
    if (
        "data_file" in cfg["files"]
        and cfg["files"]["data_file"] is not None
        and "data" in cfg
    ):
        cfg["files"]["data_file"] = str(
            Path(cfg["files"]["data_file"]).with_name(my_hash(cfg["data"])).as_posix(),
        )
    id_ = my_hash(cfg)
    
    id_ = my_hash(cfg)
    cfg["name"] = id_
    cfg["files"]["name"] = id_
    cfg["files"]["stage"] = stage
    cfg['stage'] = stage
    return cfg

def parse_stage(*args, repo, stages):
    if not isinstance(stages, list):
        stages = [stages]
    base_delay = 10
    max_retries = 100
    args = list(args)
    logger.info(f"Parsing {args[:]} from {repo} with stages {stages}")
    retries = 0
    params = params_show(*args, repo=repo, stages=stages, deps=True)
    # while retries < max_retries:
    #     try:
    #         params = params_show(*args, repo=repo, stages=stages, deps=True)
    #     except LockError:
    #         retries += 1 
    #         delay = base_delay ** 2 **retries 
    #         logger.warning(f"LockError occured. Retrying in {delay:.2f} seconds. Retry {retries} of {max_retries}.") 
    #         time.sleep(delay)
    #     raise LockError(f"LockError occured {max_retries} times. Aborting.")
    return params

def parse_stage(cfg, repo, stages):
    if not isinstance(stages, list):
        stages = [stages]
    params_subset = []
    all_params = []
    with open(Path(repo, "dvc.yaml"), "r") as f:
        for stage in stages:
            params_dict = yaml.load(f, Loader=yaml.FullLoader)["stages"][stage]['params']
            if isinstance(params_dict, dict):
                params_subset.extend(list(params_dict.keys()))
            elif isinstance(params_dict, list):
                params_subset.extend(params_dict)
            

                
    with open(Path(repo, "params.yaml"), "r") as f:
        
        
        
    


def write_stage(params: dict, stage: str, path:str, working_dir:str) -> None:
    """
    Write params to dvc.yaml
    :param params: Params to write to dvc.yaml
    :param stage: Stage to write params to
    """
    with open(Path(working_dir, "dvc.yaml"), "r") as f:
        dvc = yaml.load(f, Loader=yaml.FullLoader)
    name = Path(path).name
    stage_params = {"stages": {}}
    stage_params["stages"][f"{stage}"] = dvc["stages"][stage]
    path.mkdir(exist_ok=True, parents=True)
    with open(path / "dvc.yaml", "w") as f:
        yaml.dump(stage_params, f, default_flow_style=False)
    assert Path(path / "dvc.yaml").exists(), f"File {path/'dvc.yaml'} does not exist."
    
    with open(Path(path, "params.yaml"), "w") as f:
        yaml.dump(params, f, default_flow_style=False)
    assert Path(
        path / "params.yaml",
    ).exists(), f"File {path/'params.yaml'} does not exist."
    return None


def optimise(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(OmegaConf.create(cfg), resolve=False)
    stage = cfg.get("stage", None)
    if isinstance(stage, ListConfig):
        stage = OmegaConf.to_container(stage, resolve=True)
    if isinstance(stage, list):
        stage = stage[0]
    assert stage is not None, f"Stage must be specified. Add stage=<stage> to command line."
    assert isinstance(stage, str), f"Expected str, got {type(stage)}"
    scorer = cfg.pop("optimizers", None)
    working_dir = cfg.pop("working_dir", Path().resolve().as_posix())
    cfg = get_files(cfg, stage=stage)
    direction = cfg.pop("direction", "minimize")
    exp = instantiate(cfg)
    files = deepcopy(exp.files)()
    id_ = Path(files["score_dict_file"]).parent.name
    old = id_
    folder = Path(files["score_dict_file"]).parent
    assert write_stage(cfg, stage, path=folder, working_dir=working_dir) is None, f"Failed to write stage {stage} to {folder}"
    targets = []
    cfg = parse_stage(*targets, stages=stage, repo=folder)
    cfg['files']['name'] = old
    cfg['_target_'] = 'deckard.base.Experiment'
    cfg["files"]["_target_"] = "deckard.base.files.FileConfig"
    cfg['files']['stage'] = stage
    cfg['stage'] = stage
    exp = instantiate(cfg)
    assert isinstance(exp, Experiment), f"Expected Experiment, got {type(exp)}"
    assert exp.name == old, f"Expected {old}, got {exp.name}"
    
    try:
        scores = exp()
        if isinstance(scorer, str):
            score = scores[scorer]
        elif isinstance(scorer, list):
            score = [scores[s] for s in scorer]
        elif scorer is None:
            score = list(scores.values())[0]
        else:
            raise TypeError(f"Expected str or list, got {type(scorer)}")
    except Exception as e:
        logger.warning(
            f"Exception {e} occured while running experiment {id_}. Setting score to default for specified direction (e.g. -/+ 1e10).",
        )
        with open(Path(folder, "exception.log"), "w") as f:
            f.write(str(e))
            f.write(traceback.format_exc())
        if direction == "minimize":
            score = 1e10
        else:
            score = -1e10
    return score


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    config_path = os.environ.get(
        "DECKARD_CONFIG_PATH",
        str(Path(Path(), "conf").absolute().as_posix()),
    )
    config_name = os.environ.get("DECKARD_DEFAULT_CONFIG", "default.yaml")
    
    @hydra.main(config_path=config_path, config_name=config_name, version_base="1.3")
    def hydra_optimise(cfg: DictConfig) -> float:
        score = optimise(cfg)
        return score
    hydra_optimise()
