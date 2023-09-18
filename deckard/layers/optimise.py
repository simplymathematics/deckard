import logging
import os
import traceback
from copy import deepcopy
from pathlib import Path
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import hydra
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
    cfg["files"]["_target_"] = "deckard.base.files.FileConfig"
    id_ = my_hash(cfg)
    cfg["name"] = id_
    cfg["files"]["name"] = id_
    if stage is not None:
        cfg["files"]["stage"] = stage
    return cfg


# def save_file(cfg, folder, params_file):
#     path = Path(folder, Path(params_file).name)
#     with open(path, "w") as f:
#         yaml.safe_dump(cfg, f)
#     assert Path(path).exists()


def merge_params(default, params) -> dict:
    """
    Overwrite default params with params if key is found in default.
    :param default: Default params
    :param params: Params to overwrite default params
    :return: Merged params
    """
    for key, value in params.items():
        if key in default and isinstance(value, dict) and value is not None:
            default[key] = merge_params(default[key], value)
        elif (
            isinstance(value, (list, tuple, int, float, str, bool, dict))
            and value is not None
        ):
            default[key] = value
        elif value is None:
            continue
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
            if params[key] in deckard_nones:
                continue
            elif hasattr(params[key], "__len__") and len(params[key]) == 0:
                continue
            else:
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
        raise ValueError("Please specify a stage.")
    elif isinstance(stage, str):
        stages = [stage]
    else:
        assert isinstance(stage, list), f"args.stage is of type {type(stage)}"
        stages = stage
    if params is None:
        with open(Path(path, "params.yaml"), "r") as f:
            default_params = yaml.load(f, Loader=yaml.FullLoader)
        key_list = []
        for stage in stages:
            with open(Path(path, "dvc.yaml"), "r") as f:
                new_keys = yaml.load(f, Loader=yaml.FullLoader)["stages"][stage][
                    "params"
                ]
            key_list.extend(new_keys)
        params = read_subset_of_params(key_list, params)
        params = merge_params(default_params, params)
    elif isinstance(params, str) and Path(params).is_file() and Path(params).exists():
        with open(Path(params), "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        assert isinstance(
            params,
            dict,
        ), f"Params in file {params} must be a dict. It is a {type(params)}."
        key_list = []
        for stage in stages:
            with open(Path(path, "dvc.yaml"), "r") as f:
                new_keys = yaml.load(f, Loader=yaml.FullLoader)["stages"][stage][
                    "params"
                ]
            key_list.extend(new_keys)
        with open(Path(path, "params.yaml"), "r") as f:
            all_params = yaml.load(f, Loader=yaml.FullLoader)
        default_params = read_subset_of_params(key_list, all_params)
        params = merge_params(default_params, params)
    elif isinstance(params, dict):
        key_list = []
        for stage in stages:
            with open(Path(path, "dvc.yaml"), "r") as f:
                new_keys = yaml.load(f, Loader=yaml.FullLoader)["stages"][stage][
                    "params"
                ]
            key_list.extend(new_keys)
        with open(Path(path, "params.yaml"), "r") as f:
            all_params = yaml.load(f, Loader=yaml.FullLoader)
        default_params = read_subset_of_params(key_list, all_params)
        params = merge_params(default_params, params)
    else:
        raise TypeError(f"Expected str or dict, got {type(params)}")
    assert isinstance(
        params,
        dict,
    ), f"Params must be a dict. It is type {type(params)}."
    # Load files from dvc
    with open(Path(path, "dvc.yaml"), "r") as f:
        key_list = []
        for stage in stages:
            pipe = yaml.load(f, Loader=yaml.FullLoader)["stages"][stage]
            if "deps" in pipe:
                key_list.extend(pipe["deps"])
            if "outs" in pipe:
                key_list.extend(pipe["outs"])
            if "metrics" in pipe:
                key_list.extend(pipe["metrics"])
    with open(Path(path, "params.yaml"), "r") as f:
        all_params = yaml.load(f, Loader=yaml.FullLoader)
    files = {}
    for filename, file in all_params["files"].items():
        if filename in str(key_list):
            files[filename] = file
    files["_target_"] = "deckard.base.files.FileConfig"
    params = get_files(params, stage=stages[-1])
    return params


def write_stage(params: dict, stage: str, path=None, working_dir=None) -> None:
    """
    Write params to dvc.yaml
    :param params: Params to write to dvc.yaml
    :param stage: Stage to write params to
    """
    if path is None:
        path = Path.cwd()
    if working_dir is None:
        working_dir = Path.cwd()
    with open(Path(working_dir, "dvc.yaml"), "r") as f:
        dvc = yaml.load(f, Loader=yaml.FullLoader)
    stage_params = {"stages": {stage: {}}}
    stage_params["stages"][stage] = dvc["stages"][stage]
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
    scorer = cfg.pop("optimizers", None)
    working_dir = cfg.pop("working_dir", Path().resolve().as_posix())
    stage = cfg.pop("stage", None)
    cfg = parse_stage(params=cfg, stage=stage, path=working_dir)
    exp = instantiate(cfg)
    files = deepcopy(exp.files)()
    folder = Path(files["score_dict_file"]).parent
    Path(folder).mkdir(exist_ok=True, parents=True)
    write_stage(cfg, stage, path=folder, working_dir=working_dir)
    id_ = Path(files["score_dict_file"]).parent.name
    direction = cfg.pop("direction", "minimize")
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
    config_path = os.environ.pop(
        "DECKARD_CONFIG_PATH",
        str(Path(Path(), "conf").absolute().as_posix()),
    )
    config_name = os.environ.pop("DECKARD_DEFAULT_CONFIG", "default.yaml")

    @hydra.main(config_path=config_path, config_name=config_name, version_base="1.3")
    def hydra_optimise(cfg: DictConfig) -> float:
        score = optimise(cfg)
        return score

    hydra_optimise()
