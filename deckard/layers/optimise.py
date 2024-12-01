import logging
import os
import traceback
from pathlib import Path
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import hydra
from ..base.utils import my_hash, unflatten_dict
from .utils import deckard_nones

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)

__all__ = ["write_stage", "optimise", "parse_stage", "get_files"]

config_path = os.environ.get(
    "DECKARD_CONFIG_PATH",
    str(Path(Path.cwd(), "conf").absolute().as_posix()),
)
config_name = os.environ.get("DECKARD_DEFAULT_CONFIG", "default.yaml")
full_path = Path(config_path, config_name).as_posix()


def get_files(
    cfg,
    stage,
):
    """
    Gets the file names from cfg and calculates the hash of the attack, model and data, and files objects.
    If "files.name == 'default'", the name is set to the hash of the cfg.
    For attack, model and data, the file name is set to the hash of the respective object.
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
    cfg["files"]["name"] = (
        id_ if cfg["files"]["name"] == "default" else cfg["files"]["name"]
    )
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
        if key in default and isinstance(value, dict) and len(value) > 0:
            default[key] = merge_params(default[key], value)
        elif (
            isinstance(value, (list, tuple, int, float, str, bool, dict))
            and len(value) > 0
        ):
            default.update({key: value})
        elif value is None:
            default.update({key: {}})
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
    if isinstance(params, dict):
        key_list = []
        for stage in stages:
            stage = stage.split("@")[0]
            assert Path(
                path,
                "dvc.yaml",
            ).exists(), f"{Path(path, 'dvc.yaml')} does not exist."
            with open(Path(path, "dvc.yaml"), "r") as f:
                print()
                keys = yaml.load(f, Loader=yaml.FullLoader)["stages"]
                if stage in keys:
                    new_keys = keys[stage]
                if "foreach" in new_keys:
                    new_keys = new_keys["do"]["params"]
                else:
                    new_keys = new_keys["params"]

            key_list.extend(new_keys)
    else:
        raise TypeError(f"Expected dict, got {type(params)}")
    params = read_subset_of_params(key_list, params)
    # Load files from dvc
    with open(Path(path, "dvc.yaml"), "r") as f:
        pipe = yaml.load(f, Loader=yaml.FullLoader)
    file_list = []
    for stage in stages:
        if len(stage.split("@")) > 1:
            sub_stage = stage.split("@")[1]
            directory = stage.splits("@")[0]
            file_list.append(directory)
            file_list.append(sub_stage)
        else:
            sub_stage = None
        stage = stage.split("@")[0]
        pipe = pipe["stages"][stage]
        if "do" in pipe:
            pipe = pipe["do"]
        if "deps" in pipe:
            dep_list = [str(x).split(":")[0] for x in pipe["deps"]]
            file_list.extend(dep_list)
        if "outs" in pipe:
            out_list = [str(x).split(":")[0] for x in pipe["outs"]]
            file_list.extend(out_list)
        if "metrics" in pipe:
            metric_list = [str(x).split(":")[0] for x in pipe["metrics"]]
            file_list.extend(metric_list)
    file_string = str(file_list).replace("item.", "")
    files = params["files"]
    file_list = list(files.keys())
    for key in file_list:
        if key == "params.yaml":
            continue
        if key.endswith("_file") or key.endswith("_dir"):
            template_string = "${files." + key + "}"
            if template_string in file_string:
                pass
            else:
                params["files"].pop(key)
    params = get_files(params, stage)
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
    # with open(path / "dvc.yaml", "w") as f:
    #     yaml.dump(stage_params, f, default_flow_style=False)
    # assert Path(path / "dvc.yaml").exists(), f"File {path/'dvc.yaml'} does not exist."
    with open(Path(path, "params.yaml"), "w") as f:
        yaml.dump(params, f, default_flow_style=False)
    assert Path(
        path / "params.yaml",
    ).exists(), f"File {path/'params.yaml'} does not exist."
    return None


def optimise(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
    raise_exception = cfg.pop("raise_exception", True)
    working_dir = Path(config_path).parent
    direction = cfg.get("direction", "minimize")
    direction = [direction] if not isinstance(direction, list) else direction
    optimizers = cfg.get("optimizers", None)
    optimizers = [optimizers] if not isinstance(optimizers, list) else optimizers
    assert len(optimizers) == len(direction)
    stage = cfg.pop("stage", None)
    cfg = parse_stage(params=cfg, stage=stage, path=working_dir)
    exp = instantiate(cfg)
    files = exp.files.get_filenames()
    folder = Path(files["score_dict_file"]).parent
    Path(folder).mkdir(exist_ok=True, parents=True)
    write_stage(cfg, stage, path=folder, working_dir=working_dir)
    id_ = Path(files["score_dict_file"]).parent.name
    optimizers = [optimizers] if not isinstance(optimizers, list) else optimizers
    try:
        score_dict = exp()
        scores = []
        i = 0
        for optimizer in optimizers:
            if optimizer in score_dict:
                scores.append(score_dict[optimizer])
            else:
                if direction[i] == "minimize":
                    scores.append(1.00000000000)
                elif direction[i] == "maximize":
                    scores.append(0.00000000000)
                else:
                    scores.append(None)
            i += 1
        full_path = Path(folder).resolve().as_posix()
        # Assume it is a subpath of the working directory, and remove the working directory from the path
        rel_path = full_path.replace(Path(working_dir).resolve().as_posix(), ".")
        logger.info(f"Experiment Folder: {rel_path}")
        logger.info(f"Optimizers are : {optimizers}")
        logger.info(f"Score is : {scores}")
    except Exception as e:
        with open(Path(folder, "exception.log"), "w") as f:
            f.write(str(e))
            f.write(traceback.format_exc())
        logger.info(f"Exception: {e}")
        if not raise_exception:
            logger.warning(
                f"Exception {e} occured while running experiment {id_}. Setting score to default for specified direction (e.g. -/+ 1e10).",
            )
            fake_scores = []
            for direction in direction:
                if direction == "minimize":
                    fake_scores.append(1.00000000000)
                elif direction == "maximize":
                    fake_scores.append(0.00000000000)
                else:
                    fake_scores.append(None)
            scores = fake_scores
            logger.info(f"Optimizers: {optimizers}")
            logger.info(f"Score: {scores}")
        else:
            raise e
    if len(scores) == 1:
        scores = float(scores[0])
    else:
        scores = [float(x) for x in scores]
    return scores


@hydra.main(config_path=config_path, config_name=config_name, version_base="1.3")
def optimise_main(cfg: DictConfig) -> float:
    score = optimise(cfg)
    return score

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    assert Path(
        config_path,
    ).exists(), f"{config_path} does not exist. Please specify a config path by running `export DECKARD_CONFIG_PATH=<your/path/here>` "
    optimise_main()
