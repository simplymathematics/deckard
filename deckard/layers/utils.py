import logging
from pathlib import Path

from hydra.errors import OverrideParseException
from omegaconf import OmegaConf
from omegaconf import SCMode
from copy import deepcopy
import yaml
from hydra import initialize_config_dir, compose


from numpy import nan
from ..base.utils import my_hash

logger = logging.getLogger(__name__)

deckard_nones = [
    None,
    "None",
    "",
    "nan",
    "NAN",
    "null",
    "NULL",
    "Null",
    "none",
    "NONE",
    nan,
    "NaN",
]


def find_conf_files(
    config_subdir,
    config_dir,
    config_name=None,
    config_regex=None,
    default_file=None,
):
    if config_name is not None:
        assert config_regex is None, "Cannot specify both config_name and config_regex"
        config_dir = Path(Path(), config_dir).resolve().as_posix()
        sub_dir = Path(config_dir, config_subdir)
        files = [Path(sub_dir, config_name)]
    elif config_regex is not None:
        assert config_name is None, "Cannot specify both config_name and config_regex"
        config_dir = Path(Path(), config_dir).resolve().as_posix()
        sub_dir = Path(config_dir, config_subdir)
        files = sub_dir.glob(config_regex)
    elif default_file is not None:
        assert config_name is None, "Cannot specify both config_name and config_regex"
        config_dir = Path(Path(), config_dir).resolve().as_posix()
        sub_dir = Path(config_dir, config_subdir)
        files = [default_file]
    else:  # pragma: no cover
        raise ValueError(
            "Must specify either config_name or config_regex or default_file",
        )
    files = [file.as_posix() for file in files]
    return files


def get_overrides(overrides=None):
    if overrides is None:
        overrides = {}
    else:
        if isinstance(overrides, str):
            overrides = overrides.split(",")
        if isinstance(overrides, list):
            overrides = {
                entry.split("=")[0]: entry.split("=")[1] for entry in overrides
            }
        if isinstance(overrides, dict):
            new_dict = deepcopy(overrides)
            for k, v in new_dict.items():
                if k.startswith("++"):
                    overrides[k] = v
                elif k.startswith("+"):
                    overrides[f"++{k[1:]}"] = v
                elif k.startswith("~~"):
                    overrides[f"~~{k[2:]}"] = v
                else:
                    overrides[f"++{k}"] = v

        # assert isinstance(overrides, dict), f"Expected list, got {type(overrides)}"
    # if key is not None and len(overrides) > 0:
    #     overrides.pop(f"{key}.name", None)
    #     overrides.pop(f"files.{key}_file", None)
    #     overrides[f"++{key}.name"] = Path(file).stem
    #     overrides[f"++files.{key}_file"] = Path(file).stem
    #     overrides[f"{key}"] = Path(file).stem
    #     overrides["++stage"] = key
    return overrides


def compose_experiment(file, config_dir, overrides=None, default_file="default.yaml"):
    if hasattr(file, "as_posix"):
        file = file.as_posix()
    if overrides in [None, "", "None", "none", "NONE", "null", "Null", "NULL"]:
        overrides = []
    elif isinstance(overrides, str):
        overrides = overrides.split(",")
    if isinstance(overrides, list):
        pass
    elif isinstance(overrides, dict):
        new_dict = deepcopy(overrides)
        for k, v in new_dict.items():
            if k.startswith("++"):
                overrides[k] = v
            elif k.startswith("+"):
                overrides[f"++{k[1:]}"] = v
            elif k.startswith("--"):
                overrides[f"++{k[2:]}"] = v
            else:
                overrides[f"++{k}"] = v
    else:
        raise TypeError(f"Expected list or dict, got {type(overrides)}")
    assert isinstance(file, str), f"Expected str, got {type(file)}"
    # file = Path(data_conf_dir, file).as_posix()
    logger.info(f"Running experiment in config_dir: {config_dir}")
    logger.info(f"Running experiment with config_name: {file}")
    config_dir = Path(Path(), config_dir).resolve().as_posix()
    OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        try:
            cfg = compose(config_name=Path(default_file).stem, overrides=overrides)
        except OverrideParseException:
            raise ValueError(f"Failed to parse overrides: {overrides}")
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg["_target_"] = "deckard.Experiment"
        id_ = str(my_hash(cfg))
        cfg["name"] = id_
        cfg["files"]["name"] = id_
        return cfg


def save_params_file(
    config_dir="conf",
    config_file="default",
    params_file="params.yaml",
    working_directory = ".",
    overrides=[],
):
    config_dir = str(Path(working_directory, config_dir).absolute().as_posix())
    logger.info(f"Running save_params_file in config_dir: {config_dir}")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name=config_file, overrides=overrides)
    params = OmegaConf.to_container(cfg, resolve=True, structured_config_mode=SCMode.DICT)
    with open(params_file, "w") as f:
        yaml.dump(params, f)
    logger.info(f"Saved params file to {params_file}")
    assert Path(params_file).exists(), f"Failed to save params file to {params_file}"
    return None
