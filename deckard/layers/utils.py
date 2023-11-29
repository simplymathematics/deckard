import logging
from pathlib import Path

from hydra.errors import OverrideParseException
from omegaconf import OmegaConf
import yaml
from hydra import initialize_config_dir, compose


from numpy import nan
from ..base.utils import my_hash, flatten_dict

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


def get_overrides(file: str, folder, overrides=None):
    with open(Path(folder, file), "r") as f:
        old_cfg = yaml.safe_load(f)
    old_cfg = OmegaConf.create(old_cfg)
    old_cfg = OmegaConf.to_container(old_cfg, resolve=True)
    flat_cfg = flatten_dict(old_cfg)
    overrides = [] if overrides is None else overrides
    if isinstance(overrides, str):
        overrides = overrides.split(",")
    assert isinstance(overrides, list), f"Expected list, got {type(overrides)}"
    new_overrides = []
    for override in overrides:
        k, v = override.split("=")
        if k in flat_cfg:
            k = f"++{k}"
        elif k not in flat_cfg and not k.startswith("+"):
            k = f"+{k}"
        else:
            pass
        new_overrides.append(f"{k}={v}")
    overrides = new_overrides
    return overrides


def compose_experiment(file, config_dir, overrides=None, default_file="default.yaml"):
    overrides = get_overrides(file=file, folder=config_dir, overrides=overrides)
    logger.info(f"Running experiment in config_dir: {config_dir}")
    logger.info(f"Running experiment with config_name: {file}")
    config_dir = Path(Path(), config_dir).resolve().as_posix()
    OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        try:
            cfg = compose(config_name=Path(default_file).stem, overrides=overrides)
        except OverrideParseException:  # pragma: no cover
            raise ValueError(f"Failed to parse overrides: {overrides}")
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg["_target_"] = "deckard.Experiment"
        id_ = str(my_hash(cfg))
        cfg["name"] = id_
        cfg["files"]["name"] = id_
        cfg = OmegaConf.create(cfg)
    return cfg


def save_params_file(
    config_dir="conf",
    config_file="default",
    params_file="params.yaml",
    overrides=[],
):
    config_dir = str(Path(Path(), config_dir).absolute().as_posix())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name=config_file, overrides=overrides)
        params = OmegaConf.to_container(cfg, resolve=True)
        with open(params_file, "w") as f:
            yaml.dump(params, f)
        logger.info(f"Saved params file to {params_file}")
    assert Path(params_file).exists(), f"Failed to save params file to {params_file}"
    return None
