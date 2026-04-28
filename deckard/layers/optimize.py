import logging
import argparse
import json
from pathlib import Path
import yaml
import optuna
from numpy import nan

from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from hydra._internal.utils import get_args_parser

import json

from ..experiment import ExperimentConfig
from ..utils import ConfigBase
from ..file import FileConfig

# Set up logging
logger = logging.getLogger(__name__)

def optimize_multirun(cfg: ConfigBase, hydra_cfg, conf_obj: ExperimentConfig) -> dict:
    """
    Handles optimization in multirun mode.
    
    Parameters
    ----------
    cfg : ConfigBase
        The validated configuration object.
    hydra_cfg : HydraConfig
        The Hydra configuration object.
    conf_obj : ExperimentConfig
        The experiment conf_obj instance.
    
    Returns
    -------
    dict
        The filtered optimization scores.
    """
    assert hasattr(conf_obj, "files"), "conf_obj must have files attribute in multirun mode."
    assert hasattr(conf_obj, "optimizers"), "conf_obj must have optimizers attribute in multirun mode."
    assert hasattr(conf_obj, "directions"), "conf_obj must have directions attribute in multirun mode."
    conf_obj = prepare_multirun_file_paths(hydra_cfg, conf_obj)
    files = conf_obj.files.to_dict()
    logger.info(f"Saving multirun parameters to {conf_obj.files.params_file}")
    with open(files["params_file"], "w") as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=False), f)
    scores = conf_obj.execute()
    
    # Filter scores according to the optimizer. Directions pass +/- infinit
    optimizers = conf_obj.optimizers if hasattr(conf_obj, "optimizers") else []
    directions = conf_obj.directions if hasattr(conf_obj, "directions") else []
    filtered_scores, attributes = filter_scores(scores, optimizers, directions)
    
    assert "storage" in hydra_cfg.sweeper, "Storage must be specified in the sweeper config."
    assert "study_name" in hydra_cfg.sweeper, "Study name must be specified in the sweeper config."
    
    storage = hydra_cfg.sweeper.storage
    study_name = hydra_cfg.sweeper.study_name
    study = create_study(study_name, storage, directions, optimizers)
    set_study_metric_names(study=study, optimizers=optimizers)
    set_user_attrs(study=study, attrs=attributes)
    
    logger.info(f"Saving multirun scores to {conf_obj.files.score_file}") 
    with open(files["score_file"], "w") as f:
        json.dump(scores, f, indent=4)
    
    return filtered_scores




def optimize_main(
    cfg: ConfigBase,
) -> dict | tuple[dict, ConfigBase]:
    """
    Parameters
    ----------
    cfg : ConfigBase
        The configuration object to be used for optimization. It is converted
        to a dictionary-like structure for processing.

    Returns
    ----------
    dict: returns the scores as a dictionary.

    Notes
    ----
    - If the `cfg` contains an "optimizers" key, the scores are filtered to include
      only those corresponding to the specified optimizers.
    - If the `cfg` contains a "files" key, it is used to initialize a `FileConfig` object.
    - The function initializes an experiment configuration or conf_obj based on the `cfg`
      and executes the optimization process.
    """
    hydra_cfg = HydraConfig.get()
    mode = hydra_cfg.mode
    cfg = OmegaConf.to_container(cfg)
    cfg['_target_'] = cfg.get("_target_", "deckard.ExperimentConfig")
    
    conf_obj = instantiate(cfg)
    assert isinstance(conf_obj, ConfigBase), f"conf_obj must be an instance of ConfigBase. Got {type(conf_obj)}"
    if str(mode) == "RunMode.MULTIRUN":
        assert isinstance(conf_obj, ExperimentConfig)
        scores = optimize_multirun(cfg, hydra_cfg, conf_obj)
    else:
        scores = conf_obj.execute()
    
    return scores




def prepare_multirun_file_paths(hydra_cfg, conf_obj):
    conf_obj.experiment_name = f"{hydra_cfg.job.num}"
    conf_obj.__post_init__()
    # Set up log, score, and params file paths
    log_dir = Path(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
    log_file = log_dir / f"{hydra_cfg.job.name}.log"
    score_file = log_dir / "scores.json"
    params_file = log_dir / "params.yaml"
    error_file = log_dir/ "error.log"
    conf_obj.files.experiment_name = f"{hydra_cfg.job.num}"
    conf_obj.files.log_file = log_file.as_posix()
    conf_obj.files.score_file = score_file.as_posix()
    conf_obj.files.params_file = params_file.as_posix()
    conf_obj.files.error_file = error_file.as_posix()
    conf_obj.files.__post_init__()
    return conf_obj

def create_study(study_name, storage, directions, optimizers):
    assert len(directions) == len(optimizers), "Length of directions must match length of optimizers."
    if len(directions) == 0:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            directions=directions,
            load_if_exists=True,
            
        )
    return study

def set_study_metric_names(study, optimizers):
    if isinstance(optimizers, ListConfig):
        optimizers = list(optimizers)
    elif isinstance(optimizers, str):
        optimizers = [optimizers]
    elif isinstance(optimizers, tuple):
        optimizers = list(optimizers)
    else:
        raise ValueError(f"optimizers must be a ListConfig, str, or tuple. Got {type(optimizers)}")

    if hasattr(study, "set_metric_names") and len(optimizers) > 0:
        study.set_metric_names(optimizers)

def set_user_attrs(study, attrs):
    if isinstance(attrs, DictConfig):
        attrs = dict(attrs)
    for k, v in attrs.items():
        study.set_user_attr(key=k, value=v)
    

def save_params_file(cfg, files):
    _ = cfg.pop("params", None)
    if "params_file" in files:
        cfg = OmegaConf.create(cfg)
        Path(files["params_file"]).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, files["params_file"])
    else:
        raise ValueError("params_file must be specified in files to save parameters.")
    return cfg



def filter_scores(scores: dict, optimizers: list, directions: list) -> dict:
    """
    Overview
    ---
    Filters and processes the scores dictionary based on the specified optimizers
    and directions.

    Parameters
    ----------
    scores : dict
        A dictionary containing the scores to be filtered and processed.
    optimizers : list
        A list of optimizer names to filter the scores. If empty, all scores are returned.
    directions : list
        A list of directions ("minimize", "maximize", or "diff") corresponding to the
        optimizers. Used to further process the filtered scores.

    Returns
    -------
    dict
        A dictionary containing the filtered and processed scores.

    Raises
    -------
    ValueError
        - If the length of `directions` does not match the length of `optimizers`.
        - If an invalid direction is provided.
        - If no optimization scores are found for the specified directions.

    Notes
    -------
    - If `optimizers` is empty, the function returns the original `scores` dictionary.
    - The `directions` parameter is used to determine how the scores are processed:
        - "minimize" or "maximize": Adds the score to the optimization scores.
        - "diff": Adds the score to the attributes.
    - If no valid optimization scores are found, a `ValueError` is raised.
    """
    if not optimizers:
        return scores, {}
    other_scores = {k: v for k, v in scores.items() if k not in optimizers}
    scores = {k: v for k, v in scores.items() if k in optimizers}
    missing_scores = set(optimizers) - set(scores.keys())
    values = list(scores.values())
    if directions:
        assert len(directions) == len(
            optimizers,
        ), f"Length of directions must match length of optimizers. Got {len(directions)} and {len(optimizers)}."
        optimize_scores = []
        attributes = {}
        for i, direction in enumerate(directions):
            key = optimizers[i]
            if key in missing_scores:
                attributes[key] = float(nan)
            else:
                if direction in ["minimize", "maximize"]:
                    optimize_scores.append(scores[key])
                elif direction == "diff":
                    attributes[key] = scores[key]
                else:
                    raise ValueError(f"Invalid direction: {direction}")
        if not optimize_scores:
            raise ValueError("No optimization scores found for the specified directions.")
        if len(missing_scores) > 0:
            logger.error(f"Missing scores: {missing_scores}")
        values = optimize_scores
    else:
        attributes = {}
    attributes.update(other_scores)
    values = tuple(values)
    if isinstance(values, (tuple, list)) and len(values) == 1:
        values = values[0]
    logger.info(f"Optimization values: {values}")
    logger.info(f"Experiment attributes: {attributes}")
    return values, attributes



hydra_parser = argparse.ArgumentParser(
    parents=[get_args_parser()],
    add_help=False,
    usage="deckard optimize --config-dir=conf --config-name=default.yaml",
)