import logging
import argparse
import json
from pathlib import Path
import yaml
import optuna
from typing import Any
from hydra.experimental.callback import Callback as HydraCallback

from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from hydra._internal.utils import get_args_parser


from ..experiment import ExperimentConfig
from ..utils import ConfigBase, hash_conf_values

# Set up logging
logger = logging.getLogger(__name__)


class OptunaStudyCallback(HydraCallback):
    """Hydra-native callback that syncs study setup and metric names for multirun."""

    def __init__(self, study_name: str, storage: str, directions: list, optimizers: list):
        self.study_name = study_name
        self.storage = storage
        self.directions = directions
        self.optimizers = optimizers
        self.study = None

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        self.study = create_study(
            study_name=self.study_name,
            storage=self.storage,
            directions=self.directions,
            optimizers=self.optimizers,
        )
        set_study_metric_names(
            study=self.study,
            optimizers=self.optimizers,
            directions=self.directions,
        )

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        if self.study is None:
            return
        set_study_metric_names(
            study=self.study,
            optimizers=self.optimizers,
            directions=self.directions,
        )

    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        return

    def on_job_end(
        self,
        config: DictConfig,
        **kwargs: Any,
    ) -> None:
        return


def _ensure_experiment_hash(value) -> str:
    raw = "" if value is None else str(value).strip()
    if len(raw) == 32 and all(c in "0123456789abcdefABCDEF" for c in raw):
        return raw.lower()
    return hash_conf_values(value)


def optimize_multirun(
    cfg: ConfigBase,
    hydra_cfg,
    conf_obj: ExperimentConfig,
) -> dict:
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
    assert hasattr(
        conf_obj,
        "files",
    ), "conf_obj must have files attribute in multirun mode."
    assert hasattr(
        conf_obj,
        "optimizers",
    ), "conf_obj must have optimizers attribute in multirun mode."
    assert hasattr(
        conf_obj,
        "directions",
    ), "conf_obj must have directions attribute in multirun mode."
    conf_obj = prepare_multirun_file_paths(hydra_cfg, conf_obj)
    files = conf_obj.files._get_file_dict()
    logger.info(f"Saving multirun parameters to {conf_obj.files.params_file}")
    cfg_params = yaml.safe_load(cfg) if isinstance(cfg, str) else cfg
    params_file = files.get("params_file")
    if params_file is not None:
        params_path = Path(params_file)
        params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            yaml.dump(cfg_params, f, indent=4)
    scores = conf_obj.execute_without_mercy()
    # Filter scores according to the optimizer. Directions pass +/- infinit
    optimizers = conf_obj.optimizers if hasattr(conf_obj, "optimizers") else []
    directions = conf_obj.directions if hasattr(conf_obj, "directions") else []
    filtered_scores, _ = filter_scores(scores, optimizers, directions)
    assert (
        "storage" in hydra_cfg.sweeper
    ), "Storage must be specified in the sweeper config."
    assert (
        "study_name" in hydra_cfg.sweeper
    ), "Study name must be specified in the sweeper config."

    score_file = files.get("score_file")
    if score_file is not None:
        score_path = Path(score_file)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        with open(score_path, "w") as f:
            json.dump(scores, f, indent=4)

    return filtered_scores


def set_study_attributes(study, attrs):
    if isinstance(attrs, DictConfig):
        attrs = dict(attrs)
    for k, v in attrs.items():
        study.set_user_attr(key=k, value=v)


def optimize_main(
    cfg: ConfigBase,
) -> dict | tuple[dict, ConfigBase]:
    """
        Parameters
        ----------
    ƒ    cfg : ConfigBase
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

    if str(mode) == "RunMode.MULTIRUN":
        assert (
            "storage" in hydra_cfg.sweeper
        ), "Storage must be specified in the sweeper config."
        assert (
            "study_name" in hydra_cfg.sweeper
        ), "Study name must be specified in the sweeper config."
        explicit_name = cfg.get("experiment_name", None)
        if explicit_name is None or str(explicit_name).strip() == "":
            cfg["experiment_name"] = hash_conf_values(_root_=cfg)
        else:
            cfg["experiment_name"] = _ensure_experiment_hash(explicit_name)
        cfg["experiment_name"] = _ensure_experiment_hash(cfg.get("experiment_name"))

    cfg["_target_"] = cfg.get("_target_", "deckard.ExperimentConfig")
    cfg_yaml = OmegaConf.to_yaml(cfg)

    conf_obj = instantiate(cfg)
    assert isinstance(
        conf_obj,
        ConfigBase,
    ), f"conf_obj must be an instance of ConfigBase. Got {type(conf_obj)}"
    if str(mode) == "RunMode.MULTIRUN":
        assert isinstance(conf_obj, ExperimentConfig)
        scores = optimize_multirun(cfg_yaml, hydra_cfg, conf_obj)
    else:
        scores = conf_obj()
    return scores


def prepare_multirun_file_paths(hydra_cfg, conf_obj):
    current_name = getattr(conf_obj, "experiment_name", None)
    if current_name is None or str(current_name).strip() == "":
        if hasattr(conf_obj, "to_dict"):
            conf_obj.experiment_name = hash_conf_values(conf_obj.to_dict())
        else:
            conf_obj.experiment_name = hash_conf_values(str(conf_obj))
    else:
        conf_obj.experiment_name = _ensure_experiment_hash(current_name)
    conf_obj.__post_init__()
    # Set up log, score, and params file paths
    log_dir = Path(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
    log_file = log_dir / f"{hydra_cfg.job.name}.log"
    score_file = log_dir / "scores.json"
    params_file = log_dir / "params.yaml"
    error_file = log_dir / "error.log"
    conf_obj.experiment_name = _ensure_experiment_hash(conf_obj.experiment_name)
    conf_obj.files.log_file = log_file.as_posix()
    conf_obj.files.score_file = score_file.as_posix()
    conf_obj.files.params_file = params_file.as_posix()
    conf_obj.files.error_file = error_file.as_posix()
    conf_obj.files.__post_init__()
    return conf_obj


def create_study(study_name, storage, directions, optimizers):
    directions, optimizers = _filter_optuna_objectives(directions, optimizers)
    assert len(directions) == len(
        optimizers,
    ), "Length of directions must match length of optimizers."
    if len(directions) == 0 and len(optimizers) > 0:
        raise RuntimeError(
            "No Optuna objectives remain after filtering directions; "
            "at least one optimizer direction must be minimize/maximize.",
        )
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


def _normalize_direction(direction: str) -> str:
    d = str(direction).strip().lower()
    if "." in d:
        d = d.split(".")[-1]
    if d in ["maximize", "max"]:
        return "maximize"
    if d in ["minimize", "min"]:
        return "minimize"
    if d == "diff":
        return "diff"
    raise ValueError(f"Invalid direction: {direction}")


def _filter_optuna_objectives(directions, optimizers):
    if isinstance(directions, ListConfig):
        directions = list(directions)
    elif directions is None:
        directions = []

    if isinstance(optimizers, ListConfig):
        optimizers = list(optimizers)
    elif isinstance(optimizers, tuple):
        optimizers = list(optimizers)
    elif isinstance(optimizers, str):
        optimizers = [optimizers]
    elif optimizers is None:
        optimizers = []

    if len(directions) == 0:
        return directions, optimizers

    normalized_directions = [_normalize_direction(d) for d in directions]
    assert len(normalized_directions) == len(
        optimizers,
    ), "Length of directions must match length of optimizers."

    filtered = [
        (direction, optimizer)
        for direction, optimizer in zip(normalized_directions, optimizers)
        if direction != "diff"
    ]
    if len(filtered) == 0:
        return [], []
    filtered_directions, filtered_optimizers = zip(*filtered)
    return list(filtered_directions), list(filtered_optimizers)


def set_study_metric_names(study, optimizers, directions=None):
    if isinstance(optimizers, ListConfig):
        optimizers = list(optimizers)
    elif isinstance(optimizers, str):
        optimizers = [optimizers]
    elif isinstance(optimizers, tuple):
        optimizers = list(optimizers)
    elif isinstance(optimizers, list):
        pass
    else:
        raise ValueError(
            f"optimizers must be a ListConfig, str, or tuple. Got {type(optimizers)}",
        )

    if directions is not None:
        _, optimizers = _filter_optuna_objectives(directions, optimizers)

    if hasattr(study, "set_metric_names") and len(optimizers) > 0:
        study.set_metric_names(optimizers)


def set_trial_attributes(study, attrs, experiment_name):
    if isinstance(attrs, DictConfig):
        attrs = OmegaConf.to_container(attrs, resolve=True)

    if not attrs:
        return

    if not isinstance(attrs, dict):
        raise TypeError(f"attrs must be a dict-like object. Got {type(attrs)}")

    exp_uuid = _ensure_experiment_hash(experiment_name)
    trials = study.get_trials(deepcopy=False)
    trial = next(
        (t for t in trials if getattr(t, "user_attrs", {}).get("experiment_name") == exp_uuid),
        None,
    )

    if trial is None:
        logger.warning(
            "Skipping trial attribute sync: trial with experiment_name=%s not found in study '%s'.",
            exp_uuid,
            study.study_name,
        )
        return

    # `study.get_trials()` returns FrozenTrial objects; write attrs through storage.
    trial_id = getattr(trial, "_trial_id", None)
    if trial_id is None:
        trial_id = getattr(trial, "trial_id", None)

    if exp_uuid is not None:
        attrs = {**attrs, "experiment_name": exp_uuid}

    for k, v in attrs.items():
        if isinstance(v, (DictConfig, ListConfig)):
            v = OmegaConf.to_container(v, resolve=True)
        if trial_id is not None and hasattr(study, "_storage"):
            study._storage.set_trial_user_attr(trial_id, k, v)
        elif hasattr(trial, "set_user_attr"):
            # Fallback for tests/mocks that expose mutable trial helpers.
            trial.set_user_attr(k, v)
        else:
            raise RuntimeError(
                f"Unable to set trial attribute '{k}' for experiment_name={exp_uuid}; "
                "no Optuna storage handle found.",
            )


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
                if direction == "minimize":
                    optimize_scores.append(float("inf"))
                elif direction == "maximize":
                    optimize_scores.append(float("-inf"))
                else:
                    attributes[key] = float("inf")
            else:
                if direction in ["minimize", "maximize"]:
                    optimize_scores.append(scores[key])
                elif direction == "diff":
                    attributes[key] = scores[key]
                else:
                    raise ValueError(f"Invalid direction: {direction}")
        if not optimize_scores:
            raise RuntimeError(
                "No optimization scores found for the specified directions.",
            )
        if len(missing_scores) > 0:
            logger.warning(
                "Missing optimizer scores %s; using direction-aware fallback values.",
                missing_scores,
            )
        values = optimize_scores
    else:
        attributes = {}
    attributes.update(**other_scores)
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
