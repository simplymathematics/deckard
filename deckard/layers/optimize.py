import argparse
from dataclasses import dataclass, field
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Protocol, cast

import optuna
from hydra._internal.utils import get_args_parser
from hydra.core.hydra_config import HydraConfig
from hydra.experimental.callback import Callback as HydraCallback
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from optuna.storages._rdb import models as _optuna_rdb_models
from optuna.storages._rdb.storage import (
    _create_scoped_session as _optuna_scoped_session,
)

from ..experiment import ExperimentConfig
from ..experiment.canon import (
    CANONICAL_EXPERIMENT_STAGE_COMPONENTS,
    build_experiment_params_manifest,
    normalize_experiment_stage,
)
from ..file import FileConfig
from ..utils import BaseConfig, hash_conf_values

# Set up logging
logger = logging.getLogger(__name__)


OptimizerScalar = str | int | float | bool | None
OptimizerValue = OptimizerScalar | list["OptimizerValue"] | dict[str, "OptimizerValue"]

_RUNTIME_FILE_HASH_EXCLUDE = {
    "log_file",
    "score_file",
    "params_file",
    "error_file",
}

_RUNTIME_PARAM_EXCLUDE_KEYS = {
    "_X",
    "_y",
    "_model",
    "score_dict",
    "attack",
    "detector",
    "training_predictions",
    "predictions",
    "probabilities",
    "val_predictions",
    "val_probabilities",
    "dataset_obj",
    "loaders",
}


class OptimizerRuntimeConfigLike(Protocol):
    """Structural protocol for runtime configs exposing optimizer fields."""

    optimizers: list[str] | None
    directions: list[str] | None


@dataclass
class OptimizerConfig:
    """Runtime optimization policy object used by Hydra callback adapters.

    This object owns optimization policy state (objectives, study metadata,
    trial-attribute reporting knobs, and optional DVCLive flags) while callback
    classes own Hydra lifecycle hooks.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    directions: list[str] = field(
        default_factory=list, metadata={"help": "Configuration field: directions."}
    )
    optimizers: list[str] = field(
        default_factory=list, metadata={"help": "Configuration field: optimizers."}
    )
    study_name: str | None = None
    storage: str | None = None
    report_trial_attrs: bool = True
    pruning_enabled: bool = True
    dvclive_enabled: bool = False
    dvclive_dir: str | None = None

    @classmethod
    def from_any(
        cls,
        value: "OptimizerConfig | DictConfig | Mapping[str, OptimizerValue] | None",
        *,
        directions: list[str] | None = None,
        optimizers: list[str] | None = None,
        study_name: str | None = None,
        storage: str | None = None,
    ) -> "OptimizerConfig":
        """Normalize optimizer policy declarations into an OptimizerConfig instance.

        Args:
            value: Policy declaration payload.
            directions: Explicit optimization directions override.
            optimizers: Explicit optimizer names override.
            study_name: Optional explicit study name override.
            storage: Optional explicit storage URI override.

        Returns:
            Normalized optimizer policy object.
        """
        if isinstance(value, cls):
            cfg = value
        else:
            raw: dict[str, Any] = {}
            if isinstance(value, DictConfig):
                container = OmegaConf.to_container(value, resolve=True)
                if isinstance(container, dict):
                    raw = dict(container)
            elif isinstance(value, Mapping):
                raw = dict(value)
            cfg = cls(
                directions=[str(item) for item in list(raw.get("directions") or [])],
                optimizers=[str(item) for item in list(raw.get("optimizers") or [])],
                study_name=cast(str | None, raw.get("study_name")),
                storage=cast(str | None, raw.get("storage")),
                report_trial_attrs=bool(raw.get("report_trial_attrs", True)),
                pruning_enabled=bool(raw.get("pruning_enabled", True)),
                dvclive_enabled=bool(raw.get("dvclive_enabled", False)),
                dvclive_dir=cast(str | None, raw.get("dvclive_dir")),
            )

        # Explicit callback constructor args take precedence over embedded policy values.
        if directions is not None:
            cfg.directions = [str(item) for item in directions]
        if optimizers is not None:
            cfg.optimizers = [str(item) for item in optimizers]
        if study_name is not None:
            cfg.study_name = str(study_name)
        if storage is not None:
            cfg.storage = str(storage)
        return cfg

    def resolve_study_binding(
        self,
        hydra_cfg: DictConfig | Mapping[str, OptimizerValue],
    ) -> tuple[str | None, str | None]:
        """Resolve effective Optuna study name and storage from runtime Hydra config.

        Args:
            hydra_cfg: Runtime Hydra configuration payload.

        Returns:
            Effective study name and storage URI tuple.
        """
        sweeper = _get_sweeper_cfg(hydra_cfg)
        sweeper_study_name = (
            sweeper.get("study_name") if isinstance(sweeper, dict) else None
        )
        sweeper_storage = sweeper.get("storage") if isinstance(sweeper, dict) else None
        return self.study_name or sweeper_study_name, self.storage or sweeper_storage

    def create_study(self, *, study_name: str, storage: str) -> optuna.study.Study:
        """Create an Optuna study using current optimizer policy settings.

        Args:
            study_name: Optuna study name.
            storage: Optuna storage URI.

        Returns:
            Created Optuna study.
        """
        return create_study(
            study_name=study_name,
            storage=storage,
            directions=self.directions,
            optimizers=self.optimizers,
        )

    def set_metric_names(self, study: optuna.study.Study) -> None:
        """Apply configured metric names/directions onto the provided study object.

        Args:
            study: Optuna study to update with metric-name metadata.
        """
        set_study_metric_names(
            study=study,
            optimizers=self.optimizers,
            directions=self.directions,
        )

    def resolve_score_policy(
        self,
        config: "OptimizerRuntimeConfigLike | Mapping[str, OptimizerValue] | DictConfig",
    ) -> tuple[list[str], list[str]]:
        """Resolve optimizer names and directions from explicit policy or runtime config.

        Args:
            config: Runtime config payload exposing optimizer fields.

        Returns:
            Optimizer names and directions.
        """
        if self.optimizers:
            optimizers = list(self.optimizers)
        else:
            optimizers = list(getattr(config, "optimizers", []) or [])

        if self.directions:
            directions = list(self.directions)
        else:
            directions = list(getattr(config, "directions", []) or [])

        return optimizers, directions

    def merge_from_runtime_config(
        self,
        config: "DictConfig | Mapping[str, OptimizerValue]",
    ) -> None:
        """Merge runtime optimizer fields into this policy object in-place.

        Args:
            config: Runtime optimizer policy payload.
        """
        if isinstance(config, DictConfig):
            cfg = OmegaConf.to_container(config, resolve=True)
        elif isinstance(config, Mapping):
            cfg = dict(config)
        else:
            cfg = {}
        if not isinstance(cfg, dict):
            return

        if "directions" in cfg and cfg.get("directions") is not None:
            self.directions = [str(item) for item in list(cfg.get("directions") or [])]
        if "optimizers" in cfg and cfg.get("optimizers") is not None:
            self.optimizers = [str(item) for item in list(cfg.get("optimizers") or [])]
        if "report_trial_attrs" in cfg:
            self.report_trial_attrs = bool(cfg.get("report_trial_attrs"))
        if "pruning_enabled" in cfg:
            self.pruning_enabled = bool(cfg.get("pruning_enabled"))
        if "dvclive_enabled" in cfg:
            self.dvclive_enabled = bool(cfg.get("dvclive_enabled"))
        if "dvclive_dir" in cfg and cfg.get("dvclive_dir") is not None:
            self.dvclive_dir = str(cfg.get("dvclive_dir"))
        if "study_name" in cfg and cfg.get("study_name") is not None:
            self.study_name = str(cfg.get("study_name"))
        if "storage" in cfg and cfg.get("storage") is not None:
            self.storage = str(cfg.get("storage"))


class DefaultOptimizerCallback(HydraCallback):
    """
    Hydra-native callback that syncs study setup and metric names for multirun.

    Args:
        directions: List of optimization directions ("minimize"/"maximize").
        optimizers: List of optimizer names.
        study_name: Optional Optuna study name.
        storage: Optional Optuna storage URI.
        directory: Optional output directory.
        log_file: Optional log file path.
        params_file: Optional params file path.
        error_file: Optional error file path.
        score_file: Optional score file path.

    Note:
        Resolved per-job file paths are set during `on_compose_config`.
    """

    def __init__(
        self,
        directions: list | None = None,
        optimizers: list | None = None,
        study_name: str | None = None,
        storage: str | None = None,
        directory: str | None = None,
        log_file: str | None = None,
        params_file: str | None = None,
        error_file: str | None = None,
        score_file: str | None = None,
        optimizer: Any | None = None,
    ):
        self.optimizer = OptimizerConfig.from_any(
            optimizer,
            directions=list(directions or []),
            optimizers=list(optimizers or []),
            study_name=study_name,
            storage=storage,
        )
        self.study_name = self.optimizer.study_name
        self.storage = self.optimizer.storage
        self.directions = self.optimizer.directions
        self.optimizers = self.optimizer.optimizers
        self.study = None
        self._directory = directory
        self._log_file = log_file
        self._params_file = params_file
        self._error_file = error_file
        self._score_file = score_file
        # Resolved per-job paths set during on_compose_config
        self._resolved_params_file: str | None = None
        self._resolved_score_file: str | None = None
        self._resolved_log_file: str | None = None
        self._resolved_error_file: str | None = None

    def _configure_policy(self, config: Any) -> None:
        self.optimizer.merge_from_runtime_config(config)
        # Keep legacy attribute access stable.
        self.study_name = self.optimizer.study_name
        self.storage = self.optimizer.storage
        self.directions = self.optimizer.directions
        self.optimizers = self.optimizer.optimizers

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Create the Optuna study and initialize objective metric names.

        Args:
            config: Hydra config for the multirun.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If `study_name` or `storage` is not provided.
        """
        self._configure_policy(config)
        hydra_cfg = HydraConfig.get()
        study_name, storage = self.optimizer.resolve_study_binding(hydra_cfg)
        if study_name is None or storage is None:
            raise ValueError(
                "study_name and storage must be provided for multirun study setup",
            )
        self.study = self.optimizer.create_study(
            study_name=study_name,
            storage=storage,
        )
        self.optimizer.set_metric_names(self.study)

    def on_compose_config(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Prepare per-job naming, output paths, and persist params.yaml.

        Args:
            config: Hydra config for the current job.
            **kwargs: Additional keyword arguments.

        Note:
            Writes params.yaml after resolving file paths.
        """
        self._configure_policy(config)
        hydra_cfg = HydraConfig.get()
        _normalize_mode_cfg(config, hydra_cfg, include_file_paths=True)
        _seed_experiment_uuid_for_current_trial(
            hydra_cfg=hydra_cfg,
            experiment_name=getattr(config, "experiment_name", ""),
        )

        # Resolve per-job file paths: constructor params take priority, then
        # Hydra output directories (single-run and multirun), then no-op.
        hydra_paths = _resolve_multirun_paths(hydra_cfg)

        def _resolve(attr_val: str | None, key: str) -> str | None:
            if attr_val is not None:
                return attr_val
            return hydra_paths.get(key)

        self._resolved_log_file = _resolve(self._log_file, "log_file")
        self._resolved_params_file = _resolve(self._params_file, "params_file")
        self._resolved_error_file = _resolve(self._error_file, "error_file")
        self._resolved_score_file = _resolve(self._score_file, "score_file")

        files_cfg = FileConfig.from_payload(getattr(config, "files", None))
        resolved = {
            "log_file": self._resolved_log_file,
            "params_file": self._resolved_params_file,
            "error_file": self._resolved_error_file,
            "score_file": self._resolved_score_file,
        }
        files_cfg.apply_runtime_paths(**resolved)
        files_cfg.hash_artifact_paths(
            _runtime_context_file_hash(
                hydra_cfg,
                getattr(config, "experiment_name", None),
            ),
            exclude=_RUNTIME_FILE_HASH_EXCLUDE,
        )
        config["files"] = files_cfg.to_runtime_dict()

        # Write params.yaml now that paths are resolved.
        if self._resolved_params_file:
            save_params_file(
                config,
                {"params_file": str(self._resolved_params_file)},
                files_config=files_cfg,
                hydra_cfg=hydra_cfg,
            )

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Validate and normalize run-mode config early in the Hydra lifecycle.

        Args:
            config: Hydra config for the run.
            **kwargs: Additional keyword arguments.
        """
        self._configure_policy(config)
        hydra_cfg = HydraConfig.get()
        _normalize_mode_cfg(config, hydra_cfg, include_file_paths=False)
        if self.study is not None:
            self.optimizer.set_metric_names(self.study)

    def on_run_end(self, config: DictConfig, **kwargs: Any) -> None:
        """
        No-op run hook kept for parity with multirun lifecycle wiring.

        Args:
            config: Hydra config for the run.
            **kwargs: Additional keyword arguments.
        """
        _ = config
        _ = kwargs

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Ensure metric names remain attached after the multirun completes.

        Args:
            config: Hydra config for the multirun.
            **kwargs: Additional keyword arguments.
        """
        self._configure_policy(config)
        if self.study is None:
            return
        self.optimizer.set_metric_names(self.study)

    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Seed experiment UUID for the current trial before execution.

        Args:
            config: Hydra config for the job.
            **kwargs: Additional keyword arguments.
        """
        self._configure_policy(config)
        hydra_cfg = HydraConfig.get()
        _seed_experiment_uuid_for_current_trial(
            hydra_cfg=hydra_cfg,
            experiment_name=getattr(config, "experiment_name", ""),
        )
        _ = kwargs

    def on_job_end(
        self,
        config: DictConfig,
        job_return,
        **kwargs: Any,
    ) -> None:
        """
        Persist per-job score payload after execution when available.

        Args:
            config: Hydra config for the job.
            job_return: Return value from the job execution.
            **kwargs: Additional keyword arguments.
        """
        self._configure_policy(config)
        hydra_cfg = HydraConfig.get()
        score_file = self._resolved_score_file
        if not score_file:
            score_file = _resolve_multirun_paths(hydra_cfg).get("score_file")
            self._resolved_score_file = score_file
        if not score_file:
            return
        job_return_payload = _extract_scores_from_job_end_kwargs(
            job_return=job_return,
            kwargs=kwargs,
        )

        score_payload_dict: dict[str, Any] = {}
        if isinstance(job_return_payload, dict):
            score_payload_dict.update(
                {str(k): v for k, v in job_return_payload.items()},
            )

        # If Hydra sweeper wraps return values, recover the raw score payload that
        # optimize_main stashes onto config from BaseConfig.__call__.
        config_payload = None
        if isinstance(config, DictConfig):
            config_payload = config.get("_deckard_score_payload", None)
        elif isinstance(config, dict):
            config_payload = config.get("_deckard_score_payload", None)
        if isinstance(config_payload, DictConfig):
            config_payload = OmegaConf.to_container(config_payload, resolve=True)
        if isinstance(config_payload, dict):
            score_payload_dict.update({str(k): v for k, v in config_payload.items()})

        if not score_payload_dict:
            return
        score_payload = _inject_experiment_name(
            score_payload_dict,
            getattr(config, "experiment_name", ""),
        )
        if not isinstance(score_payload, dict):
            return
        score_payload_dict = {str(k): v for k, v in score_payload.items()}
        optimizers, directions = self.optimizer.resolve_score_policy(config)
        if optimizers:
            _, attrs = filter_scores(
                score_payload_dict,
                optimizers,
                directions,
                emit_logs=False,
            )
        else:
            attrs = dict(score_payload_dict)
        if self.optimizer.report_trial_attrs:
            _sync_multirun_trial_attributes(
                hydra_cfg=hydra_cfg,
                score_payload=score_payload_dict,
                optimizers=optimizers,
                directions=directions,
                experiment_name=getattr(config, "experiment_name", ""),
                attrs=attrs,
            )
        score_path = Path(str(score_file))
        score_path.parent.mkdir(parents=True, exist_ok=True)
        with open(score_path, "w") as f:
            json.dump(score_payload_dict, f, indent=4)
        return

    @staticmethod
    def execute_runtime_object(conf_obj: Any) -> dict[str, Any]:
        """
        Execute runtime object once and normalize its score payload to a dict.

        Args:
            conf_obj: Callable config object or object with `execute_without_mercy`.

        Returns:
            dict[str, Any]: Score payload as a dictionary.

        Raises:
            TypeError: If the object is not callable or does not return a dict.
        """
        if callable(conf_obj):
            scores = conf_obj()
        elif hasattr(conf_obj, "execute_without_mercy") and callable(
            getattr(conf_obj, "execute_without_mercy"),
        ):
            scores = conf_obj.execute_without_mercy()
        else:
            raise TypeError(
                "conf_obj must be callable or implement execute_without_mercy",
            )

        if isinstance(scores, DictConfig):
            scores = OmegaConf.to_container(scores, resolve=True)
        if not isinstance(scores, dict):
            raise TypeError(
                f"Runtime object must return a dict-like score payload. Got {type(scores)}",
            )
        return {str(k): v for k, v in scores.items()}


def _ensure_experiment_hash(value) -> str:
    fingerprint = getattr(value, "fingerprint", None)
    if isinstance(fingerprint, str):
        token = fingerprint.strip()
        if len(token) == 32 and all(c in "0123456789abcdefABCDEF" for c in token):
            return token.lower()

    raw = "" if value is None else str(value).strip()
    if len(raw) == 32 and all(c in "0123456789abcdefABCDEF" for c in raw):
        return raw.lower()
    return hash_conf_values(value)


def _runtime_context_file_hash(hydra_cfg: Any, experiment_name: Any) -> str:
    """Build a deterministic runtime-context hash for artifact file naming."""
    payload = {
        "experiment_name": _ensure_experiment_hash(experiment_name),
        "job_id": _get_hydra_job_identifier(hydra_cfg),
        "job_name": getattr(getattr(hydra_cfg, "job", None), "name", None),
        "sweep_subdir": getattr(getattr(hydra_cfg, "sweep", None), "subdir", None),
    }
    return hash_conf_values(payload)


def _hash_file_path_with_runtime_context(path_value: str, context_hash: str) -> str:
    """Replace file basename with runtime-context hash while preserving parent/suffix."""
    path = Path(path_value)
    suffix = "".join(path.suffixes)
    hashed_name = f"{context_hash}{suffix}" if suffix else context_hash
    return (path.parent / hashed_name).as_posix()


def _apply_runtime_context_file_hashes(
    files_cfg: Any,
    *,
    hydra_cfg: Any,
    experiment_name: Any,
) -> None:
    """Hash configured ``*_file`` artifact paths using runtime context."""
    if files_cfg is None:
        return

    context_hash = _runtime_context_file_hash(hydra_cfg, experiment_name)

    if isinstance(files_cfg, DictConfig):
        for key, value in list(files_cfg.items()):
            key_token = str(key)
            if (
                key_token.endswith("_file")
                and key_token not in _RUNTIME_FILE_HASH_EXCLUDE
                and isinstance(value, str)
                and value.strip() != ""
            ):
                files_cfg[key_token] = _hash_file_path_with_runtime_context(
                    value,
                    context_hash,
                )
        return

    if isinstance(files_cfg, dict):
        for key, value in list(files_cfg.items()):
            key_token = str(key)
            if (
                key_token.endswith("_file")
                and key_token not in _RUNTIME_FILE_HASH_EXCLUDE
                and isinstance(value, str)
                and value.strip() != ""
            ):
                files_cfg[key_token] = _hash_file_path_with_runtime_context(
                    value,
                    context_hash,
                )


def _is_multirun_mode(hydra_cfg) -> bool:
    return str(getattr(hydra_cfg, "mode", "")) == "RunMode.MULTIRUN"


def _is_run_mode(hydra_cfg) -> bool:
    return str(getattr(hydra_cfg, "mode", "")) == "RunMode.RUN"


def _get_sweeper_cfg(hydra_cfg):
    sweeper = getattr(hydra_cfg, "sweeper", None)
    if isinstance(sweeper, DictConfig):
        return OmegaConf.to_container(sweeper, resolve=True)
    return sweeper


def _assert_multirun_sweeper(hydra_cfg):
    sweeper = _get_sweeper_cfg(hydra_cfg)
    assert sweeper is not None, "Sweeper must be specified in multirun mode."
    assert "storage" in sweeper, "Storage must be specified in the sweeper config."
    assert (
        "study_name" in sweeper
    ), "Study name must be specified in the sweeper config."


def _resolve_multirun_sweep_paths(hydra_cfg) -> dict:
    log_dir = Path(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
    return {
        "log_file": (log_dir / f"{hydra_cfg.job.name}.log").as_posix(),
        "score_file": (log_dir / "scores.json").as_posix(),
        "params_file": (log_dir / "params.yaml").as_posix(),
        "error_file": (log_dir / "error.log").as_posix(),
    }


def _resolve_run_paths(hydra_cfg) -> dict[str, str]:
    run_dir_value = getattr(getattr(hydra_cfg, "run", None), "dir", None)
    if run_dir_value is None:
        run_dir_value = getattr(
            getattr(hydra_cfg, "runtime", None),
            "output_dir",
            None,
        )
    if run_dir_value is None:
        return {}
    run_dir = Path(run_dir_value)
    job_name = getattr(getattr(hydra_cfg, "job", None), "name", "run")
    return {
        "log_file": (run_dir / f"{job_name}.log").as_posix(),
        "score_file": (run_dir / "scores.json").as_posix(),
        "params_file": (run_dir / "params.yaml").as_posix(),
        "error_file": (run_dir / "error.log").as_posix(),
    }


def _resolve_multirun_paths(hydra_cfg) -> dict[str, str]:
    """Resolve output paths from Hydra run.dir or sweep.dir/sweep.subdir."""
    if not _is_multirun_mode(hydra_cfg):
        return _resolve_run_paths(hydra_cfg)

    has_multirun_paths = (
        hasattr(hydra_cfg, "sweep")
        and hasattr(hydra_cfg, "job")
        and getattr(getattr(hydra_cfg, "sweep", None), "dir", None) is not None
        and getattr(getattr(hydra_cfg, "sweep", None), "subdir", None) is not None
        and getattr(getattr(hydra_cfg, "job", None), "name", None) is not None
    )
    if has_multirun_paths:
        return _resolve_multirun_sweep_paths(hydra_cfg)

    has_run_paths = (
        hasattr(hydra_cfg, "run")
        and getattr(getattr(hydra_cfg, "run", None), "dir", None) is not None
    )
    if has_run_paths:
        return _resolve_run_paths(hydra_cfg)
    return {}


def _get_hydra_job_identifier(hydra_cfg):
    """Return Hydra's stable job id for trial/attribute synchronization."""
    job_cfg = getattr(hydra_cfg, "job", None)
    job_id = getattr(job_cfg, "id", None)
    if job_id is None:
        return None
    job_id_str = str(job_id)
    if job_id_str.isdigit():
        return job_id_str

    # Joblib launcher can emit ids like '__main___0'; Optuna trial numbers use
    # the trailing numeric index, so normalize to that when available.
    trial_suffix = job_id_str.rsplit("_", 1)[-1]
    if trial_suffix.isdigit():
        return trial_suffix
    return job_id


def _seed_experiment_uuid_for_current_trial(hydra_cfg, experiment_name) -> None:
    """Attach experiment UUID to the currently executing Optuna trial."""
    sweeper = _get_sweeper_cfg(hydra_cfg)
    if not isinstance(sweeper, dict):
        return

    storage = sweeper.get("storage")
    study_name = sweeper.get("study_name")
    trial_number = _get_hydra_job_identifier(hydra_cfg)
    if not storage or not study_name or trial_number is None:
        return

    study = optuna.study.load_study(storage=storage, study_name=study_name)
    exp_uuid = _ensure_experiment_hash(experiment_name)
    trials = list(study.get_trials(deepcopy=False))
    selected_trial = next(
        (t for t in trials if str(getattr(t, "number", None)) == str(trial_number)),
        None,
    )
    if selected_trial is None:
        return

    trial_id = getattr(
        selected_trial,
        "_trial_id",
        getattr(selected_trial, "trial_id", None),
    )
    if trial_id is None or not hasattr(study, "_storage"):
        return
    study._storage.set_trial_user_attr(trial_id, "experiment_name", exp_uuid)


def _inject_experiment_name(score_payload, experiment_name):
    if not isinstance(score_payload, dict):
        return score_payload
    exp = str(experiment_name).strip()
    if not exp:
        return score_payload
    exp_uuid = _ensure_experiment_hash(exp)
    return {**score_payload, "experiment_name": exp_uuid}


def _overwrite_frozen_trial_user_attr(
    study,
    trial_id: int,
    key: str,
    value: Any,
) -> bool:
    """Bypass Optuna's finished-trial mutability guard for RDB-backed studies."""
    storage = getattr(study, "_storage", None)
    backend = getattr(storage, "_backend", None)
    if (
        backend is None
        or _optuna_rdb_models is None
        or _optuna_scoped_session is None
        or not hasattr(backend, "scoped_session")
    ):
        return False

    try:
        with _optuna_scoped_session(backend.scoped_session, True) as session:
            trial_model = _optuna_rdb_models.TrialModel.find_or_raise_by_id(
                trial_id,
                session,
            )
            attribute = (
                _optuna_rdb_models.TrialUserAttributeModel.find_by_trial_and_key(
                    trial_model,
                    key,
                    session,
                )
            )
            value_json = json.dumps(value)
            if attribute is None:
                attribute = _optuna_rdb_models.TrialUserAttributeModel(
                    trial_id=trial_id,
                    key=key,
                    value_json=value_json,
                )
                session.add(attribute)
            else:
                attribute.value_json = value_json
    except Exception:
        return False
    return True


def _prepare_multirun_cfg(cfg, hydra_cfg, include_file_paths: bool = False):
    stage_payload = _build_stage_dependent_hash_payload(cfg)
    explicit_name = cfg.get("experiment_name", None)
    if explicit_name is None or str(explicit_name).strip() == "":
        cfg["experiment_name"] = _ensure_experiment_hash(
            hash_conf_values(stage_payload),
        )
    else:
        cfg["experiment_name"] = _ensure_experiment_hash(explicit_name)

    if (
        include_file_paths
        and hasattr(hydra_cfg, "sweep")
        and hasattr(hydra_cfg, "job")
        and getattr(getattr(hydra_cfg, "sweep", None), "dir", None) is not None
        and getattr(getattr(hydra_cfg, "sweep", None), "subdir", None) is not None
        and getattr(getattr(hydra_cfg, "job", None), "name", None) is not None
    ):
        file_paths = _resolve_multirun_sweep_paths(hydra_cfg)
        files_cfg = cfg.get("files")
        if isinstance(files_cfg, DictConfig):
            for k, v in file_paths.items():
                files_cfg[k] = v
        elif isinstance(files_cfg, dict):
            files_cfg.update(file_paths)
        else:
            cfg["files"] = file_paths
    return cfg


def _normalize_mode_cfg(cfg, hydra_cfg, include_file_paths: bool = False):
    """Normalize config shape for run/multirun while keeping mode deltas explicit."""
    if _is_multirun_mode(hydra_cfg):
        _assert_multirun_sweeper(hydra_cfg)
        return _prepare_multirun_cfg(
            cfg,
            hydra_cfg,
            include_file_paths=include_file_paths,
        )
    if _is_run_mode(hydra_cfg):
        explicit_name = cfg.get("experiment_name", None)
        if explicit_name is None or str(explicit_name).strip() == "":
            cfg["experiment_name"] = _ensure_experiment_hash(
                hash_conf_values(_build_stage_dependent_hash_payload(cfg)),
            )
        return cfg
    return cfg


def _build_stage_dependent_hash_payload(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(cfg, dict):
        return {"stage": "all", "cfg": str(cfg)}

    stage_token = cfg.get("stage", "all")
    try:
        stage = normalize_experiment_stage(stage_token)
    except Exception:
        stage = str(stage_token or "all")

    selected_components = CANONICAL_EXPERIMENT_STAGE_COMPONENTS.get(stage)
    if selected_components is None:
        # Union of known stage participants when stage selection is unknown/"all".
        component_union: set[str] = set()
        for components in CANONICAL_EXPERIMENT_STAGE_COMPONENTS.values():
            component_union.update(components)
        selected_components = tuple(sorted(component_union))
    payload: dict[str, Any] = {
        "stage": stage,
        "components": {key: cfg.get(key) for key in selected_components if key in cfg},
        "runtime": {
            key: cfg.get(key)
            for key in (
                "library",
                "classifier",
                "evaluation_mode",
                "score_mode",
                "random_state",
                "optimizers",
                "directions",
                "report_trial_attrs",
                "pruning_enabled",
                "dvclive_enabled",
                "dvclive_dir",
            )
            if key in cfg
        },
    }
    return payload


def _extract_scores_from_job_end_kwargs(
    job_return=None,
    kwargs: dict | None = None,
):
    if job_return is None:
        kwargs = kwargs or {}
        job_return = kwargs.get("job_return")
    if job_return is None:
        return None

    score_payload = getattr(job_return, "return_value", None)
    if score_payload is None and isinstance(job_return, dict):
        score_payload = job_return.get("return_value", None)
    if score_payload is None:
        return None
    if isinstance(score_payload, DictConfig):
        score_payload = OmegaConf.to_container(score_payload, resolve=True)
    return score_payload


def _sync_multirun_trial_attributes(
    hydra_cfg,
    score_payload,
    experiment_name,
    optimizers=None,
    directions=None,
    attrs=None,
) -> None:
    """Sync non-optimized scores as trial attributes for the current multirun trial."""
    if not isinstance(score_payload, dict):
        return

    sweeper = _get_sweeper_cfg(hydra_cfg)
    if not isinstance(sweeper, dict):
        return
    storage = sweeper.get("storage")
    study_name = sweeper.get("study_name")
    if not storage or not study_name:
        return

    if attrs is None:
        _, attrs = filter_scores(
            score_payload,
            list(optimizers or []),
            list(directions or []),
            emit_logs=False,
        )
    if not attrs:
        return

    try:
        study = optuna.study.load_study(storage=storage, study_name=study_name)
    except KeyError:
        logger.warning(
            "Skipping trial attribute sync: study '%s' not found in storage '%s'.",
            study_name,
            storage,
        )
        return

    set_trial_attributes(
        study=study,
        attrs=attrs,
        experiment_name=str(experiment_name),
        trial_number=_get_hydra_job_identifier(hydra_cfg),
    )


def set_study_attributes(
    study: optuna.study.Study,
    attrs: dict[str, Any] | DictConfig,
) -> None:
    """Attach user attributes to an Optuna study."""
    if isinstance(attrs, DictConfig):
        attrs_container = OmegaConf.to_container(attrs, resolve=True)
        attrs = cast(dict[str, Any], attrs_container)
    if not isinstance(attrs, dict):
        raise TypeError(f"attrs must be dict-like. Got {type(attrs)}")
    for k, v in attrs.items():
        study.set_user_attr(key=str(k), value=v)


def optimize_main(
    cfg: Any,
) -> Any:
    """
    Run the optimize layer entrypoint and return an unfiltered score dictionary.

    Args:
        cfg: Layer configuration payload. This may be a `DictConfig` or any
            mapping-like object that can be normalized into a dictionary and then
            instantiated via Hydra.

    Returns:
        dict[str, Any]: Unfiltered score payload produced by the instantiated runtime object.

    Raises:
        AssertionError: If the config cannot be coerced to a dictionary or instantiated.
    """
    hydra_cfg = HydraConfig.get()
    cfg_dict = _coerce_cfg_to_dict(cfg)
    # Ensure callback-style interpolation keys always exist at root during
    # resolution, even when they are not constructor args for ExperimentConfig.
    cfg_dict.setdefault("dvclive_enabled", False)
    cfg_dict.setdefault("pruning_enabled", True)
    cfg_dict.setdefault("report_trial_attrs", True)
    cfg_dict.setdefault("dvclive_dir", None)

    # Resolve interpolations before dropping non-constructor root keys.
    cfg_resolved = OmegaConf.create(cfg_dict)
    OmegaConf.resolve(cfg_resolved)
    cfg_container = OmegaConf.to_container(cfg_resolved, resolve=True)
    assert isinstance(
        cfg_container,
        dict,
    ), f"cfg must resolve to a dictionary. Got {type(cfg_container)}"
    cfg_dict = {str(k): v for k, v in cfg_container.items()}
    optimizer_names = list(cfg_dict.get("optimizers", []) or [])
    optimizer_directions = list(cfg_dict.get("directions", []) or [])

    assert isinstance(
        cfg_dict,
        dict,
    ), f"cfg must resolve to a dictionary. Got {type(cfg_dict)}"

    # Path/file enforcement for multirun is now callback-owned in on_compose_config.
    _ = hydra_cfg

    # Optimize layer always executes an ExperimentConfig payload.
    # Some config compositions can leak a root `_target_` from global search
    # overrides; force the correct root target to avoid mis-instantiation.
    cfg_dict["_target_"] = "deckard.ExperimentConfig"
    cfg_dict = _filter_experiment_config_kwargs(cfg_dict)

    conf_obj = instantiate(cfg_dict)
    assert isinstance(
        conf_obj,
        BaseConfig,
    ), f"conf_obj must be an instance of BaseConfig. Got {type(conf_obj)}"
    scores = DefaultOptimizerCallback.execute_runtime_object(conf_obj)

    # Preserve raw runtime payload for callback hooks even when sweepers wrap
    # return values for objective extraction.
    try:
        if isinstance(cfg, DictConfig):
            cfg["_deckard_score_payload"] = dict(scores)
        elif isinstance(cfg, dict):
            cfg["_deckard_score_payload"] = dict(scores)
    except Exception:
        pass

    if _should_raise_trial_pruned(scores, cfg_dict):
        raise optuna.TrialPruned("Runtime marked trial as pruned.")

    # Optuna's Hydra sweeper expects scalar/tuple objective values. In child job
    # processes Hydra may report RUN mode even for a parent multirun, so rely on
    # configured optimizer objectives instead of mode tokens alone.
    objective_values, _ = filter_scores(
        scores=scores,
        optimizers=optimizer_names,
        directions=optimizer_directions,
        emit_logs=False,
    )
    if objective_values is not scores:
        return objective_values

    return scores


def _should_raise_trial_pruned(
    scores: Mapping[str, Any],
    cfg_dict: Mapping[str, Any],
) -> bool:
    """Return whether optimize_main should raise TrialPruned for this payload."""
    if not bool(cfg_dict.get("pruning_enabled", True)):
        return False
    return bool(scores.get("pruned", False))


def _filter_experiment_config_kwargs(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """Drop root keys that are not accepted by ``ExperimentConfig``.

    Compose-only metadata keys (for example aliases or optimizer labels) can be
    present at the root config level but are not constructor params for
    ``ExperimentConfig``. Keeping them causes Hydra instantiation to fail.
    """
    allowed = set(inspect.signature(ExperimentConfig).parameters.keys())
    filtered = {"_target_": cfg_dict.get("_target_", "deckard.ExperimentConfig")}
    for key, value in cfg_dict.items():
        if key == "_target_":
            continue
        if key in allowed:
            filtered[key] = value
    return filtered


def _coerce_cfg_to_dict(cfg: Any) -> dict[str, Any]:
    """Normalize incoming optimize config payload to a mutable dictionary."""
    if isinstance(cfg, DictConfig):
        cfg_container = OmegaConf.to_container(cfg, resolve=False)
        if not isinstance(cfg_container, dict):
            raise AssertionError(
                f"cfg must resolve to a dictionary. Got {type(cfg_container)}",
            )
        return {str(k): v for k, v in cfg_container.items()}
    if isinstance(cfg, dict):
        return dict(cfg)

    cfg_container = OmegaConf.to_container(OmegaConf.create(cfg), resolve=False)
    if not isinstance(cfg_container, dict):
        raise AssertionError(
            f"cfg must resolve to a dictionary. Got {type(cfg_container)}",
        )
    return {str(k): v for k, v in cfg_container.items()}


def prepare_multirun_file_paths(
    hydra_cfg: Any,
    conf_obj: ExperimentConfig,
) -> ExperimentConfig:
    """Populate standard output file paths for a Hydra multirun job."""
    current_name = getattr(conf_obj, "experiment_name", None)
    if current_name is None or str(current_name).strip() == "":
        conf_obj.experiment_name = _ensure_experiment_hash(conf_obj)
    else:
        conf_obj.experiment_name = _ensure_experiment_hash(current_name)
    if (
        not isinstance(conf_obj, ExperimentConfig)
        and hasattr(
            conf_obj,
            "__post_init__",
        )
        and callable(getattr(conf_obj, "__post_init__"))
    ):
        conf_obj.__post_init__()
    if conf_obj.files is None:
        from ..file import FileConfig

        conf_obj.files = FileConfig()
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
    if hasattr(conf_obj.files, "_get_file_dict") and hasattr(
        conf_obj.files,
        "_resolve_paths",
    ):
        conf_obj.files._file_dict = conf_obj.files._get_file_dict()
        conf_obj.files._resolve_paths()
        conf_obj.files._file_dict = conf_obj.files._get_file_dict()
        for key, value in conf_obj.files._file_dict.items():
            setattr(conf_obj.files, key, value)
    elif hasattr(conf_obj.files, "__post_init__") and callable(
        getattr(conf_obj.files, "__post_init__"),
    ):
        conf_obj.files.__post_init__()
    return conf_obj


def create_study(
    study_name: str,
    storage: str,
    directions: list[str] | tuple[str, ...] | ListConfig,
    optimizers: list[str] | tuple[str, ...] | ListConfig,
) -> optuna.study.Study:
    """Create or load an Optuna study after filtering non-optimizing objectives."""
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


def set_study_metric_names(
    study: Any,
    optimizers: Any,
    directions: Any = None,
) -> None:
    """Set Optuna metric names using optimizer keys after direction filtering."""
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


def set_trial_attributes(
    study: Any,
    attrs: Any,
    experiment_name: str,
    trial_number: Any = None,
) -> None:
    """Persist per-trial user attributes for one unambiguous target trial."""
    if isinstance(attrs, DictConfig):
        attrs = OmegaConf.to_container(attrs, resolve=True)

    if not attrs:
        return

    if not isinstance(attrs, dict):
        raise TypeError(f"attrs must be a dict-like object. Got {type(attrs)}")

    exp_uuid = _ensure_experiment_hash(experiment_name)
    attrs = {**attrs, "experiment_name": exp_uuid}
    trials = list(study.get_trials(deepcopy=False))
    if not trials:
        logger.warning(
            "Skipping trial attribute sync: no trials found in study '%s'.",
            study.study_name,
        )
        return

    selected_trial = None
    if trial_number is not None:
        trial_number_str = str(trial_number)
        for trial in trials:
            if str(getattr(trial, "number", None)) == trial_number_str:
                selected_trial = trial
                break
    elif len(trials) == 1:
        selected_trial = trials[0]
    else:
        unique_match = None
        match_count = 0
        for trial in trials:
            if getattr(trial, "user_attrs", {}).get("experiment_name") == exp_uuid:
                unique_match = trial
                match_count += 1
                if match_count > 1:
                    unique_match = None
                    break
        selected_trial = unique_match

    if selected_trial is None:
        logger.warning(
            "Skipping trial attribute sync: target trial not found in study '%s' (trial_number=%s).",
            study.study_name,
            trial_number,
        )
        return

    trial_id = getattr(selected_trial, "_trial_id", None)
    if trial_id is None:
        trial_id = getattr(selected_trial, "trial_id", None)

    for k, v in attrs.items():
        if isinstance(v, (DictConfig, ListConfig)):
            v = OmegaConf.to_container(v, resolve=True)
        if trial_id is not None and hasattr(study, "_storage"):
            try:
                study._storage.set_trial_user_attr(trial_id, k, v)
            except optuna.exceptions.UpdateFinishedTrialError:
                if not _overwrite_frozen_trial_user_attr(study, trial_id, k, v):
                    raise
        elif hasattr(selected_trial, "set_user_attr"):
            selected_trial.set_user_attr(k, v)
        else:
            raise RuntimeError(
                f"Unable to set trial attribute '{k}' for experiment_name={exp_uuid}; "
                "no Optuna storage handle found.",
            )


_DROP_RUNTIME_VALUE = object()


def _sanitize_initialization_payload(payload: Any) -> Any:
    """Return params-safe payload containing only initialization configuration data."""
    if isinstance(payload, FileConfig):
        return payload.to_init_dict()
    if isinstance(payload, DictConfig):
        payload = OmegaConf.to_container(payload, resolve=False)
    if isinstance(payload, ListConfig):
        payload = list(payload)

    if payload is None or isinstance(payload, (str, int, float, bool)):
        return payload

    if isinstance(payload, Path):
        return payload.as_posix()

    if isinstance(payload, dict):
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            key_token = str(key)
            if key_token in _RUNTIME_PARAM_EXCLUDE_KEYS:
                continue
            sanitized = _sanitize_initialization_payload(value)
            if sanitized is _DROP_RUNTIME_VALUE:
                continue
            cleaned[key_token] = sanitized
        return cleaned

    if isinstance(payload, (list, tuple, set)):
        cleaned_list: list[Any] = []
        for value in payload:
            sanitized = _sanitize_initialization_payload(value)
            if sanitized is _DROP_RUNTIME_VALUE:
                continue
            cleaned_list.append(sanitized)
        return cleaned_list

    if callable(payload):
        return _DROP_RUNTIME_VALUE

    return _DROP_RUNTIME_VALUE


def _build_params_payload(
    cfg: dict[str, Any] | DictConfig,
    *,
    files_config: FileConfig | None = None,
    hydra_cfg: Any | None = None,
) -> dict[str, Any]:
    init_payload = _coerce_cfg_to_dict(cfg)
    init_payload.pop("params", None)
    if files_config is not None:
        init_payload["files"] = files_config.to_init_dict()
    elif "files" in init_payload:
        init_payload["files"] = FileConfig.from_payload(
            init_payload.get("files"),
        ).to_init_dict()
    sanitized_init = _sanitize_initialization_payload(init_payload)
    assert isinstance(
        sanitized_init,
        dict,
    ), f"params init payload must be dict-like. Got {type(sanitized_init)}"

    payload: dict[str, Any] = {
        "init": sanitized_init,
        "derived": {
            "params_manifest": build_experiment_params_manifest(sanitized_init),
        },
    }

    runtime_payload: dict[str, Any] = {}
    if files_config is not None:
        runtime_payload["files"] = files_config.to_runtime_dict()
    if hydra_cfg is not None:
        runtime_payload["hydra"] = {
            "mode": str(getattr(hydra_cfg, "mode", "")) or None,
        }
    if runtime_payload:
        payload["runtime"] = runtime_payload
    return payload


def save_params_file(
    cfg: dict[str, Any] | DictConfig,
    files: dict[str, str],
    *,
    files_config: FileConfig | None = None,
    hydra_cfg: Any | None = None,
) -> DictConfig:
    """Persist run parameters to ``files['params_file']`` and return DictConfig."""
    if isinstance(cfg, DictConfig):
        cfg.pop("params", None)
    elif isinstance(cfg, dict):
        cfg.pop("params", None)
    if "params_file" in files:
        cfg = OmegaConf.create(
            _build_params_payload(
                cfg,
                files_config=files_config,
                hydra_cfg=hydra_cfg,
            ),
        )
        Path(files["params_file"]).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, files["params_file"])
    else:
        raise ValueError(
            "params_file must be specified in files to save parameters.",
        )
    return cfg


def filter_scores(
    scores: dict[str, Any],
    optimizers: list[str],
    directions: list[str],
    emit_logs: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """
    Overview
    ---
    Filters and processes the scores dictionary based on the specified optimizers
    and directions.

    Args:
        scores: Score dictionary to filter and process.
        optimizers: Optimizer names to select from ``scores``. When empty, all
            scores are returned.
        directions: Directions such as ``minimize``, ``maximize``, or ``diff``
            corresponding to ``optimizers``.
        emit_logs: Whether to emit informational logging during filtering.

    Returns:
        Tuple of optimization payload and filtered score dictionary.

    Raises:
        ValueError: If the number of directions does not match the number of
            optimizers, if a direction is invalid, or if no optimization scores
            are found for the requested directions.

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
    if emit_logs:
        logger.info(f"Optimization values: {values}")
        logger.info(f"Experiment attributes: {attributes}")
    return values, attributes


hydra_parser = argparse.ArgumentParser(
    parents=[get_args_parser()],
    add_help=False,
    usage="deckard optimize --config-dir=conf --config-name=default.yaml",
)
