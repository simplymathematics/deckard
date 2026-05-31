"""Experiment orchestration primitives for deckard's Python API.

This module contains the base experiment configuration object that ties data,
model, defense, attack, files, and scorers into a single executable unit.
"""

import logging
import warnings
import hashlib
import time
import inspect
from dataclasses import dataclass, field
from typing import List, Union, Literal, Any, Mapping
from omegaconf import DictConfig, ListConfig, OmegaConf
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from hydra.utils import instantiate

from ..data import DataConfig
from ..model import ModelConfig

try:
    from ..data import FairlearnDataConfig
except ImportError:  # pragma: no cover
    FairlearnDataConfig = None
from ..model.defense.base import DefensePipelineConfig

try:
    from ..attack import AttackConfig
except Exception:  # pragma: no cover
    AttackConfig = None
from ..detector import DetectorConfig
from ..score import ScorerDictConfig
from ..file import FileConfig, data_files, model_files, attack_files
from ..utils import (
    BaseConfig,
    coerce_config,
    coerce_to_list,
    instantiate_config,
    is_default_config_value,
    is_null_config_value,
    load_class,
    merge_scores_with_collision_suffix,
    split_separated_tokens,
)
from ..score.base import coerce_scorer_config, _DataScorerMarker, _AttackProfileScorer
from ..data.sample import BaseSampler, KFoldSampler, ShuffleSampler
from ..plugins.base import HookBundle, compose_hook_plugins
from .canon import (
    CANONICAL_EXPERIMENT_PIPELINE_STAGES,
    CANONICAL_EXPERIMENT_TIMES,
    CANONICAL_EXPERIMENT_CACHE_STAGES,
    CANONICAL_EXPERIMENT_RUNTIME_SCHEMA_PREFIX,
    CANONICAL_EXPERIMENT_RUNTIME_SCHEMA_VERSION,
    build_experiment_hook_bundle,
    build_experiment_hook_graph,
    build_experiment_stage_cache_key,
    build_experiment_params_manifest,
    ensure_experiment_runtime_contract,
)

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from ..data import AnjanaDataConfig
except ImportError:  # pragma: no cover
    AnjanaDataConfig = None


FairlearnModelConfig = None
FairlearnPytorchModelConfig = None
PytorchModelConfig = None


def _load_optional_model_specializations() -> tuple[Any, Any, Any]:
    """Load optional fairness/torch model classes only when needed.

    Importing these symbols at module import time can transitively import the
    torch stack during unrelated test collection. Delay that resolution until
    fairness specialization actually runs.
    """

    global FairlearnModelConfig, FairlearnPytorchModelConfig, PytorchModelConfig

    if PytorchModelConfig is None:
        try:
            from ..frameworks.pytorch.model import (
                PytorchModelConfig as _PytorchModelConfig,
            )
        except Exception:  # pragma: no cover
            _PytorchModelConfig = None
        PytorchModelConfig = _PytorchModelConfig

    if FairlearnModelConfig is None:
        try:
            from ..plugins.fairlearn.model import (
                FairlearnModelConfig as _FairlearnModelConfig,
            )
        except Exception:  # pragma: no cover
            _FairlearnModelConfig = None
        FairlearnModelConfig = _FairlearnModelConfig

    if FairlearnPytorchModelConfig is None:
        try:
            from ..plugins.fairlearn.model import (
                FairlearnPytorchModelConfig as _FairlearnPytorchModelConfig,
            )
        except Exception:  # pragma: no cover
            _FairlearnPytorchModelConfig = None
        FairlearnPytorchModelConfig = _FairlearnPytorchModelConfig

    return PytorchModelConfig, FairlearnModelConfig, FairlearnPytorchModelConfig


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


DECKARD_CONFIG_DIR = os.environ.get("DECKARD_CONFIG_DIR", "config")
DECKARD_DEFAULT_CONFIG_FILE = os.environ.get(
    "DECKARD_DEFAULT_CONFIG_FILE",
    "default_experiment.yaml",
)


def _load_yaml_file(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


def _file_resolver(arg: str):
    """
    Usage:
      ${file:search/rf.yaml:model_search}
      ${file:./configs/search/rf.yaml:model_search.subkey}
      ${file:/abs/path/to/file.yaml}       -> returns whole file
    """
    if not arg:
        raise ValueError(
            "file resolver requires an argument like 'path/to/file.yaml[:key]'",
        )

    # split into path and optional key (only first ':' splits, keys may contain '.')
    if ":" in arg:
        path_part, key_part = arg.split(":", 1)
        key_part = key_part.strip()
    else:
        path_part, key_part = arg, None
    path = Path(DECKARD_CONFIG_DIR, path_part)
    if not path.exists():
        raise FileNotFoundError(
            f"file resolver: file not found: {path_part} in working dir {os.getcwd()}",
        )

    data = _load_yaml_file(path)
    # if user requested a nested key, walk the dict using dot-splitting
    if key_part:
        parts = key_part.split(".")
        cur = data
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                raise KeyError(
                    f"file resolver: key '{key_part}' not found in {path}",
                )
        data = cur
    data = OmegaConf.create(data)
    # Return as an OmegaConf node so structured content is preserved
    return data


# Register resolver with OmegaConf (Hydra will pick up this plugin module automatically)
OmegaConf.register_new_resolver(
    "file",
    _file_resolver,
    replace=True,
    use_cache=True,
)


def _merge_resolver(*args):
    """
    Merge multiple OmegaConf or dict objects into a single OmegaConf dict.
    Usage:
      ${merge:${file:search/rf.yaml:model_search}, ${file:search/class_labels.yaml:model_search}}
    """
    merged = OmegaConf.create()
    for arg in args:
        # Resolve any interpolations
        obj = OmegaConf.to_container(OmegaConf.create(arg), resolve=True)
        merged = OmegaConf.merge(merged, obj)
    return OmegaConf.create(merged)


OmegaConf.register_new_resolver("merge", _merge_resolver, replace=True)


@dataclass(eq=False, kw_only=True)
class ExperimentConfig(BaseConfig):
    """Compose and execute a complete deckard experiment.

    An experiment coordinates data loading, optional defense application, model
    training or loading, adversarial attack execution, scoring, and artifact
    persistence through ``FileConfig``.

    Note:
        ``evaluation_mode`` and ``score_mode`` are mutually exclusive to prevent
        ambiguous routing. Use ``evaluation_mode`` for preset routing
        (``standard``, ``tuning``, ``report``), or use ``score_mode`` for
        explicit split routing (``train``, ``test``, ``val``, ``all``),
        optionally as a list for multi-pass scoring.

    Attributes:
        data: Data configuration/runtime payload for the experiment.
        model: Model configuration/runtime payload.
        defense: Optional defense pipeline config.
        attack: Optional attack configuration.
        detector: Optional detector configuration.
        files: File configuration for artifact persistence.
        score: Experiment-level scorer configuration.
        evaluation_mode: Preset routing mode for scoring/evaluation.
        score_mode: Explicit split-mode override for scoring/evaluation.
    """

    _fairness_keys = {
        "sensitive_columns",
        "fairness_defense",
        "fairness_pipeline_step_name",
        "fairness_pipeline_step",
    }
    _anjana_keys = {
        "anjana_defense",
        "quasi_identifiers",
        "identifiers",
        "sensitive_attribute",
        "hierarchies",
        "hierarchy_interval_sizes",
    }

    def _data_to_dict(self, data_obj) -> dict:
        if isinstance(data_obj, DictConfig):
            data_dict = OmegaConf.to_container(data_obj, resolve=True)
        elif isinstance(data_obj, str):
            data_dict = DataConfig.from_yaml(data_obj).to_dict()
        elif isinstance(data_obj, BaseConfig):
            data_dict = data_obj.to_dict()
        elif isinstance(data_obj, dict):
            data_dict = data_obj
        else:
            raise ValueError(f"Unsupported type for data: {type(data_obj)}")
        if not isinstance(data_dict, dict):
            raise TypeError("Resolved data config must be a dictionary")
        return data_dict

    def _select_data_cls(self, data_dict: dict):
        if any(key in data_dict for key in self._anjana_keys):
            resolved_anjana_cls = AnjanaDataConfig
            if resolved_anjana_cls is None:
                try:
                    from ..plugins.anjana.data import (
                        AnjanaDataConfig as _AnjanaDataConfig,
                    )

                    resolved_anjana_cls = _AnjanaDataConfig
                except Exception as exc:
                    raise ImportError(
                        "AnjanaDataConfig requires optional anjana dependencies. Install deckard[anjana] to enable anjana data configs.",
                    ) from exc
            return resolved_anjana_cls
        if any(key in data_dict for key in self._fairness_keys):
            resolved_fairlearn_cls = FairlearnDataConfig
            if resolved_fairlearn_cls is None:
                try:
                    from ..plugins.fairlearn.data import (
                        FairlearnDataConfig as _FairlearnDataConfig,
                    )

                    resolved_fairlearn_cls = _FairlearnDataConfig
                except Exception as exc:
                    raise ImportError(
                        "FairlearnDataConfig requires optional fairness dependencies. Install deckard[fairlearn] to enable fairlearn data configs.",
                    ) from exc
            return resolved_fairlearn_cls
        return DataConfig

    def _resolve_data_config(self):
        if self.data is None:
            raise ValueError("data must be provided")

        if isinstance(self.data, DataConfig):
            return self.data

        if hasattr(self.data, "_target_") and not isinstance(
            self.data,
            (dict, DictConfig, str, BaseConfig),
        ):
            data_obj = instantiate(self.data)
            if not isinstance(data_obj, DataConfig):
                raise TypeError(
                    f"Object of type: {type(data_obj)} is not a DataConfig object.",
                )
            return data_obj

        data_dict = self._data_to_dict(self.data)
        data_cls = self._select_data_cls(data_dict)
        logger.info("Resolved data config class: %s", data_cls.__name__)
        data_obj = instantiate_config(
            data_dict,
            data_cls,
            default_target=f"{data_cls.__module__}.{data_cls.__name__}",
        )
        assert isinstance(data_obj, DataConfig), ValueError(
            f"Object of type: {type(data_obj)} is not a DataConfig object.",
        )
        return data_obj

    data: DataConfig
    experiment_name: str = "{hash}"
    model: ModelConfig = None
    defense: DefensePipelineConfig = None
    attack: AttackConfig = None
    detector: DetectorConfig = None
    files: FileConfig = None
    score: ScorerDictConfig = None
    random_state: int = 42
    library: Literal["sklearn", "tensorflow", "pytorch"] = "sklearn"
    device: Any = None
    classifier: Union[str, bool] = True
    evaluation_mode: Literal["standard", "tuning", "report"] = "standard"
    score_mode: Union[str, list[str], None] = field(default_factory=list)
    times: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    hook_plugins: list[Any] = field(default_factory=list)
    hook_bundles: list[Any] = field(default_factory=list)
    dvc_plugin: Any = None
    cache_enabled: bool = True

    RUNTIME_STATE_VERSION = CANONICAL_EXPERIMENT_RUNTIME_SCHEMA_VERSION
    PIPELINE_STAGE_ORDER = CANONICAL_EXPERIMENT_PIPELINE_STAGES
    HASH_EXCLUDE_FIELDS = BaseConfig.HASH_EXCLUDE_FIELDS | {"dvc_plugin"}

    def _has_explicit_score_mode(self) -> bool:
        if not hasattr(self, "score_mode"):
            return False
        if self.score_mode is None:
            return False
        if isinstance(self.score_mode, list):
            return len(self.score_mode) > 0
        return str(self.score_mode).strip() != ""

    def _validate_mode_configuration(self) -> None:
        """Ensure exactly one experiment mode-routing strategy is active."""
        # ``standard`` acts as the neutral preset, so it can coexist with
        # explicit ``score_mode`` without ambiguity.
        if self._has_explicit_score_mode() and self.evaluation_mode != "standard":
            raise ValueError(
                "evaluation_mode and score_mode are mutually exclusive. "
                "Set score_mode with evaluation_mode='standard', or unset score_mode.",
            )

    @staticmethod
    def _canonical_device(device_value: Any) -> Union[str, None]:
        if device_value is None:
            return None
        text = str(device_value).strip()
        if text == "":
            return None
        if text.lower() in {"none", "null", "auto", "best", "default"}:
            return None
        return text.lower()

    def _reconcile_component_devices(self):
        """No-op in the base class. Overridden by TorchExperimentConfig."""
        pass

    def _resolve_score_modes(self) -> list[str]:
        """Resolve concrete score modes from explicit score_mode or evaluation preset."""
        if self._has_explicit_score_mode():
            if isinstance(self.score_mode, list):
                raw_modes = list(self.score_mode)
            else:
                raw_modes = [self.score_mode]
        elif self.evaluation_mode == "standard":
            raw_modes = ["train", "test"]
        elif self.evaluation_mode == "tuning":
            raw_modes = ["test"]
        elif self.evaluation_mode == "report":
            raw_modes = ["train", "test", "val"]
        else:
            raise NotImplementedError(
                f"Evaluation mode: {self.evaluation_mode} not implemented",
            )

        allowed = {"train", "test", "val", "all"}
        modes = []
        for raw_mode in raw_modes:
            mode = str(raw_mode).strip().lower()
            if mode not in allowed:
                raise ValueError(
                    f"Unsupported score mode '{raw_mode}'. Expected one of: {sorted(allowed)}.",
                )
            modes.append(mode)
        return modes

    def _resolve_data_mode_inputs(self, mode: str) -> tuple[Any, Any]:
        if mode == "pre-sample":
            y_true = getattr(self.data, "_y", None)
            y_pred = getattr(self.data, "_X", None)
        elif mode == "train":
            y_true = getattr(self.data, "y_train", None)
            y_pred = getattr(self.data, "X_train", None)
        elif mode == "test":
            y_true = getattr(self.data, "y_test", None)
            y_pred = getattr(self.data, "X_test", None)
        elif mode == "val":
            y_true = getattr(self.data, "y_val", None)
            y_pred = getattr(self.data, "X_val", None)
        else:
            raise ValueError(f"Unsupported data scoring mode '{mode}'")

        if y_true is None or y_pred is None:
            raise ValueError(
                f"Scoring mode '{mode}' requested but required dataset split is unavailable.",
            )
        return y_true, y_pred

    @staticmethod
    def _apply_runtime_data_split_overrides(
        loaded_data: Any,
        configured_data: Any,
    ) -> None:
        """Apply split-related runtime config from *configured_data* to *loaded_data*."""
        if loaded_data is None or configured_data is None:
            return
        for attr in (
            "sampler",
            "split",
            "val_size",
            "train_size",
            "test_size",
            "stratify",
            "random_state",
        ):
            if hasattr(configured_data, attr):
                setattr(loaded_data, attr, getattr(configured_data, attr))
        if hasattr(loaded_data, "_sampler_obj"):
            loaded_data._sampler_obj = None

    def _ensure_active_mode_split_available(self) -> None:
        """Ensure required split exists for the active model score mode."""
        active_mode = self._resolve_component_score_mode()
        if active_mode != "val":
            return

        has_val = (
            getattr(self.data, "X_val", None) is not None
            and getattr(self.data, "y_val", None) is not None
        )
        if has_val:
            return

        can_resample = (
            callable(getattr(self.data, "fit", None))
            and getattr(self.data, "_X", None) is not None
            and getattr(self.data, "_y", None) is not None
        )
        if can_resample:
            self.data.data_sample_time = None
            for attr in (
                "train_indices",
                "test_indices",
                "val_indices",
                "X_train",
                "y_train",
                "X_test",
                "y_test",
                "X_val",
                "y_val",
                "train_n",
                "test_n",
                "val_n",
            ):
                setattr(self.data, attr, None)
            self.data.fit()

        if (
            getattr(self.data, "X_val", None) is None
            or getattr(self.data, "y_val", None) is None
        ):
            raise ValueError(
                "score_mode='val' requires validation data (X_val/y_val), but no validation split is available.",
            )

    def _resolve_component_score_mode(
        self,
    ) -> str:
        modes = self._resolve_score_modes()
        if not modes:
            return "test"

        # Respect explicit single-mode configuration as-is.
        if len(modes) == 1:
            mode = modes[0]
            if mode in {"test", "val", "train", "pre-sample"}:
                return mode

        # For multi-mode evaluations (e.g. standard/report), prefer test-mode
        # component scoring so model metrics retain canonical keys like
        # ``accuracy`` instead of ``training_accuracy``.
        for preferred in ("test", "val", "train", "pre-sample"):
            if preferred in modes:
                return preferred
        return "test"

    def _propagate_score_mode(self) -> str:
        """
        Propagate score mode to data, model, and attack configs.
        - DataConfig: supports pre-sample/train/test/val
        - ModelConfig: supports train/test/val (no pre-sample)
        - AttackConfig: supports test/val only
        """
        active_mode = self._resolve_component_score_mode()
        if self.data is not None and hasattr(self.data, "score_mode"):
            self.data.score_mode = (
                active_mode
                if active_mode in {"pre-sample", "train", "test", "val"}
                else "pre-sample"
            )
        if (
            self.model is not None
            and hasattr(self.model, "score_mode")
            and active_mode in {"train", "test", "val"}
        ):
            self.model.score_mode = active_mode
        attack_chain = getattr(self, "_attack_chain", None)
        if attack_chain is None:
            attack_chain = [self.attack] if self.attack is not None else []
        for attack_cfg in attack_chain:
            if (
                attack_cfg is not None
                and hasattr(attack_cfg, "set_mode")
                and active_mode in {"train", "test", "val"}
            ):
                attack_cfg.set_mode(active_mode)
        return active_mode

    def _compute_val_predictions(self):
        if self.model is None:
            raise ValueError(
                "Validation scoring requires a model, but model is None",
            )
        if (
            getattr(self.data, "X_val", None) is None
            or getattr(self.data, "y_val", None) is None
        ):
            raise ValueError(
                "Validation scoring requested but validation split is unavailable. "
                "Set data.val_size (or use a sampler that produces validation indices).",
            )
        if not hasattr(self.model, "predict"):
            raise ValueError("Validation scoring requires model.predict")
        val_predictions = self.model.predict(self.data.X_val)
        self.model.val_predictions = val_predictions
        return val_predictions

    def _clear_model_prediction_outputs(self) -> None:
        if self.model is None:
            return
        for attr in (
            "training_predictions",
            "predictions",
            "val_predictions",
            "training_probabilities",
            "probabilities",
            "val_probabilities",
        ):
            if not hasattr(self.model, attr):
                continue
            setattr(self.model, attr, None)

    def _ensure_mode_predictions(self, mode: str):
        if mode == "pre-sample":
            return
        if self.model is None:
            raise ValueError(
                f"{mode} scoring requires a model, but model is None",
            )
        if not hasattr(self.model, "predict"):
            raise ValueError(f"{mode} scoring requires model.predict")
        if mode == "train":
            if getattr(self.model, "training_predictions", None) is None:
                self.model.training_predictions = self.model.predict(
                    self.data.X_train,
                )
            return
        if mode == "test":
            if getattr(self.model, "predictions", None) is None:
                self.model.predictions = self.model.predict(self.data.X_test)
            return
        if mode == "val":
            self._compute_val_predictions()
            return

    def _resolve_mode_model_outputs(self, mode: str) -> tuple[Any, Any, Any]:
        """Return ``(y_true, y_pred, y_proba)`` for the requested experiment mode."""
        if mode == "train":
            X_split = self.data.X_train
            y_true = self.data.y_train
            pred_attr = "training_predictions"
            proba_attr = "training_probabilities"
        elif mode == "test":
            X_split = self.data.X_test
            y_true = self.data.y_test
            pred_attr = "predictions"
            proba_attr = "probabilities"
        elif mode == "val":
            X_split = self.data.X_val
            y_true = self.data.y_val
            pred_attr = "val_predictions"
            proba_attr = "val_probabilities"
        else:
            raise ValueError(f"Unsupported model scoring mode '{mode}'")

        y_pred = getattr(self.model, pred_attr, None)
        if y_pred is None and hasattr(self.model, "predict"):
            y_pred = self.model.predict(X_split)
            setattr(self.model, pred_attr, y_pred)

        y_proba = getattr(self.model, proba_attr, None)
        if y_proba is None and getattr(self.model, "classifier", False):
            predict_proba = getattr(self.model, "predict_proba", None)
            if callable(predict_proba):
                try:
                    y_proba = predict_proba(X_split)
                    setattr(self.model, proba_attr, y_proba)
                except Exception:
                    y_proba = None

        return y_true, y_pred, y_proba

    @staticmethod
    def _merge_reserved_runtime_kwargs(
        *payloads: dict[str, Any],
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for payload in payloads:
            if not payload:
                continue
            for key, value in payload.items():
                if key in merged:
                    raise ValueError(f"Reserved runtime key collision: {key}")
                merged[key] = value
        return merged

    def _build_reserved_runtime_kwargs(
        self,
        mode: str,
        y_true: Any,
        y_pred: Any,
        y_proba: Any = None,
    ) -> dict[str, Any]:
        data = self.data
        model = self.model
        defense = getattr(model, "defense", None)
        trainer = getattr(model, "trainer", None)
        pipeline = getattr(data, "pipeline", None)
        sampler = getattr(data, "sampler", None)
        sensitive = getattr(data, "_sensitive_test", None)

        return {
            "__deckard__labels__": y_true,
            f"__deckard__labels__{mode}__": y_true,
            "__deckard__predictions__": y_pred,
            f"__deckard__predictions__{mode}__": y_pred,
            "__deckard__probabilities__": y_proba,
            f"__deckard__probabilities__{mode}__": y_proba,
            "__deckard__mode__": mode,
            f"__deckard__mode__{mode}__": mode,
            "__deckard__data__": data,
            f"__deckard__data__{mode}__": data,
            "__deckard__model__": model,
            f"__deckard__model__{mode}__": model,
            "__deckard__attack__": self.attack,
            "__deckard__detector__": self.detector,
            "__deckard__experiment__": self,
            "__deckard__files__": self.files,
            "__deckard__score__": self.score,
            "__deckard__scorer__": self.score,
            "__deckard__defense__": defense,
            f"__deckard__defense__{mode}__": defense,
            "__deckard__trainer__": trainer,
            f"__deckard__trainer__{mode}__": trainer,
            "__deckard__pipeline__": pipeline,
            f"__deckard__pipeline__{mode}__": pipeline,
            "__deckard__sampler__": sampler,
            f"__deckard__sampler__{mode}__": sampler,
            "__deckard__sensitive__": sensitive,
            f"__deckard__sensitive__{mode}__": sensitive,
        }

    def _run_experiment_scorer_modes(self, score_file=None) -> dict:
        if self.score is None:
            return {}
        out = {}
        modes = self._resolve_score_modes()
        should_nest_by_mode = (
            self.evaluation_mode == "report" and not self._has_explicit_score_mode()
        ) or (self._has_explicit_score_mode() and len(modes) > 1)
        scorer_is_data_profile = isinstance(self.score, _DataScorerMarker)
        for mode in modes:
            common_kwargs = {
                "data": self.data,
                "model": self.model,
                "attack": self.attack,
                "detector": self.detector,
                "experiment": self,
                "mode": mode,
                "score_file": score_file,
            }
            if scorer_is_data_profile:
                y_true, y_pred = self._resolve_data_mode_inputs(mode)
                runtime_kwargs = self._build_reserved_runtime_kwargs(
                    mode,
                    y_true,
                    y_pred,
                )
                mode_scores = self.score(
                    **common_kwargs,
                    dep=y_true,
                    ind=y_pred,
                    **runtime_kwargs,
                )
            else:
                if mode == "pre-sample":
                    raise ValueError(
                        "pre-sample mode is only supported for data-profile experiment scorers.",
                    )
                self._ensure_mode_predictions(mode)
                y_true, y_pred, y_proba = self._resolve_mode_model_outputs(mode)
                runtime_kwargs = self._build_reserved_runtime_kwargs(
                    mode,
                    y_true,
                    y_pred,
                    y_proba,
                )
                mode_scores = self.score(
                    **common_kwargs,
                    dep=y_true,
                    ind=y_pred,
                    **runtime_kwargs,
                )
            if (
                isinstance(mode_scores, dict)
                and mode in mode_scores
                and isinstance(
                    mode_scores[mode],
                    dict,
                )
            ):
                mode_scores = mode_scores[mode]
            if not isinstance(mode_scores, dict):
                raise TypeError(
                    f"Experiment scorer for mode '{mode}' must return a dictionary, got {type(mode_scores)}",
                )
            if mode == "pre-sample":
                out["pre-sample"] = mode_scores
            elif should_nest_by_mode:
                out[mode] = mode_scores
            else:
                out.update(mode_scores)
        return out

    def _apply_attack_profile_scorer(self, scorer) -> None:
        """Apply an _AttackProfileScorer to the configured attack chain."""
        attack_chain = getattr(self, "_attack_chain", [])
        profile_attr = getattr(scorer, "_profile_attr", "evasion")
        for attack_cfg in attack_chain:
            if hasattr(attack_cfg, "scorer") and attack_cfg.scorer is not None:
                setattr(attack_cfg.scorer, profile_attr, scorer)

    def _route_scorer_to_scope(self, scope: str, scorer) -> None:
        """Attach *scorer* to the component identified by *scope*.

        Extend this method (or add a new ``elif`` branch) to support additional
        components (e.g. ``"detector"`` once detector scoring is formalised).
        """
        if scorer is None:
            return
        if scope == "data":
            self.data.scorer = scorer
        elif scope == "model":
            if self.model is not None:
                self.model.scorer = scorer
        elif scope == "attack":
            self._apply_attack_profile_scorer(scorer)
        elif scope == "detector":
            if self.detector is not None and hasattr(self.detector, "scorer"):
                self.detector.scorer = scorer
        elif scope == "experiment":
            self.score = scorer

    def _default_scorer_factory_for_scope(self, scope: str):
        if scope == "data":
            return lambda: load_class(
                (
                    "deckard.score.data.DefaultDataClassificationScorerDictConfig"
                    if bool(getattr(self.data, "classifier", True))
                    else "deckard.score.data.DefaultDataRegressionScorerDictConfig"
                ),
            )
        if scope == "model":
            return lambda: load_class(
                (
                    "deckard.score.base.DefaultClassifierScorerDictConfig"
                    if bool(getattr(self.model, "classifier", True))
                    else "deckard.score.base.DefaultRegressorScorerDictConfig"
                ),
            )
        return None

    def _merge_scope_scorers(self, scope: str, incoming_scorers: list):
        if scope not in {"data", "model"}:
            raise ValueError(f"Unsupported scope for scorer merge: {scope}")

        current = getattr(self.data if scope == "data" else self.model, "scorer", None)
        default_factory = self._default_scorer_factory_for_scope(scope)
        current = coerce_scorer_config(current, default_factory=default_factory)

        chain = []
        if current is not None:
            chain.append(current)
        chain.extend([sc for sc in incoming_scorers if sc is not None])

        if not chain:
            return None

        fairness_base = None
        for candidate in chain:
            if hasattr(candidate, "group_scorers"):
                fairness_base = candidate

        merged = ScorerDictConfig.merge(chain)
        if fairness_base is None:
            return merged

        from ..plugins.fairlearn.score import FairlearnScorerDictConfig

        return FairlearnScorerDictConfig(
            scorers=merged.scorers,
            group_scorers=dict(getattr(fairness_base, "group_scorers", {}) or {}),
            group_reduction=getattr(
                fairness_base,
                "group_reduction",
                "difference",
            ),
            group_reduction_method=getattr(
                fairness_base,
                "group_reduction_method",
                "between_groups",
            ),
            include_group_overall=bool(
                getattr(fairness_base, "include_group_overall", False),
            ),
            include_group_by_group=bool(
                getattr(fairness_base, "include_group_by_group", True),
            ),
        )

    @staticmethod
    def _is_anjana_scorer_spec(spec: Any) -> bool:
        if isinstance(spec, dict):
            score_fn = spec.get("score_function")
        else:
            score_fn = getattr(spec, "score_function", None)
        return (
            isinstance(score_fn, str) and "deckard.plugins.anjana.score." in score_fn
        )

    def _split_merged_score_profiles(
        self,
        plain: dict,
    ) -> tuple[dict | None, dict | None]:
        scorers = plain.get("scorers")
        if not isinstance(scorers, dict):
            return None, plain

        data_scorers = {
            key: value
            for key, value in scorers.items()
            if self._is_anjana_scorer_spec(value)
        }
        if not data_scorers:
            return None, plain

        remaining_scorers = {
            key: value for key, value in scorers.items() if key not in data_scorers
        }

        data_cfg = {
            "_target_": "deckard.plugins.anjana.score.DefaultAnjanaScorerDictConfig",
            "scorers": data_scorers,
        }

        model_cfg = dict(plain)
        if remaining_scorers:
            model_cfg["scorers"] = remaining_scorers
        else:
            model_cfg.pop("scorers", None)

        return data_cfg, model_cfg

    def _initialize_component_scorers(self) -> None:
        """Route the experiment-level ``score`` config to data/model/attack components.

        Routing rules (applied after Hydra config resolution):

        **Scoped dict** (Hydra ``@`` package syntax)::

            +score@score.data=data-classification
            +score@score.model=classification
            +score@score.attack=evasion-classification

        Produces ``score: {data: {...}, model: {...}, attack: {...}}``.
        Each sub-key is routed directly to its component via
        :meth:`_route_scorer_to_scope` without any type-inference.

        **Single config** (type-based fallback)::

            score=classification              # -> model.scorer
            score=data-classification        # -> data.scorer (_DataScorerMarker)
            score=evasion-classification     # -> attack scorer (_AttackProfileScorer)

        **Null / auto / default** -> components self-configure from their own defaults.
        """
        score_cfg = self.score

        # Null / auto -> let each component self-configure.
        if (
            score_cfg is None
            or is_null_config_value(score_cfg)
            or is_default_config_value(score_cfg)
        ):
            self.score = None
            return

        plain = (
            OmegaConf.to_container(score_cfg, resolve=True)
            if isinstance(score_cfg, DictConfig)
            else (dict(score_cfg) if isinstance(score_cfg, dict) else None)
        )

        if isinstance(plain, dict):
            # Auto-configure sentinel (from score/auto.yaml).
            if plain.get("_auto_configure"):
                self.score = None
                return

            split_data_cfg, split_model_cfg = self._split_merged_score_profiles(plain)
            if split_data_cfg is not None:
                data_scorer = coerce_scorer_config(split_data_cfg)
                if data_scorer is not None:
                    self.data.scorer = self._merge_scope_scorers("data", [data_scorer])

                model_scorer = coerce_scorer_config(split_model_cfg)
                if model_scorer is not None and self.model is not None:
                    self.model.scorer = self._merge_scope_scorers(
                        "model",
                        [model_scorer],
                    )

                self.score = None
                return

            # Scoped dict: keys are component names produced by Hydra @ package syntax
            # or by passing {"data": scorer, "model": scorer, ...} directly.
            _SCOPE_KEYS = ("data", "model", "attack", "detector", "experiment")
            if any(k in _SCOPE_KEYS for k in plain):
                for scope in _SCOPE_KEYS:
                    if scope in plain:
                        self._route_scorer_to_scope(
                            scope,
                            coerce_scorer_config(plain[scope]),
                        )
                # experiment scope sets self.score; all others clear it
                if "experiment" not in plain:
                    self.score = None
                return

        # Normalise to a list for type-based routing (single config or comma string).
        if isinstance(score_cfg, (list, ListConfig)):
            items = list(coerce_to_list(score_cfg))
        elif isinstance(score_cfg, str) and "," in score_cfg:
            items = split_separated_tokens(score_cfg)
        else:
            items = [score_cfg]

        data_scorers: list = []
        model_scorers: list = []

        for item in items:
            scorer = coerce_scorer_config(item)
            if scorer is None:
                continue
            if isinstance(scorer, _AttackProfileScorer):
                self._apply_attack_profile_scorer(scorer)
            elif isinstance(scorer, _DataScorerMarker):
                data_scorers.append(scorer)
            else:
                model_scorers.append(scorer)

        if data_scorers:
            self.data.scorer = self._merge_scope_scorers("data", data_scorers)
        if model_scorers and self.model is not None:
            self.model.scorer = self._merge_scope_scorers("model", model_scorers)

        # Score chain fully routed; no experiment-level scorer needed.
        self.score = None

    def _coerce_single_attack(self, attack_obj: Any) -> AttackConfig:
        try:
            attack_cfg = instantiate_config(
                attack_obj,
                AttackConfig,
                default_target="deckard.attack.AttackConfig",
            )
        except TypeError as exc:
            raise ValueError(
                f"Unsupported type for attack: {type(attack_obj)}",
            ) from exc

        assert isinstance(
            attack_cfg,
            AttackConfig,
        ), "attack must be an instance of AttackConfig"
        return attack_cfg

    def _normalize_attack_chain(self, attack_obj: Any) -> list[AttackConfig]:
        if attack_obj is None:
            return []

        if isinstance(attack_obj, (list, tuple, ListConfig)):
            raw_items = coerce_to_list(attack_obj)
        else:
            coerced = coerce_config(attack_obj)
            if isinstance(coerced, (list, tuple, ListConfig)):
                raw_items = coerce_to_list(coerced)
            else:
                raw_items = [attack_obj]

        return [self._coerce_single_attack(item) for item in raw_items]

    def _validate_multi_attack_aliases(self, attack_chain: list[AttackConfig]) -> None:
        if len(attack_chain) <= 1:
            return

        seen_aliases: set[str] = set()
        for attack_cfg in attack_chain:
            alias = str(getattr(attack_cfg, "alias", "")).strip()
            if alias == "":
                raise ValueError(
                    "Multi-attack experiments require attack.alias for each configured attack.",
                )
            if alias in seen_aliases:
                raise ValueError(
                    f"Duplicate attack.alias detected in multi-attack experiment: '{alias}'.",
                )
            seen_aliases.add(alias)

    @staticmethod
    def _suffix_file_path_with_alias(file_path: str, alias: str) -> str:
        path = Path(file_path)
        return path.with_name(f"{path.stem}_{alias}{path.suffix}").as_posix()

    def _build_attack_file_outputs_for_run(
        self,
        base_outputs: dict,
        attack_cfg: AttackConfig,
        *,
        multi_attack: bool,
    ) -> dict:
        outputs = dict(base_outputs)
        if not multi_attack:
            return outputs

        alias = str(attack_cfg.alias).strip()
        for key in ("attack_file", "attack_predictions_file"):
            path = outputs.get(key)
            if path is None or str(path).strip() == "":
                continue
            outputs[key] = self._suffix_file_path_with_alias(str(path), alias)

        # Avoid per-attack score-file clobbering; ExperimentConfig writes the final merged score file.
        outputs["score_file"] = None
        return outputs

    @staticmethod
    def _combine_attack_predictions(attack_chain: list[AttackConfig]):
        predictions = []
        for attack_cfg in attack_chain:
            value = getattr(attack_cfg, "attack_predictions", None)
            if value is None:
                continue
            if hasattr(value, "detach") and hasattr(value, "cpu"):
                value = value.detach().cpu().numpy()
            predictions.append(value)

        if len(predictions) == 0:
            return None

        try:
            if all(isinstance(value, pd.DataFrame) for value in predictions):
                return pd.concat(predictions, axis=0, ignore_index=True)
        except Exception:
            pass

        try:
            return np.concatenate([np.asarray(value) for value in predictions], axis=0)
        except Exception:
            flattened = []
            for value in predictions:
                arr = np.asarray(value)
                if arr.ndim == 0:
                    flattened.append(arr.item())
                else:
                    flattened.extend(list(arr))
            return np.asarray(flattened)

    def _build_detector_attack_view(self, attack_chain: list[AttackConfig]):
        if len(attack_chain) == 1:
            return attack_chain[0]

        combined_predictions = self._combine_attack_predictions(attack_chain)
        if combined_predictions is None:
            raise ValueError(
                "Detector phase requires at least one attack prediction in multi-attack mode.",
            )
        return SimpleNamespace(
            attack_predictions=combined_predictions,
            attacks=attack_chain,
        )

    def set_device(self, device: Union[str, int] = "cpu") -> None:
        """
        Set the computation device for the experiment based on the selected library.
        For TensorFlow, configures GPU/CPU usage.
        Args:
            device (Union[str, int]): Device to use ("cpu", "gpu", or GPU index).

        Raises:
            ImportError: If tensorflow support is requested but unavailable.
        """

        if self.library == "tensorflow":
            if tf is None:
                raise ImportError(
                    "TensorFlow support is unavailable. Install tensorflow to use library='tensorflow'.",
                )

            gpus = tf.config.list_physical_devices("GPU")
            if device == "cpu":
                tf.config.set_visible_devices([], "GPU")
                logger.info("Using CPU for TensorFlow")
            elif isinstance(device, str) and "gpu" in device.lower():
                if gpus:
                    try:
                        tf.config.set_visible_devices(gpus[0], "GPU")
                        tf.config.experimental.set_memory_growth(gpus[0], True)
                        logger.info(f"Using GPU for TensorFlow: {gpus[0]}")
                    except RuntimeError as e:
                        logger.error(e)
                else:
                    logger.warning("No GPU found, using CPU for TensorFlow")
            elif isinstance(device, int) and gpus and device < len(gpus):
                try:
                    tf.config.set_visible_devices(gpus[device], "GPU")
                    tf.config.experimental.set_memory_growth(gpus[device], True)
                    logger.info(f"Using GPU for TensorFlow: {gpus[device]}")
                except RuntimeError as e:
                    logger.error(e)
            else:
                logger.warning(
                    "Invalid device specified for TensorFlow, using default device.",
                )
        else:
            logger.info(
                "Device selection not supported for library: %s",
                self.library,
            )

    def _initialize_data_and_classifier(self) -> None:
        """Resolve data config and enforce experiment/data classifier consistency."""
        self.data = self._resolve_data_config()
        assert isinstance(
            self.data,
            DataConfig,
        ), f"data must be an instance of DataConfig. Got {type(self.data)}"
        if self.classifier in ["classifier", True]:
            self.classifier = True
        elif self.classifier in ["regressor", False]:
            self.classifier = False
        else:
            raise ValueError(
                f"classifier in experiment must be boolean or one of ['classifier', 'regressor'], got {self.classifier}",
            )
        assert (
            self.classifier == self.data.classifier
        ), f"classifier in experiment must match data.classifier. Got {self.classifier} vs {self.data.classifier}"

    def _initialize_defense(self) -> None:
        """Normalize defense config when configured."""
        if self.defense is not None:
            self.defense = DefensePipelineConfig.coerce(self.defense)
            assert isinstance(
                self.defense,
                DefensePipelineConfig,
            ), "defense must be an instance of DefensePipelineConfig"

    def _coerce_model(self) -> None:
        """Normalize model config and enforce model/classifier consistency."""
        if self.model is None:
            return

        try:
            self.model = self.coerce_component(
                self.model,
                ModelConfig,
                default_target="deckard.model.ModelConfig",
                overrides=(
                    {"defense": self.defense} if self.defense is not None else None
                ),
            )
        except TypeError as exc:
            raise ValueError(
                f"Unsupported type for model: {type(self.model)}",
            ) from exc

        if self.defense is not None:
            self.model.defense = self.defense
        assert isinstance(
            self.model,
            ModelConfig,
        ), "model must be an instance of ModelConfig"
        assert (
            self.classifier == self.model.classifier
        ), f"classifier in experiment must match model.classifier. Got {self.classifier} vs {self.model.classifier}"

    def _specialize_model_for_data(self) -> None:
        """Swap/attach model subtype for fairness data configs."""
        if self.model is None:
            return

        if FairlearnDataConfig is not None and isinstance(
            self.data,
            FairlearnDataConfig,
        ):
            (
                resolved_pytorch_model_cls,
                resolved_fairlearn_model_cls,
                resolved_fairlearn_pytorch_model_cls,
            ) = _load_optional_model_specializations()
            is_torch_model = resolved_pytorch_model_cls is not None and isinstance(
                self.model,
                resolved_pytorch_model_cls,
            )
            if is_torch_model:
                if resolved_fairlearn_pytorch_model_cls is None:
                    # Some environments can instantiate the plugin class directly
                    # while this module-level optional import remains unavailable.
                    if self.model.__class__.__name__ == "FairlearnPytorchModelConfig":
                        self.model.data = self.data
                        return
                    raise ImportError(
                        "FairlearnPytorchModelConfig requires optional fairness and torch dependencies. "
                        "Install deckard[fairlearn,torch] to enable fairness-aware pytorch model configs.",
                    )
                target_model_cls = resolved_fairlearn_pytorch_model_cls
            else:
                if resolved_fairlearn_model_cls is None:
                    raise ImportError(
                        "FairlearnModelConfig requires optional fairness dependencies. Install deckard[fairlearn] to enable fairlearn model configs.",
                    )
                target_model_cls = resolved_fairlearn_model_cls

            if target_model_cls is None:
                raise ImportError(
                    "Fairlearn model specialization dependencies are unavailable. "
                    "Install deckard[fairlearn] (and torch extras for pytorch flows).",
                )

            fairness_types = tuple(
                cfg
                for cfg in (
                    resolved_fairlearn_model_cls,
                    resolved_fairlearn_pytorch_model_cls,
                )
                if cfg is not None
            )

            if not isinstance(self.model, fairness_types):
                model_name = self.model.resolve_name(default=None)
                if model_name is None:
                    raise ValueError(
                        "ModelConfig.name must be set for fairness specialization",
                    )
                self.model = target_model_cls(
                    name=model_name,
                    classifier=self.model.classifier,
                    model_params=self.model.model_params,
                    probability=self.model.probability,
                    alias=self.model.alias,
                    defense=self.model.defense,
                    plugins=self.model.plugins,
                    scorer=self.model.scorer,
                    data=self.data,
                )
            else:
                self.model.data = self.data
            return

    def _initialize_attack_chain(self) -> None:
        """Normalize configured attacks and establish primary attack view."""
        self._attack_chain = self._normalize_attack_chain(self.attack)
        self._validate_multi_attack_aliases(self._attack_chain)
        if len(self._attack_chain) > 0:
            self.attack = self._attack_chain[0]
        else:
            self.attack = None

    def _initialize_detector(self) -> None:
        """Normalize detector config while preserving callable detector passthrough."""
        if self.detector is None:
            return

        try:
            self.detector = self.coerce_component(
                self.detector,
                DetectorConfig,
                default_target="deckard.detector.DetectorConfig",
                allow_passthrough=lambda obj: callable(getattr(obj, "__call__", None)),
            )
        except TypeError as exc:
            raise ValueError(
                f"Unsupported type for detector: {type(self.detector)}",
            ) from exc
        assert isinstance(
            self.detector,
            DetectorConfig,
        ) or callable(
            getattr(self.detector, "__call__", None),
        ), "detector must be a DetectorConfig or callable detector runtime"

    def _initialize_files(self) -> None:
        """Normalize file config and ensure a FileConfig instance is available."""
        if self.files is None:
            self.files = FileConfig()
        else:
            try:
                self.files = self.coerce_component(
                    self.files,
                    FileConfig,
                    default_target="deckard.file.FileConfig",
                )
            except TypeError as exc:
                raise ValueError(
                    f"Unsupported type for files: {type(self.files)}",
                ) from exc
        assert isinstance(
            self.files,
            FileConfig,
        ), "file must be an instance of FileConfig"

    def compose_components(self, **kwargs: Any) -> "ExperimentConfig":
        """Apply native component overrides and re-compose experiment runtime state.

        Supported keys include data/model/attack/detector/score/files/defense and
        hook-oriented keys (hook_plugins/hook_bundles).

        Args:
            **kwargs: Component overrides and runtime orchestration options.

        Returns:
            This ExperimentConfig instance.

        Raises:
            ValueError: If unsupported override keys are provided.
        """
        supported = {
            "data",
            "model",
            "attack",
            "detector",
            "score",
            "files",
            "defense",
            "hook_plugins",
            "hook_bundles",
            "evaluation_mode",
            "score_mode",
            "cache_enabled",
        }
        unknown = sorted(set(kwargs) - supported)
        if unknown:
            raise ValueError(
                f"Unsupported compose_components keys: {unknown}. Supported keys: {sorted(supported)}",
            )

        for key, value in kwargs.items():
            setattr(self, key, value)

        ensure_experiment_runtime_contract(self)
        self._validate_specialization_pre_init()
        self._initialize_data_and_classifier()
        self._validate_mode_configuration()
        self._initialize_defense()
        self._coerce_model()
        self._specialize_model_for_data()
        self._initialize_attack_chain()
        self._initialize_detector()
        self._initialize_files()
        self._initialize_component_scorers()
        self._validate_scorer_scope_configuration()
        self._initialize_hook_orchestration()
        self._finalize_specialization()
        self.params = build_experiment_params_manifest(self)
        self._runtime_cache = self._load_runtime_cache()
        return self

    def __post_init__(self) -> None:
        ensure_experiment_runtime_contract(self)
        if not hasattr(self, "_target_") or self._target_ is None:
            self._target_ = "deckard.experiment.ExperimentConfig"
        self._validate_specialization_pre_init()
        # Set random seed
        self.set_random_seed()
        self._initialize_data_and_classifier()
        self._validate_mode_configuration()
        self._initialize_defense()
        self._coerce_model()
        self._specialize_model_for_data()
        self._initialize_attack_chain()
        self._initialize_detector()
        # Set experiment name if not provided
        if self.experiment_name in [None, "", "{hash}", "*"]:
            config_list = [self.data]
            if self.model:
                config_list.append(self.model)
            if len(self._attack_chain) > 0:
                config_list.extend(self._attack_chain)
            if self.detector and isinstance(self.detector, BaseConfig):
                config_list.append(self.detector)
            if self.score:
                config_list.append(self.score)
            self.experiment_name = self._hash_from_list(config_list)
        self._initialize_files()
        self._initialize_component_scorers()
        self._validate_scorer_scope_configuration()

        # Reconcile and enforce a single device across experiment/data/model.
        self._reconcile_component_devices()
        if self.library not in ["sklearn"]:
            self.set_device(self.device if self.device is not None else "cpu")

        self._initialize_hook_orchestration()
        self._finalize_specialization()

        # Store a serializable initialization-time manifest for persistence and caching.
        self.params = build_experiment_params_manifest(self)
        self._runtime_cache = self._load_runtime_cache()

    def _validate_specialization_pre_init(self) -> None:
        """Allow subclasses to fail fast before base composition runs."""
        return

    def _finalize_specialization(self) -> None:
        """Allow subclasses to apply backend-specific checks after base composition."""
        return

    def _validate_scorer_scope_configuration(self) -> None:
        """Validate scorer scope and requested score modes at initialization time."""
        return

    def _initialize_hook_orchestration(self) -> None:
        """Compose canonical and user-provided hook plugins/bundles."""
        from .dvc import build_dvc_experiment_plugin_hooks
        from .repro import build_repro_experiment_plugin_hooks
        from .power import build_power_plugin_hooks

        canonical_bundle = build_experiment_hook_bundle()
        dvc_first_hooks, dvc_last_hooks = build_dvc_experiment_plugin_hooks(
            self.dvc_plugin,
        )
        repro_first_hooks, repro_last_hooks = build_repro_experiment_plugin_hooks(
            self.dvc_plugin,
        )
        power_first_hooks, power_last_hooks = build_power_plugin_hooks(
            enabled=callable(getattr(self, "_log_power_score", None)),
        )
        user_bundles: list[HookBundle] = []
        for bundle in coerce_to_list(self.hook_bundles):
            if isinstance(bundle, HookBundle):
                user_bundles.append(bundle)
            elif isinstance(bundle, dict) and "name" in bundle and "hooks" in bundle:
                hooks = tuple(bundle.get("hooks") or ())
                if all(hasattr(hook, "hook_name") for hook in hooks):
                    user_bundles.append(
                        HookBundle(name=str(bundle["name"]), hooks=hooks),
                    )

        self.outputs.setdefault("hooks", {})
        self.outputs["hooks"]["graph"] = build_experiment_hook_graph()
        self._composed_hook_plugins = compose_hook_plugins(
            dvc_first_hooks,
            repro_first_hooks,
            power_first_hooks,
            canonical_bundle,
            user_bundles,
            self.hook_plugins,
            power_last_hooks,
            repro_last_hooks,
            dvc_last_hooks,
        )

    def _run_experiment_stage_hooks(
        self,
        when: str,
        stage: str,
        *,
        component: str,
        **kwargs: Any,
    ) -> list[Any]:
        event = str(when).strip().lower()
        if event not in {"before", "after"}:
            raise ValueError(f"Hook event must be 'before' or 'after', got {when}")
        hook_name = f"{event}_{str(stage).strip().lower().replace('-', '_')}"
        outputs: list[Any] = []
        for plugin in getattr(self, "_composed_hook_plugins", []):
            hook = getattr(plugin, hook_name, None)
            if callable(hook):
                hook_output = hook(
                    self,
                    component=component,
                    stage=stage,
                    event=event,
                    **kwargs,
                )
                outputs.append(hook_output)
                if (
                    event == "after"
                    and isinstance(hook_output, dict)
                    and str(stage).strip().lower().replace("-", "_")
                    in {"data_score", "model_score", "attack_score", "detector_score"}
                ):
                    self._merge_stage_hook_scores(component, hook_output)
        return outputs

    def _resolve_stage_component_target(self, component: str) -> Any:
        token = str(component).strip().lower()
        if token == "data":
            return self.data
        if token == "model":
            return self.model
        if token == "detector":
            return self.detector
        if token.startswith("attack"):
            _, _, alias = token.partition(":")
            if alias:
                for attack_cfg in getattr(self, "_attack_chain", []) or []:
                    if str(getattr(attack_cfg, "alias", "")).strip().lower() == alias:
                        return attack_cfg
            return self.attack
        return None

    def _merge_stage_hook_scores(
        self,
        component: str,
        hook_scores: Mapping[str, Any],
    ) -> None:
        target = self._resolve_stage_component_target(component)
        if target is not None:
            target_scores = getattr(target, "score_dict", None)
            if not isinstance(target_scores, dict):
                target_scores = {}
                setattr(target, "score_dict", target_scores)
            target_scores.update(dict(hook_scores))
        experiment_scores = getattr(self, "score_dict", None)
        if isinstance(experiment_scores, dict):
            experiment_scores.update(dict(hook_scores))

    def _experiment_stage_hook(
        self,
        *,
        component: str,
        stage: str,
        event: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Default experiment hook callback used by canonical HookPlugins."""
        hook_bucket = self.outputs.setdefault("hooks", {})
        trace = hook_bucket.setdefault("trace", [])
        trace.append(
            {
                "component": component,
                "stage": stage,
                "event": event,
                "run": kwargs.get("run_idx"),
            },
        )
        return {
            "component": component,
            "stage": stage,
            "event": event,
        }

    def _dvc_experiment_plugin_hook(
        self,
        *,
        dvc_plugin: Any,
        plugin_position: str,
        component: str,
        stage: str,
        event: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Delegate optional DVCExperimentPlugin hook handling to dvc helpers."""
        from .dvc import run_dvc_experiment_plugin_hook

        return run_dvc_experiment_plugin_hook(
            self,
            dvc_plugin=dvc_plugin,
            plugin_position=plugin_position,
            component=component,
            stage=stage,
            event=event,
            **kwargs,
        )

    def _repro_experiment_plugin_hook(
        self,
        *,
        repro_plugin: Any,
        plugin_position: str,
        component: str,
        stage: str,
        event: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Delegate optional DVC persistence hook handling to repro helpers."""
        from .repro import run_repro_experiment_plugin_hook

        return run_repro_experiment_plugin_hook(
            self,
            repro_plugin=repro_plugin,
            plugin_position=plugin_position,
            component=component,
            stage=stage,
            event=event,
            **kwargs,
        )

    def _power_experiment_plugin_hook(
        self,
        *,
        power_plugin: Any,
        namespace: str,
        component: str,
        stage: str,
        event: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Delegate optional power metrics hook handling to power helpers."""
        from .power import run_power_experiment_plugin_hook

        return run_power_experiment_plugin_hook(
            self,
            power_plugin=power_plugin,
            namespace=namespace,
            component=component,
            stage=stage,
            event=event,
            **kwargs,
        )

    def _cache_file_path(self) -> str | None:
        if not self.cache_enabled:
            return None
        params_file = getattr(self.files, "params_file", None)
        if params_file in [None, ""]:
            return None
        params_path = Path(str(params_file))
        if params_path.suffix.lower() in {".yaml", ".yml"}:
            return params_path.with_name(
                f"{params_path.stem}.runtime_cache.pkl",
            ).as_posix()
        return params_path.with_suffix(".runtime_cache.pkl").as_posix()

    @staticmethod
    def _extract_schema_major(version: str) -> int:
        token = str(version).strip()
        if not token.startswith(CANONICAL_EXPERIMENT_RUNTIME_SCHEMA_PREFIX):
            raise ValueError(
                "Unsupported experiment runtime schema version "
                f"'{version}'. Expected prefix '{CANONICAL_EXPERIMENT_RUNTIME_SCHEMA_PREFIX}'.",
            )
        suffix = token[len(CANONICAL_EXPERIMENT_RUNTIME_SCHEMA_PREFIX) :]
        if suffix == "" or not suffix.isdigit():
            raise ValueError(
                f"Malformed experiment runtime schema version '{version}'.",
            )
        return int(suffix)

    def _runtime_state_file_path(self, file_dict: Mapping[str, Any]) -> str | None:
        params_file = file_dict.get("params_file")
        if params_file in [None, ""]:
            return None
        return str(self._resolve_yaml_write_path(str(params_file)))

    def _build_runtime_state_payload(
        self,
        file_dict: Mapping[str, Any],
    ) -> dict[str, Any]:
        hook_outputs = dict(self.outputs.get("hooks", {}) or {})
        cache_outputs = dict(self.outputs.get("cache", {}) or {})
        cache_outputs.setdefault("path", self._cache_file_path())
        cache_outputs["enabled"] = bool(self.cache_enabled)
        cache_outputs["hits_count"] = len(cache_outputs.get("hits", []) or [])
        cache_outputs["writes_count"] = len(cache_outputs.get("writes", []) or [])

        experiment_payload = self._sanitize_runtime_instantiation_payload(
            self.to_dict(for_hash=True),
        )
        files_payload = experiment_payload.get("files")
        if isinstance(files_payload, dict):
            files_payload.pop("handler", None)

        return {
            "schema_version": self.RUNTIME_STATE_VERSION,
            "experiment": experiment_payload,
            "params": build_experiment_params_manifest(self),
            "runtime": {
                "times": dict(self.times or {}),
                "outputs": {
                    "files": dict(file_dict),
                    "hooks": hook_outputs,
                    "cache": cache_outputs,
                    "scores": dict(self.score_dict or {}),
                },
            },
        }

    def _persist_runtime_state(self, file_dict: Mapping[str, Any]) -> None:
        runtime_state_path = self._runtime_state_file_path(file_dict)
        if runtime_state_path is None:
            return
        payload = self._build_runtime_state_payload(file_dict)
        path = Path(runtime_state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        self.outputs.setdefault("files", {})["params_file"] = path.as_posix()

    @staticmethod
    def _sanitize_runtime_instantiation_payload(payload: Any) -> Any:
        if isinstance(payload, list):
            return [
                ExperimentConfig._sanitize_runtime_instantiation_payload(item)
                for item in payload
            ]

        if not isinstance(payload, dict):
            return payload

        sanitized = {
            key: ExperimentConfig._sanitize_runtime_instantiation_payload(value)
            for key, value in payload.items()
        }
        target = sanitized.get("_target_")
        if not isinstance(target, str):
            return sanitized

        try:
            target_cls = load_class(target)
            init_sig = inspect.signature(target_cls.__init__)
        except Exception:
            return sanitized

        params = init_sig.parameters
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in params.values()
        )
        if accepts_var_kwargs:
            return sanitized

        allowed = {
            name
            for name, parameter in params.items()
            if name != "self"
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        filtered = {"_target_": target}
        for key, value in sanitized.items():
            if key == "_target_" or key in allowed:
                filtered[key] = value
        return filtered

    def _load_runtime_cache(self) -> dict[str, Any]:
        if not self.cache_enabled:
            return {}
        cache_path = self._cache_file_path()
        if cache_path is None or not Path(cache_path).exists():
            return {}
        payload = self.load_object(
            cache_path,
            ignore_corrupt=True,
            delete_corrupt=True,
        )
        if isinstance(payload, dict):
            return payload
        return {}

    def _persist_runtime_cache(self) -> None:
        if not self.cache_enabled:
            return
        cache_path = self._cache_file_path()
        if cache_path is None:
            return
        self.save_object(getattr(self, "_runtime_cache", {}), cache_path)

    def _build_stage_cache_key(
        self,
        *,
        stage: str,
        component: str,
        identity: Mapping[str, Any] | None = None,
    ) -> str:
        params_manifest = getattr(self, "params", None)
        if not isinstance(params_manifest, Mapping):
            params_manifest = {}
        return build_experiment_stage_cache_key(
            params_manifest=params_manifest,
            stage=stage,
            component=component,
            identity=identity,
        )

    def _cache_stage_get(
        self,
        *,
        stage: str,
        component: str,
        identity: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.cache_enabled or stage not in CANONICAL_EXPERIMENT_CACHE_STAGES:
            return None
        cache_key = self._build_stage_cache_key(
            stage=stage,
            component=component,
            identity=identity,
        )
        cache_store = getattr(self, "_runtime_cache", {})
        stage_bucket = cache_store.get(stage, {})
        cached = stage_bucket.get(cache_key)
        if isinstance(cached, dict):
            outputs = getattr(self, "outputs", None)
            if not isinstance(outputs, dict):
                outputs = {}
                self.outputs = outputs
            outputs.setdefault("cache", {}).setdefault("hits", []).append(
                {"stage": stage, "component": component, "key": cache_key},
            )
            return cached
        return None

    def _cache_stage_set(
        self,
        *,
        stage: str,
        component: str,
        value: Mapping[str, Any],
        identity: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.cache_enabled or stage not in CANONICAL_EXPERIMENT_CACHE_STAGES:
            return
        cache_key = self._build_stage_cache_key(
            stage=stage,
            component=component,
            identity=identity,
        )
        cache_store = getattr(self, "_runtime_cache", {})
        stage_bucket = cache_store.setdefault(stage, {})
        stage_bucket[cache_key] = dict(value)
        self._runtime_cache = cache_store
        outputs = getattr(self, "outputs", None)
        if not isinstance(outputs, dict):
            outputs = {}
            self.outputs = outputs
        outputs.setdefault("cache", {}).setdefault("writes", []).append(
            {"stage": stage, "component": component, "key": cache_key},
        )

    @staticmethod
    def _sample_cache_fields() -> tuple[str, ...]:
        return (
            "train_indices",
            "test_indices",
            "val_indices",
            "X_train",
            "y_train",
            "X_test",
            "y_test",
            "X_val",
            "y_val",
            "train_n",
            "test_n",
            "val_n",
            "pipeline_fit_n",
            "pipeline_transform_n",
            "pipeline_fit_time",
            "pipeline_transform_time",
            "pipeline_y_fit_n",
            "pipeline_y_transform_n",
            "pipeline_y_fit_time",
            "pipeline_y_transform_time",
            "data_load_time",
            "data_sample_time",
        )

    def _capture_sample_cache_value(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "score_dict": dict(getattr(self.data, "score_dict", {}) or {}),
        }
        for attr in self._sample_cache_fields():
            value[attr] = getattr(self.data, attr, None)
        return value

    def _apply_sample_cache_value(self, value: Mapping[str, Any]) -> None:
        for attr in self._sample_cache_fields():
            if attr in value:
                setattr(self.data, attr, value.get(attr))
        self.data.score_dict = dict(value.get("score_dict", {}) or {})

    def set_random_seed(self) -> None:
        """Set deterministic random seed for the configured runtime library.

        Raises:
            ImportError: If selected library dependency is unavailable.
            ValueError: If runtime library is unsupported.
        """
        if self.library in ["sklearn"]:
            np.random.seed(self.random_state)
        elif self.library in ["tensorflow"]:
            if tf is None:
                raise ImportError(
                    "TensorFlow support is unavailable. Install tensorflow to use library='tensorflow'.",
                )

            tf.random.set_seed(self.random_state)
        elif self.library in ["pytorch"]:
            if torch is None:
                raise ImportError(
                    "PyTorch support is unavailable. Install torch to use library='pytorch'.",
                )

            torch.manual_seed(self.random_state)
        else:
            raise ValueError(f"Unsupported library: {self.library}")

    def _hash_from_list(self, config_list: List[Any]) -> str:
        """
        Generate a hash string from a list of BaseConfig objects.
        The hash is generated by concatenating the string representations of the configurations
        and computing the MD5 hash of the resulting string.
        Args:
            config_list (List[BaseConfig]): List of BaseConfig objects to generate the hash from.
        Returns:
            str: The generated hash string.
        """
        hash_parts = []
        for conf in config_list:
            normalized = coerce_config(conf)
            if (
                isinstance(normalized, (dict, list, tuple, str, int, float, bool))
                or normalized is None
            ):
                hash_parts.append(str(normalized))
                continue

            assert isinstance(
                conf,
                BaseConfig,
            ), "All items in config_list must be BaseConfig or config-like values"
            hash_parts.append(
                "".join(
                    [
                        str(getattr(conf, attr))
                        for attr in dir(conf)
                        if not attr.startswith("_")
                        and not callable(getattr(conf, attr))
                    ],
                ),
            )
        return hashlib.md5("".join(hash_parts).encode()).hexdigest()

    def _detect_n_repeats(self) -> tuple[int, str]:
        """Return repeated-evaluation count and key suffix for sampler-driven runs.

        Returns:
            ``(n_splits, "fold")`` for ``KFoldSampler``,
            ``(n_splits, "split")`` for ``ShuffleSampler``, and
            ``(1, "fold")`` for all other samplers.
        """
        sampler = BaseSampler.resolve(self.data)
        if isinstance(sampler, KFoldSampler):
            return sampler.n_splits, "fold"
        if isinstance(sampler, ShuffleSampler):
            return sampler.n_splits, "split"
        return 1, "fold"

    def _run_single_pipeline(
        self,
        model_file_outputs: dict,
        attack_file_outputs: dict,
        *,
        run_idx: int | None = None,
    ) -> dict:
        """Run model training, optional attack, and optional custom scoring for the
        current state of ``self.data`` (already loaded and sampled).

        Returns the accumulated score dict for this pipeline pass.
        """
        scores = {}
        scores.update(**self.data.score_dict)
        self._propagate_score_mode()

        if self.model:
            if self.defense is not None:
                self._run_experiment_stage_hooks(
                    "before",
                    "apply_fit_defense",
                    component="defense",
                    run_idx=run_idx,
                )
            self._run_experiment_stage_hooks(
                "before",
                "train",
                component="model",
                run_idx=run_idx,
            )
            cached_model = self._cache_stage_get(
                stage="train",
                component="model",
                identity={"run_idx": run_idx},
            )
            if isinstance(cached_model, dict):
                self.model.score_dict = dict(cached_model.get("score_dict", {}))
                self.model.training_predictions = cached_model.get(
                    "training_predictions",
                )
                self.model.predictions = cached_model.get("predictions")
                self.model.training_probabilities = cached_model.get(
                    "training_probabilities",
                )
                self.model.probabilities = cached_model.get("probabilities")
                scores.update(**self.model.score_dict)
                self._run_experiment_stage_hooks(
                    "after",
                    "train",
                    component="model",
                    run_idx=run_idx,
                    cache_hit=True,
                )
            else:
                if hasattr(self.model, "set_epoch_attack") and callable(
                    getattr(self.model, "set_epoch_attack"),
                ):
                    self.model.set_epoch_attack(self.attack)
                self._clear_model_prediction_outputs()
                self.model(data=self.data, **model_file_outputs)
                assert hasattr(
                    self.model,
                    "training_predictions",
                ), "model must have training_predictions attribute after training"
                assert hasattr(
                    self.model,
                    "predictions",
                ), "model must have predictions attribute after training"
                assert hasattr(
                    self.model,
                    "score_dict",
                ), "model must have score_dict attribute after training"
                scores.update(**self.model.score_dict)
                if hasattr(self.model, "set_epoch_attack") and callable(
                    getattr(self.model, "set_epoch_attack"),
                ):
                    self.model.set_epoch_attack(None)
                self._cache_stage_set(
                    stage="train",
                    component="model",
                    identity={"run_idx": run_idx},
                    value={
                        "score_dict": dict(
                            getattr(self.model, "score_dict", {}) or {},
                        ),
                        "training_predictions": getattr(
                            self.model,
                            "training_predictions",
                            None,
                        ),
                        "predictions": getattr(self.model, "predictions", None),
                        "training_probabilities": getattr(
                            self.model,
                            "training_probabilities",
                            None,
                        ),
                        "probabilities": getattr(self.model, "probabilities", None),
                    },
                )
                self._run_experiment_stage_hooks(
                    "after",
                    "train",
                    component="model",
                    run_idx=run_idx,
                    cache_hit=False,
                )
            if self.defense is not None:
                self._run_experiment_stage_hooks(
                    "after",
                    "apply_fit_defense",
                    component="defense",
                    run_idx=run_idx,
                )
                self._run_experiment_stage_hooks(
                    "before",
                    "apply_predict_defense",
                    component="defense",
                    run_idx=run_idx,
                )
                self._run_experiment_stage_hooks(
                    "after",
                    "apply_predict_defense",
                    component="defense",
                    run_idx=run_idx,
                )
            self._run_experiment_stage_hooks(
                "before",
                "model_score",
                component="model",
                run_idx=run_idx,
            )
            self._run_experiment_stage_hooks(
                "after",
                "model_score",
                component="model",
                run_idx=run_idx,
            )
            scores.update(**dict(getattr(self.model, "score_dict", {}) or {}))
            self._run_experiment_stage_hooks(
                "before",
                "model_persist",
                component="model",
                run_idx=run_idx,
            )
            self._run_experiment_stage_hooks(
                "after",
                "model_persist",
                component="model",
                run_idx=run_idx,
            )
        else:
            logger.info("No model config provided, skipping model training.")

        attack_chain = getattr(self, "_attack_chain", None)
        if attack_chain is None:
            attack_chain = [self.attack] if self.attack is not None else []

        if len(attack_chain) > 0:
            multi_attack = len(attack_chain) > 1
            try:
                for attack_cfg in attack_chain:
                    attack_component = (
                        f"attack:{attack_cfg.alias}" if multi_attack else "attack"
                    )
                    self._run_experiment_stage_hooks(
                        "before",
                        "generation",
                        component=attack_component,
                        run_idx=run_idx,
                    )
                    self._run_experiment_stage_hooks(
                        "before",
                        "attack",
                        component=attack_component,
                        run_idx=run_idx,
                    )
                    cached_attack = self._cache_stage_get(
                        stage="attack",
                        component=attack_component,
                        identity={"run_idx": run_idx},
                    )
                    if isinstance(cached_attack, dict):
                        attack_cfg.score_dict = dict(
                            cached_attack.get("score_dict", {}),
                        )
                        attack_cfg.attack_predictions = cached_attack.get(
                            "attack_predictions",
                        )
                        scores = merge_scores_with_collision_suffix(
                            scores,
                            attack_cfg.score_dict,
                            alias=attack_cfg.alias if multi_attack else None,
                        )
                        self._run_experiment_stage_hooks(
                            "after",
                            "attack",
                            component=attack_component,
                            run_idx=run_idx,
                            cache_hit=True,
                        )
                        self._run_experiment_stage_hooks(
                            "after",
                            "generation",
                            component=attack_component,
                            run_idx=run_idx,
                            cache_hit=True,
                        )
                        self._run_experiment_stage_hooks(
                            "before",
                            "attack_score",
                            component=attack_component,
                            run_idx=run_idx,
                        )
                        self._run_experiment_stage_hooks(
                            "after",
                            "attack_score",
                            component=attack_component,
                            run_idx=run_idx,
                        )
                        scores = merge_scores_with_collision_suffix(
                            scores,
                            dict(getattr(attack_cfg, "score_dict", {}) or {}),
                            alias=attack_cfg.alias if multi_attack else None,
                        )
                        self._run_experiment_stage_hooks(
                            "before",
                            "attack_persist",
                            component=attack_component,
                            run_idx=run_idx,
                        )
                        self._run_experiment_stage_hooks(
                            "after",
                            "attack_persist",
                            component=attack_component,
                            run_idx=run_idx,
                        )
                        continue
                    run_outputs = self._build_attack_file_outputs_for_run(
                        attack_file_outputs,
                        attack_cfg,
                        multi_attack=multi_attack,
                    )
                    attack_cfg(
                        data=self.data,
                        model=self.model,
                        **run_outputs,
                    )
                    assert hasattr(
                        attack_cfg,
                        "attack",
                    ), "attack must have attack attribute after training"
                    assert hasattr(
                        attack_cfg,
                        "attack_predictions",
                    ), "attack must have a predictions attribute after training"
                    assert hasattr(
                        attack_cfg,
                        "score_dict",
                    ), "attack must have score_dict attribute after training"
                    scores = merge_scores_with_collision_suffix(
                        scores,
                        attack_cfg.score_dict,
                        alias=attack_cfg.alias if multi_attack else None,
                    )
                    self._cache_stage_set(
                        stage="attack",
                        component=attack_component,
                        identity={"run_idx": run_idx},
                        value={
                            "score_dict": dict(
                                getattr(attack_cfg, "score_dict", {}) or {},
                            ),
                            "attack_predictions": getattr(
                                attack_cfg,
                                "attack_predictions",
                                None,
                            ),
                        },
                    )
                    self._run_experiment_stage_hooks(
                        "after",
                        "attack",
                        component=attack_component,
                        run_idx=run_idx,
                        cache_hit=False,
                    )
                    self._run_experiment_stage_hooks(
                        "after",
                        "generation",
                        component=attack_component,
                        run_idx=run_idx,
                        cache_hit=False,
                    )
                    self._run_experiment_stage_hooks(
                        "before",
                        "attack_score",
                        component=attack_component,
                        run_idx=run_idx,
                    )
                    self._run_experiment_stage_hooks(
                        "after",
                        "attack_score",
                        component=attack_component,
                        run_idx=run_idx,
                    )
                    scores = merge_scores_with_collision_suffix(
                        scores,
                        dict(getattr(attack_cfg, "score_dict", {}) or {}),
                        alias=attack_cfg.alias if multi_attack else None,
                    )
                    self._run_experiment_stage_hooks(
                        "before",
                        "attack_persist",
                        component=attack_component,
                        run_idx=run_idx,
                    )
                    self._run_experiment_stage_hooks(
                        "after",
                        "attack_persist",
                        component=attack_component,
                        run_idx=run_idx,
                    )

                self.attack = attack_chain[0]
            except ValueError as e:
                logger.debug(e)
                raise
        else:
            logger.info("No attack config provided, skipping attack.")

        if self.detector:
            if len(attack_chain) == 0:
                raise ValueError(
                    "Detector phase requires an attack configuration/output.",
                )
            self._run_experiment_stage_hooks(
                "before",
                "detector-train",
                component="detector",
                run_idx=run_idx,
            )
            self._run_experiment_stage_hooks(
                "before",
                "detector-defense",
                component="detector",
                run_idx=run_idx,
            )
            self._run_experiment_stage_hooks(
                "before",
                "defense",
                component="detector",
                run_idx=run_idx,
            )
            cached_detector = self._cache_stage_get(
                stage="defense",
                component="detector",
                identity={"run_idx": run_idx},
            )
            if isinstance(cached_detector, dict):
                self.detector.score_dict = dict(cached_detector.get("score_dict", {}))
                scores.update(**self.detector.score_dict)
                self._run_experiment_stage_hooks(
                    "after",
                    "defense",
                    component="detector",
                    run_idx=run_idx,
                    cache_hit=True,
                )
                self._run_experiment_stage_hooks(
                    "after",
                    "detector-defense",
                    component="detector",
                    run_idx=run_idx,
                    cache_hit=True,
                )
                self._run_experiment_stage_hooks(
                    "after",
                    "detector-train",
                    component="detector",
                    run_idx=run_idx,
                    cache_hit=True,
                )
            else:
                detector_attack = self._build_detector_attack_view(attack_chain)
                self.detector(
                    data=self.data,
                    model=self.model,
                    attack=detector_attack,
                )
                assert hasattr(
                    self.detector,
                    "score_dict",
                ), "detector must have score_dict attribute after execution"
                scores.update(**self.detector.score_dict)
                self._cache_stage_set(
                    stage="defense",
                    component="detector",
                    identity={"run_idx": run_idx},
                    value={
                        "score_dict": dict(
                            getattr(self.detector, "score_dict", {}) or {},
                        ),
                    },
                )
                self._run_experiment_stage_hooks(
                    "after",
                    "defense",
                    component="detector",
                    run_idx=run_idx,
                    cache_hit=False,
                )
                self._run_experiment_stage_hooks(
                    "after",
                    "detector-defense",
                    component="detector",
                    run_idx=run_idx,
                    cache_hit=False,
                )
                self._run_experiment_stage_hooks(
                    "after",
                    "detector-train",
                    component="detector",
                    run_idx=run_idx,
                    cache_hit=False,
                )
            self._run_experiment_stage_hooks(
                "before",
                "detector_score",
                component="detector",
                run_idx=run_idx,
            )
            self._run_experiment_stage_hooks(
                "after",
                "detector_score",
                component="detector",
                run_idx=run_idx,
            )
            scores.update(**dict(getattr(self.detector, "score_dict", {}) or {}))
            self._run_experiment_stage_hooks(
                "before",
                "detector_persist",
                component="detector",
                run_idx=run_idx,
            )
            self._run_experiment_stage_hooks(
                "after",
                "detector_persist",
                component="detector",
                run_idx=run_idx,
            )
        else:
            logger.info("No detector config provided, skipping detector phase.")

        self._run_experiment_stage_hooks(
            "before",
            "score",
            component="experiment",
            run_idx=run_idx,
        )
        cached_scores = self._cache_stage_get(
            stage="score",
            component="experiment",
            identity={"run_idx": run_idx},
        )
        if isinstance(cached_scores, dict):
            custom_scores = dict(cached_scores.get("score_dict", {}))
            self._run_experiment_stage_hooks(
                "after",
                "score",
                component="experiment",
                run_idx=run_idx,
                cache_hit=True,
            )
        else:
            custom_scores = self._run_experiment_scorer_modes(score_file=None)
            self._cache_stage_set(
                stage="score",
                component="experiment",
                identity={"run_idx": run_idx},
                value={"score_dict": dict(custom_scores or {})},
            )
            self._run_experiment_stage_hooks(
                "after",
                "score",
                component="experiment",
                run_idx=run_idx,
                cache_hit=False,
            )
        if custom_scores:
            scores = {**scores, **custom_scores}

        return scores

    @staticmethod
    def _aggregate_repeated_score_values(values: list[Any]) -> Any:
        """Aggregate repeated score values, averaging numeric leaves recursively."""
        if not values:
            return None

        present = [value for value in values if value is not None]
        if not present:
            return values[-1]

        if all(isinstance(value, Mapping) for value in present):
            aggregated: dict[str, Any] = {}
            all_keys = set().union(*(value.keys() for value in present))
            for key in all_keys:
                aggregated[key] = ExperimentConfig._aggregate_repeated_score_values(
                    [
                        value.get(key) if isinstance(value, Mapping) else None
                        for value in values
                    ],
                )
            return aggregated

        try:
            return float(np.mean([float(value) for value in present]))
        except (TypeError, ValueError):
            return present[-1]

    @staticmethod
    def _aggregate_repeated_scores(
        per_run_scores: list,
        suffix: str = "fold",
    ) -> dict:
        """Merge per-run score dicts into a single dict.

        Per-run score payloads are stored under ``{suffix}-{i}`` keys. Top-level
        values are aggregated recursively: numeric leaves use the mean across
        runs, while non-numeric leaves keep the last non-``None`` value.

        Args:
            per_run_scores: One score dict per repeated run, in order.
            suffix: Suffix used for per-run keys such as ``fold`` or ``split``.

        Returns:
            Aggregated score dictionary.
        """
        if not per_run_scores:
            return {}

        aggregated = {
            f"{suffix}-{i}": dict(run or {}) for i, run in enumerate(per_run_scores)
        }
        all_keys = set().union(*per_run_scores)
        for key in all_keys:
            values = [run.get(key) for run in per_run_scores]
            aggregated[key] = ExperimentConfig._aggregate_repeated_score_values(values)
        return aggregated

    @staticmethod
    def from_yaml(filepath: str) -> "ExperimentConfig":
        """Load an ExperimentConfig runtime snapshot from canonical YAML.

        Args:
            filepath: Path to runtime YAML payload.

        Returns:
            Instantiated ExperimentConfig runtime object.

        Raises:
            TypeError: If loaded payload cannot be instantiated as ExperimentConfig.
            ValueError: If schema payload is malformed or from unsupported future version.
        """
        resolved_path = BaseConfig._resolve_yaml_read_path(filepath)
        payload = OmegaConf.to_container(OmegaConf.load(resolved_path), resolve=True)
        if not isinstance(payload, dict):
            raise TypeError(
                f"Loaded config is not a dictionary from {resolved_path}",
            )

        schema_version = payload.get("schema_version")
        if schema_version is None:
            instance = instantiate(payload)
            if not isinstance(instance, ExperimentConfig):
                raise TypeError(
                    f"Object loaded from {resolved_path} is not an ExperimentConfig: {type(instance)}",
                )
            return instance

        major = ExperimentConfig._extract_schema_major(str(schema_version))
        current_major = ExperimentConfig._extract_schema_major(
            CANONICAL_EXPERIMENT_RUNTIME_SCHEMA_VERSION,
        )
        if major > current_major:
            raise ValueError(
                "Cannot load future experiment runtime schema "
                f"'{schema_version}' with runtime expecting up to v{current_major}.",
            )

        experiment_payload = payload.get("experiment")
        if not isinstance(experiment_payload, dict):
            raise ValueError(
                "Experiment runtime YAML is missing required 'experiment' payload.",
            )

        instance = instantiate(experiment_payload)
        if not isinstance(instance, ExperimentConfig):
            raise TypeError(
                "Runtime state payload did not instantiate ExperimentConfig; "
                f"got {type(instance)}",
            )
        return instance

    def run(
        self,
    ) -> dict:
        """Execute full experiment pipeline and return aggregated score payload.

        This is the canonical public entrypoint for experiment execution.

        Returns:
            Aggregated experiment score payload.
        """
        run_start = time.process_time()
        self.params = build_experiment_params_manifest(self)
        self._runtime_cache = self._load_runtime_cache()
        self.outputs.setdefault("cache", {})
        # Initialize Scores
        scores = {}
        # Set random seed
        self.set_random_seed()
        # Set device
        if self.library not in ["sklearn"]:
            self.set_device()
        # Get file paths
        file_dict = self.files.as_dict()
        data_file_outputs = {
            file: getattr(self.files, file) for file in data_files if file in file_dict
        }
        model_file_outputs = {
            file: getattr(self.files, file)
            for file in model_files
            if file in file_dict
        }
        attack_file_outputs = {
            file: getattr(self.files, file)
            for file in attack_files
            if file in file_dict
        }

        # ------------------------------------------------------------------
        # Data loading (always done once; sampling may repeat per fold)
        # ------------------------------------------------------------------
        self._run_experiment_stage_hooks("before", "load", component="data")
        data_file_path = data_file_outputs.get("data_file", None)
        if (
            isinstance(data_file_path, (str, Path))
            and str(data_file_path).strip() != ""
            and Path(data_file_path).exists()
        ):
            configured_data = self.data
            cached_data = self.load_object(
                str(data_file_path),
                ignore_corrupt=True,
                delete_corrupt=True,
            )
            if cached_data is not None:
                self.data = cached_data
                self._apply_runtime_data_split_overrides(self.data, configured_data)
            else:
                # A truncated pickle should behave like a cache miss so reruns can recover.
                self._run_experiment_stage_hooks(
                    "before",
                    "data_score",
                    component="data",
                    run_idx=None,
                )
                self.data(files=data_file_outputs)
                self._run_experiment_stage_hooks(
                    "after",
                    "data_score",
                    component="data",
                    run_idx=None,
                )
        else:
            # Load raw data only (no sample yet when evaluating repeated splits)
            n_repeats, _ = self._detect_n_repeats()
            if n_repeats > 1:
                self.data.load_dataset()
            else:
                self._run_experiment_stage_hooks(
                    "before",
                    "data_score",
                    component="data",
                    run_idx=None,
                )
                self.data(files=data_file_outputs)
                self._run_experiment_stage_hooks(
                    "after",
                    "data_score",
                    component="data",
                    run_idx=None,
                )
        self._run_experiment_stage_hooks("after", "load", component="data")

        assert hasattr(self.data, "X_train") or hasattr(
            self.data,
            "_X",
        ), "data must be loaded before running the pipeline"

        self._ensure_active_mode_split_available()

        n_repeats, run_suffix = self._detect_n_repeats()

        if n_repeats > 1:
            # ------------------------------------------------------------------
            # Repeated split evaluation: run one pipeline pass per split/fold
            # ------------------------------------------------------------------
            logger.info(
                f"Running {n_repeats} repeated {run_suffix} evaluations.",
            )
            per_run_scores: list = []
            for run_idx in range(n_repeats):
                logger.info(f"  {run_suffix.title()} {run_idx + 1}/{n_repeats}")
                # Reset sampling state so _sample() runs fresh for this run
                self.data.split = run_idx
                self.data.data_sample_time = None
                for attr in self._sample_cache_fields():
                    setattr(self.data, attr, None)
                self.data.score_dict = {}
                self._run_experiment_stage_hooks(
                    "before",
                    "sample",
                    component="data",
                    run_idx=run_idx,
                )
                cached_sample = self._cache_stage_get(
                    stage="sample",
                    component="data",
                    identity={"run_idx": run_idx, "suffix": run_suffix},
                )
                if isinstance(cached_sample, dict):
                    self._apply_sample_cache_value(cached_sample)
                    self._run_experiment_stage_hooks(
                        "after",
                        "sample",
                        component="data",
                        run_idx=run_idx,
                        cache_hit=True,
                    )
                else:
                    # Run full data runtime per fold so DataConfig hooks and
                    # transformations (e.g., StringDistanceTransformer) execute.
                    self._run_experiment_stage_hooks(
                        "before",
                        "data_score",
                        component="data",
                        run_idx=run_idx,
                    )
                    self.data(
                        files={
                            "data_file": None,
                            "score_file": None,
                        },
                    )
                    self._run_experiment_stage_hooks(
                        "after",
                        "data_score",
                        component="data",
                        run_idx=run_idx,
                    )
                    self.data.score_dict.update(
                        data_load_time=self.data.data_load_time,
                        data_sample_time=self.data.data_sample_time,
                        train_n=self.data.train_n,
                        test_n=self.data.test_n,
                    )
                    self._cache_stage_set(
                        stage="sample",
                        component="data",
                        identity={"run_idx": run_idx, "suffix": run_suffix},
                        value=self._capture_sample_cache_value(),
                    )
                    self._run_experiment_stage_hooks(
                        "after",
                        "sample",
                        component="data",
                        run_idx=run_idx,
                        cache_hit=False,
                    )
                fold_scores = self._run_single_pipeline(
                    model_file_outputs,
                    attack_file_outputs,
                    run_idx=run_idx,
                )
                per_run_scores.append(fold_scores)

            scores = self._aggregate_repeated_scores(per_run_scores, run_suffix)
        else:
            # ------------------------------------------------------------------
            # Single-pass (non-fold) pipeline
            # ------------------------------------------------------------------
            assert hasattr(
                self.data,
                "X_train",
            ), "data must return an object with X_train attribute"
            assert hasattr(
                self.data,
                "y_train",
            ), "data must return an object with y_train attribute"
            assert hasattr(
                self.data,
                "X_test",
            ), "data must return an object with X_test attribute"
            assert hasattr(
                self.data,
                "y_test",
            ), "data must return an object with y_test attribute"
            assert hasattr(
                self.data,
                "score_dict",
            ), "data must have score_dict attribute after loading"
            scores = self._run_single_pipeline(
                model_file_outputs,
                attack_file_outputs,
                run_idx=None,
            )
            if self.model is None:
                self.model = None

        self.score_dict = dict(scores)
        self.merge_runtime_files(file_dict)
        scores = dict(self.score_dict)

        self.times.update(
            {
                "data_load_time": getattr(self.data, "data_load_time", None),
                "data_sample_time": getattr(self.data, "data_sample_time", None),
                "model_training_time": (
                    getattr(self.model, "training_time", None)
                    if self.model is not None
                    else None
                ),
                "attack_time": (
                    sum(
                        float(getattr(attack_cfg, "attack_time", 0.0) or 0.0)
                        for attack_cfg in getattr(self, "_attack_chain", [])
                    )
                    if len(getattr(self, "_attack_chain", [])) > 0
                    else None
                ),
                "detector_time": (
                    getattr(self.detector, "detector_time", None)
                    if self.detector is not None
                    else None
                ),
            },
        )
        self.times["experiment_total_time"] = time.process_time() - run_start

        self.outputs["scores"] = dict(scores)
        self.outputs["files"] = dict(file_dict)
        self.outputs["cache"]["path"] = self._cache_file_path()
        self.params = build_experiment_params_manifest(self)

        self._run_experiment_stage_hooks("before", "persist", component="experiment")
        if "score_file" in file_dict:
            scores = self.merge_and_persist_scores(
                scores,
                file_dict["score_file"],
            )
        else:
            logger.info("No score_file specified, skipping score saving.")
        self._persist_runtime_cache()
        self._persist_runtime_state(file_dict)
        self._run_experiment_stage_hooks("after", "persist", component="experiment")

        for key in CANONICAL_EXPERIMENT_TIMES:
            self.score_dict[key] = self.times.get(key)
        return scores

    def __call__(
        self,
    ) -> dict:
        """Backward-compatible callable alias for experiment execution."""
        return self.run()
