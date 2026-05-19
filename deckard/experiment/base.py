"""Experiment orchestration primitives for deckard's Python API.

This module contains the base experiment configuration object that ties data,
model, defense, attack, files, and scorers into a single executable unit.
"""

import hashlib
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Literal, Union

import numpy as np
import pandas as pd
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from ..attack import AttackConfig
from ..data import DataConfig, DataPipelineConfig
from ..data.sample import KFoldSampler, ShuffleSampler
from ..detector import DetectorConfig
from ..file import AttackFiles, BaseFiles, FileConfig, ModelFiles
from ..frameworks import ExperimentContractMixin, FrameworkExperimentConfig
from ..model import ModelConfig
from ..model.defend import DefensePipelineConfig
from ..score import ScorerDictConfig
from ..score.base import _AttackProfileScorer, _DataScorerMarker, coerce_scorer_config
from ..utils import (
    ConfigBase,
    coerce_config,
    coerce_to_list,
    instantiate_config,
    is_default_config_value,
    is_null_config_value,
    load_class,
    merge_scores_with_collision_suffix,
    split_comma_separated_tokens,
)

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


try:
    from ..model import FairlearnModelConfig, FairlearnPytorchModelConfig
except ImportError:  # pragma: no cover
    FairlearnModelConfig = None
    FairlearnPytorchModelConfig = None

try:
    from ..model import PytorchModelConfig
except ImportError:  # pragma: no cover
    PytorchModelConfig = None


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


class DataConfigResolutionMixin:
    """Resolve ExperimentConfig.data into the appropriate DataConfig subtype."""

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
        elif isinstance(data_obj, ConfigBase):
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
            try:
                from deckard.plugins.anjana.data import AnjanaDataConfig
            except ImportError:
                raise ImportError(
                    "Privacy features need `anjana`. Install with `pip install deckard[anjana]`",
                )
            return AnjanaDataConfig

        if any(key in data_dict for key in self._fairness_keys):
            try:
                from deckard.plugins.fairlearn.data import FairlearnDataConfig
            except ImportError:
                raise ImportError(
                    "Fairness features need `fairlearn`. Install with `pip install deckard[fairlearn]`",
                )
            return FairlearnDataConfig

        if "pipeline" in data_dict:
            return DataPipelineConfig

        return DataConfig

    def _resolve_data_config(self):
        if self.data is None:
            raise ValueError("data must be provided")

        if isinstance(self.data, DataConfig):
            return self.data

        if hasattr(self.data, "_target_") and not isinstance(
            self.data,
            (dict, DictConfig, str, ConfigBase),
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


@dataclass(eq=False, kw_only=True)
class ExperimentConfig(
    DataConfigResolutionMixin,
    ExperimentContractMixin,
    ConfigBase,
    FrameworkExperimentConfig,
):
    """Compose and execute a complete deckard experiment.

    This config coordinates data loading, optional defenses, model runtime,
    attacks, scoring, and artifact persistence through ``FileConfig``.
    """

    data: Union[DataConfig, DataPipelineConfig]
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
    score_mode: Union[
        Literal["train", "test", "val", "pre-sample"],
        list[Literal["train", "test", "val", "pre-sample"]],
        None,
    ] = None

    def _validate_mode_configuration(self) -> None:
        """Ensure exactly one experiment mode-routing strategy is active."""
        # ``standard`` acts as the neutral preset, so it can coexist with
        # explicit ``score_mode`` without ambiguity.
        if self.score_mode is not None and self.evaluation_mode != "standard":
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
        if self.score_mode is not None:
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

        allowed = {"pre-sample", "train", "test", "val"}
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
            y_true = getattr(self.data, "y", None)
            y_pred = getattr(self.data, "X", None)
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
            "sample",
            "fold",
            "val_size",
            "train_size",
            "test_size",
            "stratify",
            "random_state",
        ):
            if hasattr(configured_data, attr):
                setattr(loaded_data, attr, getattr(configured_data, attr))

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
            hasattr(self.data, "_sample")
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
            self.data._sample()

        if (
            getattr(self.data, "X_val", None) is None
            or getattr(self.data, "y_val", None) is None
        ):
            raise ValueError(
                "score_mode='val' requires validation data (X_val/y_val), but no validation split is available.",
            )

    def _resolve_component_score_mode(
        self,
    ) -> Literal["train", "test", "val", "pre-sample"]:
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

    def _propagate_score_mode(self) -> Literal["train", "test", "val", "pre-sample"]:
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
        attack_chain = self.attack_chain
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

    @staticmethod
    def _normalize_mode_score_keys(mode: str, mode_scores: dict) -> dict:
        if mode == "val":
            return {f"validation_{k}": v for k, v in mode_scores.items()}
        if mode == "pre-sample":
            return {f"presample_{k}": v for k, v in mode_scores.items()}
        return mode_scores

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
        if not hasattr(self.model, "_predict"):
            raise ValueError("Validation scoring requires model._predict")
        val_predictions = self.model._predict(self.data.X_val)
        self.model.val_predictions = val_predictions
        return val_predictions

    def _ensure_mode_predictions(self, mode: str):
        if mode == "pre-sample":
            return
        if self.model is None:
            raise ValueError(
                f"{mode} scoring requires a model, but model is None",
            )
        if not hasattr(self.model, "_predict"):
            raise ValueError(f"{mode} scoring requires model._predict")
        if mode == "train":
            if getattr(self.model, "training_predictions", None) is None:
                self.model.training_predictions = self.model._predict(
                    self.data.X_train,
                )
            return
        if mode == "test":
            if getattr(self.model, "predictions", None) is None:
                self.model.predictions = self.model._predict(self.data.X_test)
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
        if y_pred is None and hasattr(self.model, "_predict"):
            y_pred = self.model._predict(X_split)
            setattr(self.model, pred_attr, y_pred)

        y_proba = getattr(self.model, proba_attr, None)
        if y_proba is None and getattr(self.model, "classifier", False):
            predict_proba = getattr(self.model, "_predict_proba", None)
            if callable(predict_proba):
                try:
                    y_proba = predict_proba(X_split)
                    setattr(self.model, proba_attr, y_proba)
                except Exception:
                    y_proba = None

        return y_true, y_pred, y_proba

    def _run_experiment_scorer_modes(self, score_file=None) -> dict:
        if self.score is None:
            return {}
        out = {}
        scorer_is_data_profile = isinstance(self.score, _DataScorerMarker)
        for mode in self._resolve_score_modes():
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
                y_proba = None
                if (
                    mode != "pre-sample"
                    and self.model is not None
                    and getattr(self.model, "classifier", False)
                ):
                    try:
                        _, _, y_proba = self._resolve_mode_model_outputs(mode)
                    except Exception:
                        y_proba = None
                mode_scores = self.score(
                    **common_kwargs,
                    y_true=y_true,
                    y_pred=y_pred,
                    y_proba=y_proba,
                )
            else:
                if mode == "pre-sample":
                    raise ValueError(
                        "pre-sample mode is only supported for data-profile experiment scorers.",
                    )
                self._ensure_mode_predictions(mode)
                y_true, y_pred, y_proba = self._resolve_mode_model_outputs(mode)
                mode_scores = self.score(
                    **common_kwargs,
                    y_true=y_true,
                    y_pred=y_pred,
                    y_proba=y_proba,
                )
            out.update(self._normalize_mode_score_keys(mode, mode_scores))
        return out

    def _apply_attack_profile_scorer(self, scorer) -> None:
        """Apply an _AttackProfileScorer to the configured attack chain."""
        attack_chain = self.attack_chain or []
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
                    "deckard.score.data.DefaultDataClassificationConfig"
                    if bool(getattr(self.data, "classifier", True))
                    else "deckard.score.data.DefaultDataRegressionConfig"
                ),
            )
        if scope == "model":
            return lambda: load_class(
                (
                    "deckard.score.base.DefaultClassifierConfig"
                    if bool(getattr(self.model, "classifier", True))
                    else "deckard.score.base.DefaultRegressorConfig"
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

        from ..plugins.fairlearn.score import FairlearnScoreDictConfig

        return FairlearnScoreDictConfig(
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
            "_target_": "deckard.plugins.anjana.score.DefaultAnjanaScorerConfig",
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

                Example:

                ```text
                Scoped dict (Hydra @ package syntax):
                    +score@score.data=data-classification
                    +score@score.model=classification
                    +score@score.attack=evasion-classification

                Produces score: {data: {...}, model: {...}, attack: {...}}.
                Each sub-key is routed directly to its component via
                _route_scorer_to_scope without any type-inference.

                Single config (type-based fallback):
                    score=classification              # -> model.scorer
                    score=data-classification         # -> data.scorer (_DataScorerMarker)
                    score=evasion-classification      # -> attack scorer (_AttackProfileScorer)
                ```

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
            _SCOPE_KEYS = {"data", "model", "attack", "detector", "experiment"}
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
            items = split_comma_separated_tokens(score_cfg)
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

    def set_device(self, device: Union[str, int] = "cpu"):
        """
        Set the computation device for the experiment based on the selected library.
        For TensorFlow, configures GPU/CPU usage.
        Args:
            device (Union[str, int]): Device to use ("cpu", "gpu", or GPU index).
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

        from deckard.plugins.fairlearn.data import FairlearnDataConfig

        if isinstance(
            self.data,
            FairlearnDataConfig,
        ):
            fairlearn_model_cls = FairlearnModelConfig
            fairlearn_pytorch_model_cls = FairlearnPytorchModelConfig
            pytorch_model_cls = PytorchModelConfig
            if (
                fairlearn_model_cls is None
                or fairlearn_pytorch_model_cls is None
                or pytorch_model_cls is None
            ):
                try:
                    from ..model import (
                        FairlearnModelConfig as _FairlearnModelConfig,
                    )
                    from ..model import (
                        FairlearnPytorchModelConfig as _FairlearnPytorchModelConfig,
                    )
                    from ..model import (
                        PytorchModelConfig as _PytorchModelConfig,
                    )
                except ImportError as exc:
                    raise ImportError(
                        "FairlearnModelConfig requires optional fairness dependencies. Install deckard[fairlearn] to enable fairlearn model configs.",
                    ) from exc

                fairlearn_model_cls = _FairlearnModelConfig
                fairlearn_pytorch_model_cls = _FairlearnPytorchModelConfig
                pytorch_model_cls = _PytorchModelConfig

            if fairlearn_model_cls is None:
                raise ImportError(
                    "FairlearnModelConfig requires optional fairness dependencies. Install deckard[fairlearn] to enable fairlearn model configs.",
                )

            is_torch_model = pytorch_model_cls is not None and isinstance(
                self.model,
                pytorch_model_cls,
            )
            target_model_cls = (
                fairlearn_pytorch_model_cls if is_torch_model else fairlearn_model_cls
            )

            if is_torch_model and target_model_cls is None:
                raise ImportError(
                    "FairlearnPytorchModelConfig requires optional fairness and torch dependencies. "
                    "Install deckard[fairlearn,torch] to enable fairness-aware pytorch model configs.",
                )

            fairness_types = tuple(
                cfg
                for cfg in (fairlearn_model_cls, fairlearn_pytorch_model_cls)
                if cfg is not None
            )

            if not isinstance(self.model, fairness_types):
                self.model = target_model_cls(
                    model_type=self.model.model_type,
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
        self.attack_chain = self._normalize_attack_chain(self.attack)
        self._validate_multi_attack_aliases(self.attack_chain)
        if len(self.attack_chain) > 0:
            # Preserve backward compatibility for single-attack call sites.
            self.attack = self.attack_chain[0]
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

    def __post_init__(self) -> None:
        if not hasattr(self, "score_dict") or self.score_dict is None:
            self.score_dict = {}
        if not hasattr(self, "_target_") or self._target_ is None:
            self._target_ = "deckard.experiment.ExperimentConfig"
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
            if len(self.attack_chain) > 0:
                config_list.extend(self.attack_chain)
            if self.detector and isinstance(self.detector, ConfigBase):
                config_list.append(self.detector)
            if self.score:
                config_list.append(self.score)
            self.experiment_name = self._hash_from_list(config_list)
        self._initialize_files()
        self._initialize_component_scorers()

        # Reconcile and enforce a single device across experiment/data/model.
        self._reconcile_component_devices()
        if self.library not in ["sklearn"]:
            self.set_device(self.device if self.device is not None else "cpu")

    def set_random_seed(self) -> None:
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
        Generate a hash string from a list of ConfigBase objects.
        The hash is generated by concatenating the string representations of the configurations
        and computing the MD5 hash of the resulting string.
        Args:
            config_list (List[ConfigBase]): List of ConfigBase objects to generate the hash from.
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
                ConfigBase,
            ), "All items in config_list must be ConfigBase or config-like values"
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

        Returns
        -------
        tuple[int, str]
            ``(n_splits, "fold")`` for ``KFoldSampler``,
            ``(n_splits, "split")`` for ``ShuffleSampler``,
            and ``(1, "fold")`` for all other samplers.
        """
        sampler = self.data._resolve_sample()
        if isinstance(sampler, KFoldSampler):
            return sampler.n_splits, "fold"
        if isinstance(sampler, ShuffleSampler):
            return sampler.n_splits, "split"
        return 1, "fold"

    def _run_single_pipeline(
        self,
        model_file_outputs: dict,
        attack_file_outputs: dict,
    ) -> dict:
        """Run model training, optional attack, and optional custom scoring for the
        current state of ``self.data`` (already loaded and sampled).

        Returns the accumulated score dict for this pipeline pass.
        """
        scores = {}
        scores.update(**self.data.score_dict)
        self._propagate_score_mode()

        if self.model:
            if hasattr(self.model, "set_epoch_attack") and callable(
                getattr(self.model, "set_epoch_attack"),
            ):
                self.model.set_epoch_attack(self.attack)
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
        else:
            logger.info("No model config provided, skipping model training.")

        attack_chain = self.attack_chain
        if attack_chain is None:
            attack_chain = [self.attack] if self.attack is not None else []

        if len(attack_chain) > 0:
            multi_attack = len(attack_chain) > 1
            try:
                for attack_cfg in attack_chain:
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
        else:
            logger.info("No detector config provided, skipping detector phase.")

        custom_scores = self._run_experiment_scorer_modes(score_file=None)
        if custom_scores:
            scores = {**scores, **custom_scores}

        return scores

    @staticmethod
    def _aggregate_repeated_scores(
        per_run_scores: list,
        suffix: str = "fold",
    ) -> dict:
        """Merge per-run score dicts into a single dict.

        For each key that is numeric in every run, the top-level value is the
        mean across runs and per-run values are stored under
        ``{key}_{suffix}_{i}``. Non-numeric values use the last run's value for
        the top-level key.

        Parameters
        ----------
        per_run_scores : list of dict
            One score dict per repeated run, in order.

        suffix : str, default "fold"
            Suffix used for per-run keys (e.g., ``fold`` or ``split``).

        Returns
        -------
        dict
        """
        if not per_run_scores:
            return {}
        aggregated = {}
        all_keys = set().union(*per_run_scores)
        for key in all_keys:
            values = [run.get(key) for run in per_run_scores]
            # Store per-fold values under qualified keys
            for i, v in enumerate(values):
                aggregated[f"{key}_{suffix}_{i}"] = v
            # Attempt numeric average for the top-level key
            try:
                numeric = [float(v) for v in values if v is not None]
                if numeric:
                    aggregated[key] = float(np.mean(numeric))
                else:
                    aggregated[key] = values[-1]
            except (TypeError, ValueError):
                aggregated[key] = values[-1]
        return aggregated

    @property
    def runtime_scores(self) -> dict[str, Any]:
        """Public accessor for the latest experiment score payload."""
        if self.score_dict is None:
            self.score_dict = {}
        return self.score_dict

    @runtime_scores.setter
    def runtime_scores(self, value: dict[str, Any] | None) -> None:
        """Set the latest experiment score payload."""
        self.score_dict = value or {}

    @property
    def attack_chain(self) -> list[AttackConfig] | None:
        """Public accessor for normalized attack chain runtime state."""
        return getattr(self, "_attack_chain", None)

    @attack_chain.setter
    def attack_chain(self, value: list[AttackConfig] | None) -> None:
        """Set normalized attack chain runtime state."""
        self._attack_chain = value

    def compose_file_output_behavior(self) -> tuple[dict, dict, dict, dict]:
        """Compose runtime file-output mappings for data/model/attack stages."""
        file_dict = self.files.as_dict()
        base_keys = set(BaseFiles.__annotations__.keys())
        model_keys = set(ModelFiles.__annotations__.keys())
        attack_keys = set(AttackFiles.__annotations__.keys())
        data_file_outputs = {
            file: getattr(self.files, file, None)
            for file in base_keys
            if file in file_dict
        }
        model_file_outputs = {
            file: getattr(self.files, file, None)
            for file in model_keys
            if file in file_dict
        }
        attack_file_outputs = {
            file: getattr(self.files, file, None)
            for file in attack_keys
            if file in file_dict
        }
        return file_dict, data_file_outputs, model_file_outputs, attack_file_outputs

    def compose_data_loading_behavior(self, data_file_outputs: dict) -> None:
        """Compose data loading behavior based on cache presence and repeat strategy."""
        data_file = data_file_outputs.get("data_file")
        if data_file and Path(data_file).exists():
            configured_data = self.data
            self.data = self.load_object(data_file)
            self._apply_runtime_data_split_overrides(self.data, configured_data)
            return

        # Load raw data only (no sample yet when evaluating repeated splits)
        n_repeats, _ = self._detect_n_repeats()
        if n_repeats > 1:
            self.data._load_data()
            return
        self.data(**data_file_outputs)

    def compose_repeat_strategy(self) -> tuple[int, str]:
        """Compose repeat strategy from configured sampler behavior."""
        return self._detect_n_repeats()

    def _reset_data_runtime_for_repeat(self, run_idx: int) -> None:
        """Reset split-dependent runtime state before a repeated split run."""
        self.data.split = run_idx
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
            "pipeline_fit_n",
            "pipeline_transform_n",
            "pipeline_fit_time",
            "pipeline_transform_time",
            "pipeline_y_fit_n",
            "pipeline_y_transform_n",
            "pipeline_y_fit_time",
            "pipeline_y_transform_time",
        ):
            setattr(self.data, attr, None)
        self.data.score_dict = {}

    def _run_repeated_pipeline_behavior(
        self,
        n_repeats: int,
        run_suffix: str,
        model_file_outputs: dict,
        attack_file_outputs: dict,
    ) -> dict:
        """Compose repeated split/fold pipeline behavior and aggregate scores."""
        logger.info(
            f"Running {n_repeats} repeated {run_suffix} evaluations.",
        )
        per_run_scores: list = []
        for run_idx in range(n_repeats):
            logger.info(f"  {run_suffix.title()} {run_idx + 1}/{n_repeats}")
            self._reset_data_runtime_for_repeat(run_idx)
            # Run full data runtime per split/fold so pipeline hooks and
            # transformations execute in the same path as normal single runs.
            self.data(
                data_file=None,
                score_file=None,
            )
            self.data.score_dict.update(
                data_load_time=self.data.data_load_time,
                data_sample_time=self.data.data_sample_time,
                train_n=self.data.train_n,
                test_n=self.data.test_n,
            )
            split_scores = self._run_single_pipeline(
                model_file_outputs,
                attack_file_outputs,
            )
            per_run_scores.append(split_scores)
        return self._aggregate_repeated_scores(per_run_scores, run_suffix)

    def _run_single_pass_pipeline_behavior(
        self,
        model_file_outputs: dict,
        attack_file_outputs: dict,
    ) -> dict:
        """Compose single-pass pipeline behavior."""
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
        )
        if self.model is None:
            self.model = None
        return scores

    def compose_pipeline_execution_behavior(
        self,
        model_file_outputs: dict,
        attack_file_outputs: dict,
    ) -> dict:
        """Compose experiment pipeline execution across repeated/single strategies."""
        n_repeats, run_suffix = self.compose_repeat_strategy()
        if n_repeats > 1:
            return self._run_repeated_pipeline_behavior(
                n_repeats=n_repeats,
                run_suffix=run_suffix,
                model_file_outputs=model_file_outputs,
                attack_file_outputs=attack_file_outputs,
            )
        return self._run_single_pass_pipeline_behavior(
            model_file_outputs=model_file_outputs,
            attack_file_outputs=attack_file_outputs,
        )

    def __call__(
        self,
    ) -> dict:
        # Initialize Scores
        scores = {}
        # Set random seed
        self.set_random_seed()
        # Set device
        if self.library not in ["sklearn"]:
            self.set_device()
        file_dict, data_file_outputs, model_file_outputs, attack_file_outputs = (
            self.compose_file_output_behavior()
        )

        # Data loading (always done once; sampling may repeat per split/fold)
        self.compose_data_loading_behavior(data_file_outputs)

        assert hasattr(self.data, "X_train") or hasattr(
            self.data,
            "_X",
        ), "data must be loaded before running the pipeline"

        self._ensure_active_mode_split_available()

        scores = self.compose_pipeline_execution_behavior(
            model_file_outputs=model_file_outputs,
            attack_file_outputs=attack_file_outputs,
        )

        if "score_file" in file_dict:
            scores = self.merge_and_persist_scores(
                scores,
                file_dict["score_file"],
            )
        else:
            logger.info("No score_file specified, skipping score saving.")
        return scores
