"""Experiment orchestration primitives for Deckard's Python API.

This module contains the base experiment configuration object that ties data,
model, defense, attack, files, and scorers into a single executable unit.
"""

import logging
import warnings
import hashlib
from typing import List, Union, Literal, Any
from omegaconf import DictConfig, ListConfig, OmegaConf
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from hydra.utils import instantiate

from ..data import DataConfig, DataPipelineConfig
from ..model import ModelConfig

try:
    from ..data import FairlearnDataConfig
except ImportError:  # pragma: no cover
    FairlearnDataConfig = None
from ..model.defend import DefensePipelineConfig
from ..attack import AttackConfig
from ..detector import DetectorConfig
from ..score import ScorerDictConfig
from ..file import FileConfig, data_files, model_files, attack_files
from ..utils import (
    ConfigBase,
    coerce_config,
    coerce_to_list,
    instantiate_config,
    is_default_config_value,
    is_null_config_value,
    merge_scores_with_collision_suffix,
    split_comma_separated_tokens,
)
from ..score.base import coerce_scorer_config, _DataScorerMarker, _AttackProfileScorer
from ..data.sample import KFoldSampler, ShuffleSampler

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from ..data import AnjanaDataConfig
except ImportError:  # pragma: no cover
    AnjanaDataConfig = None


try:
    from ..model import FairlearnModelConfig
except ImportError:  # pragma: no cover
    FairlearnModelConfig = None


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
            if AnjanaDataConfig is None:
                raise ImportError(
                    "AnjanaDataConfig requires optional anjana dependencies. Install deckard[anjana] to enable anjana data configs.",
                )
            return AnjanaDataConfig
        if any(key in data_dict for key in self._fairness_keys):
            if FairlearnDataConfig is None:
                raise ImportError(
                    "FairlearnDataConfig requires optional fairness dependencies. Install deckard[fairlearn] to enable fairlearn data configs.",
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

        if hasattr(self.data, "_target_") and not isinstance(self.data, (dict, DictConfig, str, ConfigBase)):
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


class ExperimentConfig(DataConfigResolutionMixin, ConfigBase):
    """Compose and execute a complete Deckard experiment.

    An experiment coordinates data loading, optional defense application, model
    training or loading, adversarial attack execution, scoring, and artifact
    persistence through ``FileConfig``.
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
    score_modes: Union[list[str], None] = None

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
        if self.score_modes is not None:
            return list(self.score_modes)
        if self.evaluation_mode == "tuning":
            return ["val"]
        if self.evaluation_mode == "report":
            return ["train", "test", "val"]
        return ["train"]

    @staticmethod
    def _normalize_mode_score_keys(mode: str, mode_scores: dict) -> dict:
        if mode == "val":
            return {f"validation_{k}": v for k, v in mode_scores.items()}
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

    def _run_experiment_scorer_modes(self, score_file=None) -> dict:
        if self.score is None:
            return {}
        out = {}
        for mode in self._resolve_score_modes():
            self._ensure_mode_predictions(mode)
            mode_scores = self.score(
                data=self.data,
                model=self.model,
                attack=self.attack,
                mode=mode,
                score_file=None,
            )
            out.update(self._normalize_mode_score_keys(mode, mode_scores))
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

            score=classification              # → model.scorer
            score=data-classification        # → data.scorer (_DataScorerMarker)
            score=evasion-classification     # → attack scorer (_AttackProfileScorer)

        **Null / auto / default** → components self-configure from their own defaults.
        """
        score_cfg = self.score

        # Null / auto → let each component self-configure.
        if score_cfg is None or is_null_config_value(score_cfg) or is_default_config_value(score_cfg):
            self.score = None
            return

        plain = OmegaConf.to_container(score_cfg, resolve=True) if isinstance(score_cfg, DictConfig) else (dict(score_cfg) if isinstance(score_cfg, dict) else None)

        if isinstance(plain, dict):
            # Auto-configure sentinel (from score/auto.yaml).
            if plain.get("_auto_configure"):
                self.score = None
                return

            # Scoped dict: keys are component names produced by Hydra @ package syntax
            # or by passing {"data": scorer, "model": scorer, ...} directly.
            _SCOPE_KEYS = {"data", "model", "attack", "detector", "experiment"}
            if any(k in _SCOPE_KEYS for k in plain):
                for scope in _SCOPE_KEYS:
                    if scope in plain:
                        self._route_scorer_to_scope(scope, coerce_scorer_config(plain[scope]))
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
            self.data.scorer = (
                ScorerDictConfig.merge(data_scorers) if len(data_scorers) > 1 else data_scorers[0]
            )
        if model_scorers and self.model is not None:
            self.model.scorer = (
                ScorerDictConfig.merge(model_scorers) if len(model_scorers) > 1 else model_scorers[0]
            )

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
            raise ValueError(f"Unsupported type for attack: {type(attack_obj)}") from exc

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
        if not hasattr(self, "classifier"):
            self.classifier = self.data.classifier
        else:
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
                overrides={"defense": self.defense} if self.defense is not None else None,
            )
        except TypeError as exc:
            raise ValueError(f"Unsupported type for model: {type(self.model)}") from exc

        if self.defense is not None:
            self.model.defense = self.defense
        assert isinstance(
            self.model,
            ModelConfig,
        ), "model must be an instance of ModelConfig"
        if self.classifier is None:
            self.classifier = self.model.classifier
        else:
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
            if FairlearnModelConfig is None:
                raise ImportError(
                    "FairlearnModelConfig requires optional fairness dependencies. Install deckard[fairlearn] to enable fairlearn model configs.",
                )
            if not isinstance(self.model, FairlearnModelConfig):
                self.model = FairlearnModelConfig(
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
        self._attack_chain = self._normalize_attack_chain(self.attack)
        self._validate_multi_attack_aliases(self._attack_chain)
        if len(self._attack_chain) > 0:
            # Preserve backward compatibility for single-attack call sites.
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
                raise ValueError(f"Unsupported type for files: {type(self.files)}") from exc
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
            if isinstance(normalized, (dict, list, tuple, str, int, float, bool)) or normalized is None:
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
                        if not attr.startswith("_") and not callable(getattr(conf, attr))
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

        attack_chain = getattr(self, "_attack_chain", None)
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
        # Get file paths
        file_dict = self.files._get_file_dict()
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
        if (
            "data_file" in data_file_outputs
            and Path(data_file_outputs["data_file"]).exists()
        ):
            self.data = self.load_object(data_file_outputs["data_file"])
        else:
            # Load raw data only (no sample yet when evaluating repeated splits)
            n_repeats, _ = self._detect_n_repeats()
            if n_repeats > 1:
                self.data._load_data()
            else:
                self.data(**data_file_outputs)

        assert hasattr(self.data, "X_train") or hasattr(
            self.data,
            "_X",
        ), "data must be loaded before running the pipeline"

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
                self.data.fold = run_idx
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
                self.data.score_dict = {}
                self.data._sample()
                self.data.score_dict.update(
                    data_load_time=self.data.data_load_time,
                    data_sample_time=self.data.data_sample_time,
                    train_n=self.data.train_n,
                    test_n=self.data.test_n,
                )
                fold_scores = self._run_single_pipeline(
                    model_file_outputs,
                    attack_file_outputs,
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
            )
            if self.model is None:
                self.model = None

        if "score_file" in file_dict:
            scores = self.merge_and_persist_scores(
                scores,
                file_dict["score_file"],
            )
        else:
            logger.info("No score_file specified, skipping score saving.")
        return scores
