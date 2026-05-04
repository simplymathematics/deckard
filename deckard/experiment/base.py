"""Experiment orchestration primitives for Deckard's Python API.

This module contains the base experiment configuration object that ties data,
model, defense, attack, files, and scorers into a single executable unit.
"""

import logging
import warnings
import hashlib
from typing import List, Union, Literal, Any
from omegaconf import DictConfig, OmegaConf
import os
import yaml
import numpy as np
from pathlib import Path
from hydra.utils import instantiate

from ..data import DataConfig, DataPipelineConfig
from ..model import ModelConfig

try:
    from ..data import FairlearnDataConfig
except ImportError:  # pragma: no cover
    FairlearnDataConfig = None
from ..model.defend import DefensePipelineConfig
from ..attack import AttackConfig
from ..score import ScorerDictConfig
from ..file import FileConfig, data_files, model_files, attack_files
from ..utils import ConfigBase, coerce_config

try:
    from ..data import AnjanaDataConfig
except ImportError:  # pragma: no cover
    AnjanaDataConfig = None


try:
    from ..model import FairlearnModelConfig
except ImportError:  # pragma: no cover
    FairlearnModelConfig = None
try:
    from ..model import AnjanaModelConfig
except ImportError:  # pragma: no cover
    AnjanaModelConfig = None


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

        if hasattr(self.data, "_target_"):
            resolved = instantiate(self.data)
            if not isinstance(resolved, DataConfig):
                raise TypeError(
                    f"Resolved data target must be DataConfig-compatible, got {type(resolved)}",
                )
            return resolved

        data_dict = self._data_to_dict(self.data)
        data_cls = self._select_data_cls(data_dict)
        logger.info("Resolved data config class: %s", data_cls.__name__)
        data_obj = data_cls(**data_dict)
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

    def _coerce_scorer_config(self, scorer_obj: Any):
        if scorer_obj is None:
            return None
        if isinstance(scorer_obj, ScorerDictConfig):
            return scorer_obj
        # List of scorer specs → merge all into one ScorerDictConfig
        from omegaconf import ListConfig

        if isinstance(scorer_obj, (list, ListConfig)):
            return ScorerDictConfig.merge(list(scorer_obj))
        scorer_obj = coerce_config(scorer_obj)
        if isinstance(scorer_obj, str):
            scorer_obj = ScorerDictConfig.from_yaml(scorer_obj).to_dict()
        if isinstance(scorer_obj, dict):
            if "scorers" in scorer_obj:
                return ScorerDictConfig(**scorer_obj)
            return ScorerDictConfig(scorers=scorer_obj)
        raise ValueError(f"Unsupported scorer config type: {type(scorer_obj)}")

    def set_device(self, device: Union[str, int] = "cpu"):
        """
        Set the computation device for the experiment based on the selected library.
        For TensorFlow, configures GPU/CPU usage.
        Args:
            device (Union[str, int]): Device to use ("cpu", "gpu", or GPU index).
        """

        if self.library == "tensorflow":
            import tensorflow as tf

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

    def __post_init__(self):
        if not hasattr(self, "score_dict") or self.score_dict is None:
            self.score_dict = {}
        if not hasattr(self, "_target_") or self._target_ is None:
            self._target_ = "deckard.experiment.ExperimentConfig"
        # Set random seed
        self.set_random_seed()
        # Validate and initialize data config
        self.data = self._resolve_data_config()

        assert isinstance(
            self.data,
            DataConfig,
        ), f"data must be an instance of DataConfig. Got {type(self.data)}"
        self.data.__post_init__()
        if not hasattr(self, "classifier"):
            self.classifier = self.data.classifier
        else:
            assert (
                self.classifier == self.data.classifier
            ), f"classifier in experiment must match data.classifier. Got {self.classifier} vs {self.data.classifier}"
        if self.defense is not None:
            self.defense = DefensePipelineConfig.coerce(self.defense)
            assert isinstance(
                self.defense,
                DefensePipelineConfig,
            ), "defense must be an instance of DefensePipelineConfig"
            self.defense.__post_init__()
        if self.model is not None:
            if self.defense is not None:
                self.model.defense = self.defense
            if isinstance(self.model, ModelConfig):
                pass
            else:
                if hasattr(self.model, "_target_"):
                    self.model = instantiate(self.model)
                elif isinstance(self.model, DictConfig):
                    model_dict = OmegaConf.to_container(self.model)
                    self.model = ModelConfig(**model_dict)
                elif isinstance(self.model, str):
                    model_dict = ModelConfig.from_yaml(self.model).to_dict()
                    self.model = ModelConfig(**model_dict)
                elif isinstance(self.model, ConfigBase):
                    model_dict = self.model.to_dict()
                    self.model = ModelConfig(**model_dict)
                elif isinstance(self.model, dict):
                    model_dict = self.model
                    self.model = ModelConfig(**model_dict)
                else:
                    raise ValueError(
                        f"Unsupported type for model: {type(self.model)}",
                    )
            assert isinstance(
                self.model,
                ModelConfig,
            ), "model must be an instance of ModelConfig"

            self.model.__post_init__()
            if self.classifier is None:
                self.classifier = self.model.classifier
            else:
                assert (
                    self.classifier == self.model.classifier
                ), f"classifier in experiment must match model.classifier. Got {self.classifier} vs {self.model.classifier}"

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
            elif AnjanaDataConfig is not None and isinstance(
                self.data,
                AnjanaDataConfig,
            ):
                if AnjanaModelConfig is None:
                    raise ImportError(
                        "AnjanaModelConfig requires optional anjana dependencies. Install deckard[anjana] to enable anjana model configs.",
                    )
                if not isinstance(self.model, AnjanaModelConfig):
                    self.model = AnjanaModelConfig(
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
        if self.attack is not None:
            if isinstance(self.attack, AttackConfig):
                pass
            else:
                if isinstance(self.attack, DictConfig):
                    attack_dict = OmegaConf.to_container(self.attack)
                elif isinstance(self.attack, str):
                    attack_dict = AttackConfig.from_yaml(self.attack).to_dict()
                elif isinstance(self.attack, ConfigBase):
                    attack_dict = self.attack.to_dict()
                elif isinstance(self.attack, dict):
                    attack_dict = OmegaConf.to_container(
                        OmegaConf.create(self.attack),
                    )
                else:
                    raise ValueError(
                        f"Unsupported type for attack: {type(self.attack)}",
                    )
                if "_target_" not in attack_dict:
                    self.attack = AttackConfig(**attack_dict)
                else:
                    self.attack = instantiate(self.attack)
            assert isinstance(
                self.attack,
                AttackConfig,
            ), "attack must be an instance of AttackConfig"
            self.attack.__post_init__()
        # Set experiment name if not provided
        if self.experiment_name in [None, "", "{hash}", "*"]:
            config_list = [self.data]
            if self.model:
                config_list.append(self.model)
            if self.attack:
                config_list.append(self.attack)
            if self.score:
                config_list.append(self.score)
            self.experiment_name = self._hash_from_list(config_list)
        # Initialize FileConfig, ensuring experiment_name is set
        if self.files is None:
            self.files = FileConfig()
        elif isinstance(self.files, FileConfig):
            self.files.__post_init__()
        elif isinstance(self.files, ConfigBase):
            file_dict = self.files.to_dict()
            self.files = FileConfig(**file_dict)
        elif isinstance(self.files, DictConfig):
            file_dict = OmegaConf.to_container(self.files)
            self.files = FileConfig(**file_dict)
        elif isinstance(self.files, str):
            file_dict = FileConfig.from_yaml(self.files).to_dict()
            self.files = FileConfig(**file_dict)
        elif isinstance(self.files, dict):
            file_dict = self.files
            self.files = FileConfig(**file_dict)
        else:
            raise ValueError(f"Unsupported type for files: {type(self.files)}")
        assert isinstance(
            self.files,
            FileConfig,
        ), "file must be an instance of FileConfig"
        self.files.__post_init__()

        # Set scorers
        self.data_scorer = None
        self.model_scorer = None
        self.experiment_scorer = None
        if self.score is not None:
            score_cfg = self.score
            if isinstance(score_cfg, DictConfig):
                score_cfg = OmegaConf.to_container(score_cfg, resolve=True)
            if isinstance(score_cfg, dict) and any(
                key in score_cfg for key in ["data", "model", "experiment"]
            ):
                self.data_scorer = self._coerce_scorer_config(
                    score_cfg.get("data"),
                )
                self.model_scorer = self._coerce_scorer_config(
                    score_cfg.get("model"),
                )
                self.experiment_scorer = self._coerce_scorer_config(
                    score_cfg.get("experiment"),
                )
            else:
                # Backward-compatible shorthand: a single score config targets model scoring.
                self.model_scorer = self._coerce_scorer_config(score_cfg)

        # Attach component scorers so DataConfig/ModelConfig execute runtime-configured scoring.
        if self.data_scorer is not None:
            self.data.scorer = self.data_scorer
        if self.model is not None and self.model_scorer is not None:
            self.model.scorer = self.model_scorer

        # Keep `score` as experiment-level scorer only.
        self.score = self.experiment_scorer

        # Reconcile and enforce a single device across experiment/data/model.
        self._reconcile_component_devices()
        if self.library not in ["sklearn"]:
            self.set_device(self.device if self.device is not None else "cpu")

    def set_random_seed(self):
        if self.library in ["sklearn"]:
            np.random.seed(self.random_state)
        elif self.library in ["tensorflow"]:
            import tensorflow as tf

            tf.random.set_seed(self.random_state)
        elif self.library in ["pytorch"]:
            import torch

            torch.manual_seed(self.random_state)
        else:
            raise ValueError(f"Unsupported library: {self.library}")

    def _hash_from_list(self, config_list: List[ConfigBase]) -> str:
        """
        Generate a hash string from a list of ConfigBase objects.
        The hash is generated by concatenating the string representations of the configurations
        and computing the MD5 hash of the resulting string.
        Args:
            config_list (List[ConfigBase]): List of ConfigBase objects to generate the hash from.
        Returns:
            str: The generated hash string.
        """
        for conf in config_list:
            assert isinstance(
                conf,
                ConfigBase,
            ), "All items in config_list must be instances of ConfigBase"
            to_string = "".join(
                [
                    str(getattr(conf, attr))
                    for attr in dir(conf)
                    if not attr.startswith("_") and not callable(getattr(conf, attr))
                ],
            )
        return hashlib.md5(to_string.encode()).hexdigest()

    def _detect_n_repeats(self) -> tuple[int, str]:
        """Return repeated-evaluation count and key suffix for sampler-driven runs.

        Returns
        -------
        tuple[int, str]
            ``(n_splits, "fold")`` for ``KFoldSampler``,
            ``(n_splits, "split")`` for ``ShuffleSampler``,
            and ``(1, "fold")`` for all other samplers.
        """
        from ..data.sample import KFoldSampler, ShuffleSampler

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

        if self.attack:
            try:
                self.attack(
                    data=self.data,
                    model=self.model,
                    **attack_file_outputs,
                )
                assert hasattr(
                    self.attack,
                    "attack",
                ), "attack must have attack attribute after training"
                assert hasattr(
                    self.attack,
                    "attack_predictions",
                ), "attack must have a predictions attribute after training"
                assert hasattr(
                    self.attack,
                    "score_dict",
                ), "attack must have score_dict attribute after training"
                scores.update(**self.attack.score_dict)
            except ValueError as e:
                logger.debug(e)
                raise
        else:
            logger.info("No attack config provided, skipping attack.")

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
    ):
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

        if "score_file" in file_dict and not Path(file_dict["score_file"]).exists():
            self.save_scores(scores, file_dict["score_file"])
        elif "score_file" in file_dict:
            old_scores = self.load_scores(file_dict["score_file"])
            new_scores = {**old_scores, **scores}
            self.save_scores(new_scores, file_dict["score_file"])
        else:
            logger.info("No score_file specified, skipping score saving.")
        return scores
