import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf

from ..model import ModelConfig
from ..score.base import (
    DefaultModelScoreConfig,
    ScorerConfig,
    ScorerDictConfig,
    _TaskAwareScorerMixin,
    coerce_scorer_config,
)
from ..utils import ConfigBase, coerce_config, resolve_class

if TYPE_CHECKING:
    from ..attack import AttackConfig
    from ..data import DataConfig


@dataclass(eq=False)
class DetectorScorerConfig(_TaskAwareScorerMixin, ScorerDictConfig):
    """Task-aware scorer config for detector outputs."""

    classifier: Union[bool, str, None] = None
    scorers: dict[str, Union[ScorerConfig, dict[str, Any]]] = field(
        default_factory=dict,
    )

    def _build_default_scorers(
        self,
        classifier: bool,
    ) -> dict[str, Union[ScorerConfig, dict[str, Any]]]:
        shared_defaults = DefaultModelScoreConfig(classifier=classifier).scorers
        if classifier:
            # Detector scoring uses class-label outputs; keep the label metrics subset.
            keys = ("accuracy", "precision", "recall", "f1")
            return {k: shared_defaults[k] for k in keys if k in shared_defaults}
        keys = ("mse", "mae", "r2")
        return {k: shared_defaults[k] for k in keys if k in shared_defaults}

    def __post_init__(self):
        self._initialize_task_aware_scorers(default=True)
        super().__post_init__()


@dataclass(eq=False)
class DetectorConfig(ConfigBase):
    """Auxiliary detector runtime for adversarial-vs-clean detection tasks."""

    detector_type: str = "art.defences.detector.evasion.BinaryInputDetector"
    detector_params: dict[str, Any] = field(default_factory=dict)
    fit_params: dict[str, Any] = field(default_factory=dict)
    detector_model: Union[ModelConfig, dict, str, None] = None
    scorer: Union[
        DetectorScorerConfig,
        ScorerDictConfig,
        dict[str, Any],
        None,
    ] = None
    alias: str = field(default_factory=str)

    detector: Any = None
    score_dict: dict[str, float | int] = field(default_factory=dict)
    detector_training_time: Union[float, None] = None
    detector_detection_time: Union[float, None] = None
    _target_: Union[str, None] = None

    def __post_init__(self):
        if self._target_ is None:
            self._target_ = "deckard.detector.DetectorConfig"
        self.detector_params = self.detector_params or {}
        self.fit_params = self.fit_params or {}
        self.score_dict = self.score_dict or {}
        self.detector_model = self._coerce_detector_model(self.detector_model)
        self.scorer = self._coerce_scorer(self.scorer)

    def _coerce_detector_model(
        self,
        value: Union[ModelConfig, dict[str, Any], str, None],
    ) -> Union[ModelConfig, None]:
        if value is None:
            return None
        if isinstance(value, ModelConfig):
            return value
        if isinstance(value, DictConfig):
            raw_value = OmegaConf.to_container(value, resolve=True)
            if not isinstance(raw_value, dict):
                raise TypeError(
                    f"detector_model DictConfig must resolve to a dictionary, got {type(raw_value)}",
                )
            value = dict(raw_value)
        if isinstance(value, str):
            value = ModelConfig.from_yaml(value).to_dict()
        else:
            value = coerce_config(value)
        if isinstance(value, dict):
            return ModelConfig(**value)
        raise TypeError(f"Unsupported detector_model type: {type(value)}")

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _coerce_scorer(
        value: Union[DetectorScorerConfig, ScorerDictConfig, dict[str, Any], None],
    ) -> ScorerDictConfig:
        scorer = coerce_scorer_config(
            value,
            default_factory=lambda: DetectorScorerConfig(classifier=True),
        )
        if scorer is None:
            return DetectorScorerConfig(classifier=True)
        if isinstance(scorer, ScorerDictConfig):
            return scorer
        raise TypeError(f"Unsupported detector scorer type: {type(value)}")

    def _build_detector_dataset(
        self,
        data: "DataConfig",
        attack: "AttackConfig",
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if attack is None:
            raise ValueError(
                "DetectorConfig requires an attack object to build labels.",
            )
        x_adv = getattr(attack, "attack_predictions", None)
        if x_adv is None:
            raise ValueError(
                "DetectorConfig requires attack.attack_predictions (adversarial samples).",
            )

        split = str(self.fit_params.get("split", "test")).lower()
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported detector split: {split}")

        x_clean = getattr(data, "X_train" if split == "train" else "X_test")
        x_clean = self._to_numpy(x_clean)
        x_adv = self._to_numpy(x_adv)

        n = int(min(len(x_clean), len(x_adv)))
        if n <= 0:
            raise ValueError(
                "Detector dataset must contain at least one clean and one adversarial sample.",
            )

        x_clean = x_clean[:n]
        x_adv = x_adv[:n]
        y_clean = np.zeros(n, dtype=np.int64)
        y_adv = np.ones(n, dtype=np.int64)

        x = np.concatenate([x_clean, x_adv], axis=0)
        y = np.concatenate([y_clean, y_adv], axis=0)
        return x, y, n

    def _build_detector_backend(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Any:
        if self.detector_model is None:
            raise ValueError(
                "DetectorConfig requires detector_model to train auxiliary detector.",
            )

        detector_model_cfg = self.detector_model
        assert isinstance(detector_model_cfg, ModelConfig)
        detector_model_cfg.__post_init__()
        detector_model_cfg._train(x_train, y_train)

        data_stub = SimpleNamespace(
            X_train=x_train,
            y_train=y_train,
            X_test=x_train,
            y_test=y_train,
        )
        return detector_model_cfg.get_art_model(data_stub)

    def __call__(
        self,
        data: "DataConfig",
        model: ModelConfig | None = None,
        attack: "AttackConfig | None" = None,
    ) -> dict[str, float | int]:
        _ = model
        x, y, n = self._build_detector_dataset(data=data, attack=attack)

        backend = self._build_detector_backend(x_train=x, y_train=y)
        detector_cls = resolve_class(self.detector_type)
        fit_kwargs = {k: v for k, v in self.fit_params.items() if k != "split"}
        # Detector constructors differ between evasion and poisoning detectors.
        try:
            detector = detector_cls(detector=backend, **self.detector_params)
        except TypeError:
            try:
                detector = detector_cls(
                    classifier=backend,
                    x_train=x,
                    y_train=y,
                    **self.detector_params,
                )
            except TypeError:
                detector = detector_cls(backend, **self.detector_params)

        y_pred = None
        if hasattr(detector, "fit") and callable(getattr(detector, "fit")):
            start = time.process_time()
            detector.fit(x, y, **fit_kwargs)
            self.detector_training_time = time.process_time() - start

        start = time.process_time()
        if hasattr(detector, "detect") and callable(getattr(detector, "detect")):
            _, is_adversarial = detector.detect(
                x,
                batch_size=int(fit_kwargs.get("batch_size", 128)),
            )
            y_pred = self._to_numpy(is_adversarial).reshape(-1).astype(int)
        elif hasattr(detector, "detect_poison") and callable(
            getattr(detector, "detect_poison"),
        ):
            _, is_clean = detector.detect_poison(**fit_kwargs)
            is_clean_arr = np.asarray(is_clean)
            if is_clean_arr.ndim == 1 and len(is_clean_arr) == len(y):
                clean_mask = is_clean_arr.astype(int)
            elif is_clean_arr.ndim == 1 and len(is_clean_arr) < len(y):
                # Some ART poison detectors can return suspected-poison indices.
                clean_mask = np.ones(len(y), dtype=int)
                poison_idx = is_clean_arr.astype(int)
                poison_idx = poison_idx[(poison_idx >= 0) & (poison_idx < len(y))]
                clean_mask[poison_idx] = 0
            else:
                raise ValueError(
                    "Unsupported detect_poison output shape "
                    f"for detector {self.detector_type}: {is_clean_arr.shape}",
                )
            y_pred = 1 - clean_mask.reshape(-1)
        else:
            raise AttributeError(
                f"Detector {self.detector_type} exposes neither detect() nor detect_poison().",
            )
        self.detector_detection_time = time.process_time() - start

        if y_pred is None:
            raise RuntimeError("Detector prediction output was not produced.")
        y_true = y.reshape(-1).astype(int)

        self.detector = detector
        metric_scores = self.scorer(
            mode=None,
            y_true=y_true,
            y_pred=y_pred,
            data=data,
            model=model,
            attack=attack,
        )
        prefixed_scores = {}
        for key, value in metric_scores.items():
            score_key = key if str(key).startswith("detector_") else f"detector_{key}"
            prefixed_scores[score_key] = float(value)

        self.score_dict = {
            **self.score_dict,
            **prefixed_scores,
            "detector_n": int(len(y_true)),
            "detector_clean_n": int(n),
            "detector_adversarial_n": int(n),
            "detector_training_time": float(self.detector_training_time or 0.0),
            "detector_detection_time": float(self.detector_detection_time),
        }
        return self.score_dict
