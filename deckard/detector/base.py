import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Protocol, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf

from ..artifacts import ScoreDict
from ..model import ModelConfig
from ..score.base import (
    DefaultModelScorerDictConfig,
    ScorerConfig,
    ScorerDictConfig,
    TaskAwareScorerMixin,
    coerce_scorer_config,
)
from ..utils import (
    BaseConfig,
    coerce_config,
    instantiate_plugin_spec,
    load_class,
    normalize_plugin_specs,
    resolve_class,
)
from .canon import ensure_detector_runtime_contract, normalize_detector_stage

if TYPE_CHECKING:
    from ..attack import AttackConfig
    from ..data import DataConfig


DetectorFitParamValue = str | int | float | bool | None
DetectorFileValue = str | None


class DetectorRuntimeLike(Protocol):
    """Structural protocol for detector runtime objects used in DetectorConfig."""

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs: DetectorFitParamValue,
    ) -> Any:
        """Fit detector state from runtime arrays.

        Args:
            x: Runtime feature matrix.
            y: Runtime labels.
            **kwargs: Detector fit kwargs.

        Returns:
            Detector fit output.
        """
        ...

    def detect(
        self,
        x: np.ndarray,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict adversarial flags for runtime arrays.

        Args:
            x: Runtime feature matrix.
            batch_size: Inference batch size.

        Returns:
            Tuple containing detector scores and adversarial mask.
        """
        ...

    def detect_poison(
        self, **kwargs: DetectorFitParamValue
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict poison flags for runtime arrays.

        Args:
            **kwargs: Poison detection kwargs.

        Returns:
            Tuple containing detector scores and clean/poison mask.
        """
        ...


@dataclass(eq=False, kw_only=True)
class DetectorScorerConfig(TaskAwareScorerMixin, ScorerDictConfig):
    """Task-aware scorer config for detector outputs."""

    classifier: Union[bool, str] = True
    scorers: dict[str, Union[ScorerConfig, dict[str, Any]]] = field(
        default_factory=dict,
    )

    def _build_default_scorers(
        self,
        classifier: bool,
    ) -> dict[str, Union[ScorerConfig, dict[str, Any]]]:
        shared_defaults = DefaultModelScorerDictConfig(classifier=classifier).scorers
        if classifier:
            # Detector scoring uses class-label outputs; keep the label metrics subset.
            keys = ("accuracy", "precision", "recall", "f1")
            return {k: shared_defaults[k] for k in keys if k in shared_defaults}
        keys = ("mse", "mae", "r2")
        return {k: shared_defaults[k] for k in keys if k in shared_defaults}

    def __post_init__(self):
        self._initialize_task_aware_scorers(default=True)
        super().__post_init__()


@dataclass(eq=False, kw_only=True)
class DetectorConfig(BaseConfig):
    """Auxiliary detector runtime for adversarial-vs-clean detection tasks."""

    detector_type: str = "art.defences.detector.evasion.BinaryInputDetector"
    detector_params: dict[str, Any] = field(default_factory=dict)
    fit_params: dict[str, Any] = field(default_factory=dict)
    mode: str = "train"
    filter_mode: str = "auto"
    detector_model: Union[ModelConfig, dict, str, None] = None
    scorer: Union[
        DetectorScorerConfig,
        ScorerDictConfig,
        dict[str, Any],
        None,
    ] = None
    alias: str = field(default_factory=str)
    plugins: list = field(default_factory=list)

    detector: Any = None
    detector_predictions: Any = None
    score_dict: ScoreDict = field(default_factory=ScoreDict)
    detector_training_time: Union[float, None] = None
    detector_detection_time: Union[float, None] = None
    _target_: Union[str, None] = None
    _plugin_objects: Union[list, None] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self._initialize_target_reference()
        self._initialize_runtime_defaults()
        self._initialize_detector_model_config()
        self._initialize_detector_scorer()
        ensure_detector_runtime_contract(self)

    def _initialize_target_reference(self) -> None:
        """Set canonical runtime target path."""
        if self._target_ is None:
            self._target_ = "deckard.detector.DetectorConfig"

    def _initialize_runtime_defaults(self) -> None:
        """Normalize mutable runtime defaults."""
        self.detector_params = self.detector_params or {}
        self.fit_params = self.fit_params or {}
        self.mode = str(self.mode or "train").strip().lower()
        if self.mode not in {"train", "filter"}:
            raise ValueError(f"Unsupported detector mode: {self.mode}")
        self.filter_mode = str(self.filter_mode or "auto").strip().lower()
        if self.filter_mode not in {"auto", "poison", "evasion"}:
            raise ValueError(f"Unsupported detector filter_mode: {self.filter_mode}")
        self.score_dict = ScoreDict.from_payload(self.score_dict or {})

    def _initialize_detector_model_config(self) -> None:
        """Coerce detector-model config into a runtime ModelConfig object."""
        self.detector_model = self._coerce_detector_model(self.detector_model)

    def _initialize_detector_scorer(self) -> None:
        """Coerce detector scorer into a supported scorer config."""
        self.scorer = self._coerce_scorer(self.scorer)

    def _instantiate_plugin(self, plugin_spec: Any):
        return instantiate_plugin_spec(plugin_spec, loader=load_class)

    def _get_plugins(self) -> list:
        if self._plugin_objects is None:
            plugin_specs = normalize_plugin_specs(self.plugins)
            self._plugin_objects = [
                self._instantiate_plugin(spec) for spec in plugin_specs
            ]
        return self._plugin_objects

    def _run_plugin_hook(self, hook_name: str, **kwargs) -> list[Any]:
        outputs = []
        for plugin in self._get_plugins():
            hook = getattr(plugin, hook_name, None)
            if callable(hook):
                outputs.append(hook(self, **kwargs))
        return outputs

    def _merge_plugin_scores(self, hook_outputs: list[Any]) -> None:
        if self.score_dict is None:
            self.score_dict = {}
        for output in hook_outputs:
            if isinstance(output, dict):
                self.score_dict.update(output)

    def _run_detector_stage_hooks(self, when: str, stage: str, **kwargs: Any) -> None:
        event = str(when).strip().lower()
        if event not in {"before", "after"}:
            raise ValueError(f"Invalid detector hook event: {when}")
        canonical_stage = normalize_detector_stage(stage)
        outputs = self._run_plugin_hook(
            f"{event}_detector_stage",
            stage=canonical_stage,
            **kwargs,
        )
        self._merge_plugin_scores(outputs)

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

    def _resolve_runtime_data(self, data: "DataConfig") -> "DataConfig":
        if data is None:
            raise ValueError("DetectorConfig requires data to execute.")
        has_runtime = hasattr(data, "X_train") and hasattr(data, "X_test")
        if has_runtime:
            return data
        if callable(data):
            data()
            if hasattr(data, "X_train") and hasattr(data, "X_test"):
                return data
        raise ValueError("DetectorConfig could not resolve runtime data arrays.")

    def _resolve_runtime_model(
        self,
        model: ModelConfig | None,
        data: "DataConfig",
    ) -> ModelConfig | None:
        if model is None:
            return None
        has_estimator = getattr(model, "_model", None) is not None
        if has_estimator:
            return model
        if callable(model):
            model(data)
        return model

    def _resolve_runtime_attack(
        self,
        attack: "AttackConfig | None",
        data: "DataConfig",
        model: ModelConfig | None,
    ) -> "AttackConfig | None":
        if attack is None:
            return None
        if getattr(attack, "attack_predictions", None) is not None:
            return attack
        if callable(attack):
            attack(data=data, model=model)
        return attack

    @staticmethod
    def _to_label_vector(values: Any) -> np.ndarray:
        arr = np.asarray(values)
        if arr.ndim == 1:
            return arr.reshape(-1)
        if arr.ndim >= 2 and arr.shape[1] > 1:
            return np.argmax(arr, axis=1)
        return arr.reshape(-1)

    def _resolve_filter_family(self, attack: "AttackConfig | None") -> str:
        if self.filter_mode in {"poison", "evasion"}:
            return self.filter_mode
        attack_type = str(getattr(attack, "attack_type", "") or "").lower()
        if "poison" in attack_type:
            return "poison"
        return "evasion"

    def _apply_poison_filter(
        self,
        *,
        data: "DataConfig",
        model: ModelConfig | None,
        attack: "AttackConfig | None",
        x_clean: np.ndarray,
        x_adv: np.ndarray,
        y_pred: np.ndarray,
        n: int,
    ) -> float:
        poison_mask = y_pred[n:].reshape(-1).astype(int) == 1
        filtered_adv = np.array(x_adv, copy=True)
        if len(filtered_adv) and len(x_clean):
            filtered_adv[poison_mask] = x_clean[poison_mask]

        if attack is not None:
            attack.attack_predictions = filtered_adv

        split = str(self.fit_params.get("split", "test")).lower()
        x_attr = "X_train" if split == "train" else "X_test"
        y_attr = "y_train" if split == "train" else "y_test"
        setattr(data, x_attr, filtered_adv)

        if model is not None and hasattr(data, y_attr):
            y_values = self._to_numpy(getattr(data, y_attr))[: len(filtered_adv)]
            if hasattr(model, "train") and callable(getattr(model, "train")):
                model.train(filtered_adv, y_values)

        return float(np.mean(poison_mask)) if len(poison_mask) else 0.0

    def _apply_evasion_filter(
        self,
        *,
        data: "DataConfig",
        model: ModelConfig | None,
        attack: "AttackConfig | None",
        x_clean: np.ndarray,
        x_adv: np.ndarray,
        y_pred: np.ndarray,
        n: int,
    ) -> float:
        evasion_mask = y_pred[n:].reshape(-1).astype(int) == 1
        split = str(self.fit_params.get("split", "test")).lower()
        y_attr = "y_train" if split == "train" else "y_test"
        y_clean = self._to_label_vector(self._to_numpy(getattr(data, y_attr)))[:n]

        if attack is not None:
            filtered_inputs = np.array(x_adv, copy=True)
            if len(filtered_inputs) and len(x_clean):
                filtered_inputs[evasion_mask] = x_clean[evasion_mask]
            setattr(attack, "filtered_attack_inputs", filtered_inputs)

            attack_preds = getattr(attack, "attack_predictions", None)
            if attack_preds is not None:
                adv_labels = self._to_label_vector(self._to_numpy(attack_preds))[:n]
                filtered_labels = np.array(adv_labels, copy=True)
                filtered_labels[evasion_mask] = y_clean[evasion_mask]
                attack.attacked_labels = filtered_labels
                setattr(attack, "filtered_attack_labels", filtered_labels)

        return float(np.mean(evasion_mask)) if len(evasion_mask) else 0.0

    def _apply_filtering_behavior(
        self,
        *,
        data: "DataConfig",
        model: ModelConfig | None,
        attack: "AttackConfig | None",
        x: np.ndarray,
        y_pred: np.ndarray,
        n: int,
    ) -> tuple[float, float]:
        x_clean = x[:n]
        x_adv = x[n : 2 * n]
        family = self._resolve_filter_family(attack)
        if family == "poison":
            poison_success = self._apply_poison_filter(
                data=data,
                model=model,
                attack=attack,
                x_clean=x_clean,
                x_adv=x_adv,
                y_pred=y_pred,
                n=n,
            )
            return poison_success, 0.0
        evasion_success = self._apply_evasion_filter(
            data=data,
            model=model,
            attack=attack,
            x_clean=x_clean,
            x_adv=x_adv,
            y_pred=y_pred,
            n=n,
        )
        return 0.0, evasion_success

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
        detector_model_cfg.train(x_train, y_train)

        data_stub = SimpleNamespace(
            X_train=x_train,
            y_train=y_train,
            X_test=x_train,
            y_test=y_train,
        )
        return detector_model_cfg.get_art_model(data_stub)

    def compose_detector_dataset_behavior(
        self,
        data: "DataConfig",
        attack: "AttackConfig",
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Compose detector dataset behavior from clean/adversarial samples.

        Args:
            data: Runtime data configuration.
            attack: Runtime attack configuration.

        Returns:
            Tuple containing features, labels, and clean-sample count.
        """
        return self._build_detector_dataset(data=data, attack=attack)

    def compose_detector_backend_behavior(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> DetectorRuntimeLike:
        """Compose backend detector-estimator behavior from configured model.

        Args:
            x_train: Detector training features.
            y_train: Detector training labels.

        Returns:
            Trained backend estimator for detector wrapping.
        """
        return self._build_detector_backend(x_train=x_train, y_train=y_train)

    def compose_detector_runtime_behavior(
        self,
        backend: DetectorRuntimeLike,
        x: np.ndarray,
        y: np.ndarray,
        fit_kwargs: dict[str, DetectorFitParamValue],
    ) -> DetectorRuntimeLike:
        """Compose concrete detector runtime object for evasion/poison APIs.

        Args:
            backend: Trained backend estimator.
            x: Detector training features.
            y: Detector training labels.
            fit_kwargs: Runtime detector-fit kwargs.

        Returns:
            Instantiated detector runtime object.
        """
        detector_cls = resolve_class(self.detector_type)
        # Detector constructors differ between evasion and poisoning detectors.
        try:
            return detector_cls(detector=backend, **self.detector_params)
        except TypeError:
            try:
                return detector_cls(
                    classifier=backend,
                    x_train=x,
                    y_train=y,
                    **self.detector_params,
                )
            except TypeError:
                return detector_cls(backend, **self.detector_params)

    def execute_detector_behavior(
        self,
        detector: DetectorRuntimeLike,
        x: np.ndarray,
        y: np.ndarray,
        fit_kwargs: dict[str, DetectorFitParamValue],
    ) -> np.ndarray:
        """Execute detector fit/predict behavior and return detector labels.

        Args:
            detector: Detector runtime object.
            x: Detector input features.
            y: Detector binary labels.
            fit_kwargs: Runtime detector-fit kwargs.

        Returns:
            Detector prediction labels.

        Raises:
            ValueError: If poison-detection output shape is unsupported.
            AttributeError: If detector runtime does not expose detection methods.
            RuntimeError: If detector prediction output is not produced.
        """
        self._run_detector_stage_hooks(
            "before",
            "pre-fit",
            detector=detector,
            x=x,
            y=y,
            fit_kwargs=fit_kwargs,
        )
        if hasattr(detector, "fit") and callable(getattr(detector, "fit")):
            start = time.process_time()
            detector.fit(x, y, **fit_kwargs)
            self.detector_training_time = time.process_time() - start
        self._run_detector_stage_hooks(
            "after",
            "post-fit",
            detector=detector,
            x=x,
            y=y,
            fit_kwargs=fit_kwargs,
        )

        self._run_detector_stage_hooks(
            "before",
            "pre-detect",
            detector=detector,
            x=x,
            y=y,
            fit_kwargs=fit_kwargs,
        )
        start = time.process_time()
        y_pred = None
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
        self._run_detector_stage_hooks(
            "after",
            "post-detect",
            detector=detector,
            x=x,
            y=y,
            fit_kwargs=fit_kwargs,
            y_pred=y_pred,
        )

        if y_pred is None:
            raise RuntimeError("Detector prediction output was not produced.")
        return y_pred

    def filter(
        self,
        data: "DataConfig",
        model: ModelConfig | None = None,
        attack: "AttackConfig | None" = None,
        files: dict[str, DetectorFileValue] | None = None,
        detector_file: str | None = None,
        detected_predictions_file: str | None = None,
        score_file: str | None = None,
    ) -> dict[str, float | int]:
        """Execute detector runtime specifically in filter mode.

        This public method applies detector filtering behavior to attack/runtime
        payloads and returns the resulting detector score dictionary.

        Args:
            data: Runtime data configuration.
            model: Optional runtime model configuration.
            attack: Optional runtime attack configuration.
            files: Optional runtime artifact file mapping.
            detector_file: Optional detector artifact path.
            detected_predictions_file: Optional detected-predictions artifact path.
            score_file: Optional detector score artifact path.

        Returns:
            Detector score payload from a filter-mode execution.
        """
        previous_mode = self.mode
        self.mode = "filter"
        try:
            return self(
                data=data,
                model=model,
                attack=attack,
                files=files,
                detector_file=detector_file,
                detected_predictions_file=detected_predictions_file,
                score_file=score_file,
            )
        finally:
            self.mode = previous_mode

    def __call__(
        self,
        data: "DataConfig",
        model: ModelConfig | None = None,
        attack: "AttackConfig | None" = None,
        files: dict[str, DetectorFileValue] | None = None,
        detector_file: str | None = None,
        detected_predictions_file: str | None = None,
        score_file: str | None = None,
    ) -> dict[str, float | int]:
        """Execute detector runtime lifecycle and return detection score payload.

        Args:
            data: Runtime data configuration.
            model: Optional runtime model configuration.
            attack: Optional runtime attack configuration.
            files: Optional runtime artifact file mapping.
            detector_file: Optional detector artifact path.
            detected_predictions_file: Optional detected-predictions artifact path.
            score_file: Optional detector score artifact path.

        Returns:
            Detector score payload.
        """
        files = dict(files or {})
        if detector_file is None:
            detector_file = files.get(
                "detector_file", files.get("detector_model_file")
            )
        if detected_predictions_file is None:
            detected_predictions_file = files.get("detected_predictions_file")
        if score_file is None:
            score_file = files.get("score_file")

        ensure_detector_runtime_contract(self)
        data = self._resolve_runtime_data(data)
        model = self._resolve_runtime_model(model=model, data=data)
        attack = self._resolve_runtime_attack(attack=attack, data=data, model=model)

        x, y, n = self.compose_detector_dataset_behavior(data=data, attack=attack)

        backend = self.compose_detector_backend_behavior(x_train=x, y_train=y)
        fit_kwargs = {k: v for k, v in self.fit_params.items() if k != "split"}
        detector = self.compose_detector_runtime_behavior(
            backend=backend,
            x=x,
            y=y,
            fit_kwargs=fit_kwargs,
        )
        y_pred = self.execute_detector_behavior(
            detector=detector,
            x=x,
            y=y,
            fit_kwargs=fit_kwargs,
        )
        y_true = y.reshape(-1).astype(int)
        self.detector_predictions = y_pred

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

        poison_filter_success = 0.0
        evasion_filter_success = 0.0
        if self.mode == "filter":
            poison_filter_success, evasion_filter_success = (
                self._apply_filtering_behavior(
                    data=data,
                    model=model,
                    attack=attack,
                    x=x,
                    y_pred=y_pred,
                    n=n,
                )
            )

        self.score_dict = {
            **self.score_dict,
            **prefixed_scores,
            "detector_n": int(len(y_true)),
            "detector_clean_n": int(n),
            "detector_adversarial_n": int(n),
            "detector_training_time": float(self.detector_training_time or 0.0),
            "detector_detection_time": float(self.detector_detection_time),
            "detector_stage": normalize_detector_stage("post-detect"),
            "detector_execution_order": "post-attack",
            "poison_filter_success": float(poison_filter_success),
            "evasion_filter_success": float(evasion_filter_success),
        }
        self.merge_runtime_files(
            {
                "detector_model_file": detector_file,
                "detected_predictions_file": detected_predictions_file,
                "score_file": score_file,
            },
        )

        if detector_file is not None:
            self.save_object(self.detector, detector_file)
        if detected_predictions_file is not None:
            self.save_data(self.detector_predictions, detected_predictions_file)
        self.score_dict = self.merge_and_persist_scores(self.score_dict, score_file)
        return self.score_dict
