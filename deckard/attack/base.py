# Standard library imports
import copy
import pickle
import logging

from pathlib import Path
import pandas as pd

# Typing imports
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Union

# Sklearn and numpy imports
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import numpy as np

# ART imports
from art.config import ART_NUMPY_DTYPE

from omegaconf import DictConfig, OmegaConf

from ..artifacts import ScoreDict
from ..data import DataConfig
from ..model import ModelConfig
from ..frameworks.types import ArrayLike, AttackLike, EstimatorLike, MatrixLike
from ..model.defense.base import _get_art_symbols
from ..score.base import (
    DefaultClassifierScorerDictConfig,
    ScorerDictConfig,
)
from ..utils import (
    BaseConfig,
    is_default_config_value,
    is_null_config_value,
    load_class,
    normalize_plugin_specs,
    instantiate_plugin_spec,
    resolve_class,
    resolve_torch_device,
)
from ..frameworks.pytorch.torch_utils import (
    build_torch_art_model,
    collect_subset_from_dataloader,
    is_dataloader,
    is_tensor,
    is_torch_model,
    tensor_to_numpy,
)
from .canon import (
    ensure_attack_runtime_contract,
    normalize_attack_mode,
    normalize_attack_stage,
)

from ..score.attack import AttackScorerConfig

logger = logging.getLogger(__name__)


def _sensitive_slice(sensitive, n):
    """Return the first *n* rows of *sensitive*, or None if unavailable."""
    if sensitive is None:
        return None
    arr = np.asarray(sensitive)
    return arr[:n]


class SensitiveFeaturesWrapper(BaseEstimator):
    """Wraps an estimator that requires `sensitive_features` in predict.

    At predict time the stored sensitive features are sliced to match the
    number of rows in ``X``, so adversarial examples (same n rows, different
    feature values) continue to work correctly.
    """

    def __init__(self, estimator, sensitive_features):
        self.estimator = estimator
        self._sensitive = np.asarray(sensitive_features)

    def fit(self, X: Any, y: Any, **kwargs: Any) -> Any:
        """Fit wrapped estimator.

        Args:
            X: Training features.
            y: Training labels.
            **kwargs: Extra fit kwargs forwarded to wrapped estimator.

        Returns:
            Fitted wrapped estimator.
        """
        return self.estimator.fit(X, y, **kwargs)

    def predict(self, X: Any) -> Any:
        """Predict labels with stored sensitive features.

        Args:
            X: Feature matrix.

        Returns:
            Predicted labels.
        """
        n = len(X)
        sf = self._sensitive[:n]
        return self.estimator.predict(X, sensitive_features=sf)

    def predict_proba(self, X: Any) -> Any:
        """Predict class probabilities with stored sensitive features.

        Args:
            X: Feature matrix.

        Returns:
            Probability matrix (or synthetic one-hot fallback).
        """
        n = len(X)
        sf = self._sensitive[:n]
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X, sensitive_features=sf)
        # Fall back: convert hard labels to a two-column probability matrix
        labels = self.estimator.predict(X, sensitive_features=sf)
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        n_classes = max(len(unique_labels), 2)
        proba = np.zeros((len(labels), n_classes), dtype=float)
        for i, label in enumerate(labels):
            idx = int(label) if label < n_classes else n_classes - 1
            proba[i, idx] = 1.0
        return proba

    def get_params(self, deep: bool = True) -> dict:
        """Return sklearn-compatible params for the wrapper.

        Args:
            deep: Included for sklearn API compatibility.

        Returns:
            Parameter mapping for estimator and sensitive features.
        """
        return {
            "estimator": self.estimator,
            "sensitive_features": self._sensitive,
        }

    def set_params(self, **params: Any) -> "SensitiveFeaturesWrapper":
        """Set sklearn-compatible wrapper parameters.

        Args:
            **params: Parameters to set on wrapper state.

        Returns:
            This wrapper instance.
        """
        if "estimator" in params:
            self.estimator = params["estimator"]
        if "sensitive_features" in params:
            self._sensitive = np.asarray(params["sensitive_features"])
        return self

    def _sensitive_slice(self, sensitive, n):
        """Return the first *n* rows of *sensitive*, or None if unavailable."""
        if sensitive is None:
            return None
        arr = np.asarray(sensitive)
        return arr[:n]


supported_attacks = [
    "blackbox_membership_inference",
    "blackbox_evasion",
    "whitebox_evasion",
    "blackbox_attribute_inference",
    "whitebox_attribute_inference",
]


@dataclass(eq=True)
class AttackMixin:
    """Base callable attack handler used by runtime attack context resolution.

    The ``runtime`` attribute is the active ``AttackConfig`` instance owned by
    ``AttackConfig.__call__``. Mixins delegate mutable runtime state such as
    timers, predictions, and ``score_dict`` to that object.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    runtime: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runtime, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "runtime":
            object.__setattr__(self, name, value)
            return
        runtime = object.__getattribute__(self, "runtime")
        if runtime is None:
            object.__setattr__(self, name, value)
            return
        setattr(runtime, name, value)

    def __call__(
        self,
        *,
        data: DataConfig,
        model: ModelConfig | EstimatorLike | BaseEstimator,
        art_model: EstimatorLike,
        attack: AttackLike,
        attack_type: str,
        attack_subtype: str,
    ) -> ScoreDict:
        """Execute one attack handler.

        Args:
            data: Data runtime containing train/test/val splits.
            model: User model object or model config supplied to ``AttackConfig``.
            art_model: ART estimator wrapper used by the selected attack implementation.
            attack: Instantiated attack object (for example ART attack instance).
            attack_type: Parsed attack family.
            attack_subtype: Parsed attack subtype from attack path.

        Returns:
            Score dictionary merged into runtime ``score_dict``.

        Raises:
            NotImplementedError: Always raised by the base attack mixin.
        """
        raise NotImplementedError(
            "Attack handlers must implement __call__",
        )

    def attack(
        self,
        *,
        data: DataConfig,
        model: ModelConfig | EstimatorLike | BaseEstimator,
        art_model: EstimatorLike,
        attack: AttackLike,
        attack_type: str,
        attack_subtype: str,
    ) -> ScoreDict:
        """Public alias for base attack dispatch.

        This mirrors the canonical attack type name and keeps a consistent
        public entrypoint across attack mixins.
        """
        return self(
            data=data,
            model=model,
            art_model=art_model,
            attack=attack,
            attack_type=attack_type,
            attack_subtype=attack_subtype,
        )


@dataclass(eq=False, kw_only=True)
class AttackTypePlugin:
    """Generic attack plugin that binds one mixin to one attack family/subtype.

    Initialization fields
    ---------------------
    mixin_type : Any
        Mixin class (or import path) implementing runtime ``__call__``.
    attack_type : str
        Attack family this plugin matches.
    attack_subtype : str | None
        Optional subtype constraint.
    excluded_subtypes : tuple[str, ...]
        Subtypes explicitly excluded from this plugin match.

    Runtime behavior
    ----------------
    - ``resolve_attack_mixins`` contributes mixins to runtime context assembly.
    - ``resolve_attack_handler`` returns callable handler for dispatch.
    - ``__call__`` forwards ``*args``/``**kwargs`` to the configured mixin
      instance bound to the runtime config.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    mixin_type: Any
    attack_type: str
    attack_subtype: Union[str, None] = None
    excluded_subtypes: tuple[str, ...] = field(default_factory=tuple)

    def _resolve_mixin_type(self) -> type:
        if isinstance(self.mixin_type, str):
            resolved = resolve_class(self.mixin_type)
            self.mixin_type = resolved
            return resolved
        return self.mixin_type

    def _matches(self, *, attack_type: str, attack_subtype: str) -> bool:
        if (attack_type or "").lower() != (self.attack_type or "").lower():
            return False
        subtype = (attack_subtype or "").lower()
        if self.attack_subtype is not None and subtype != self.attack_subtype.lower():
            return False
        if subtype in {item.lower() for item in self.excluded_subtypes}:
            return False
        return True

    def resolve_attack_mixins(
        self,
        runtime: "AttackConfig",
        *,
        attack_type: str,
        attack_subtype: str,
        default_mixins: tuple[type, ...],
    ) -> tuple[type, ...]:
        """Return mixin tuple for matching attack family/subtype.

        Args:
            runtime: Active runtime attack config.
            attack_type: Requested attack family.
            attack_subtype: Requested attack subtype.
            default_mixins: Default mixins for this attack family.

        Returns:
            Mixin tuple to attach to runtime context.
        """
        _ = (runtime, default_mixins)
        if not self._matches(attack_type=attack_type, attack_subtype=attack_subtype):
            return ()
        mixin = self._resolve_mixin_type()
        return (mixin,)

    def resolve_attack_handler(
        self,
        runtime: "AttackConfig",
        *,
        attack_type: str,
        attack_subtype: str,
        default_handler: Callable[..., ScoreDict] | None,
        default_mixins: tuple[type, ...],
    ) -> Callable[..., ScoreDict] | None:
        """Return callable runtime handler for matching attack family/subtype.

        Args:
            runtime: Active runtime attack config.
            attack_type: Requested attack family.
            attack_subtype: Requested attack subtype.
            default_handler: Existing resolved runtime handler.
            default_mixins: Existing resolved mixins.

        Returns:
            Callable runtime handler when plugin matches; otherwise ``None``.
        """
        _ = (default_handler, default_mixins)
        if not self._matches(attack_type=attack_type, attack_subtype=attack_subtype):
            return None
        return lambda *args, **kwargs: self(runtime, *args, **kwargs)

    def __call__(self, runtime: "AttackConfig", *args, **kwargs) -> ScoreDict:
        """Delegate runtime attack execution to configured mixin handler.

        Args:
            runtime: Runtime config instance currently orchestrating the attack.
            *args: Positional runtime args forwarded to mixin ``__call__``.
            **kwargs: Keyword runtime args forwarded to mixin ``__call__``.

        Returns:
            Runtime attack score payload.
        """
        mixin = self._resolve_mixin_type()
        handler = mixin(runtime)
        return ScoreDict.from_payload(handler(*args, **kwargs))


def _get_sklearn_dict() -> dict[str, Any]:
    return _get_art_symbols()["sklearn_dict"]


def _get_supported_models() -> tuple[type, ...]:
    return tuple(_get_sklearn_dict().values())


@dataclass(eq=False, kw_only=True)
class AttackConfig(BaseConfig):
    """Runtime attack configuration with plugin-driven dispatch.

    Attack behavior is resolved at runtime via mixins and optional plugins.
    Concrete attack logic lives in type-specific modules, while this class
    owns orchestration, timing, scoring, and plugin hook execution.

    Note:
        Runtime hook names include ``resolve_attack_mixins``,
        ``resolve_attack_handler``, ``before_attack_dispatch``, and
        ``after_attack_dispatch``. Dictionary outputs from post-dispatch hooks
        are merged into ``score_dict``.

        ``attack_params`` are constructor kwargs filtered during
        ``_initialize_attack``. Some families also consume runtime control keys
        (for example poisoning trigger controls or inference split controls)
        before attack object construction.

    Attributes:
        attack_type: Attack family/type path used to resolve attack class.
        attack_params: Constructor and runtime parameters for attack execution.
        attack_size: Number of samples used for attack execution.
        plugins: Runtime attack plugins used for dispatch/hook extension.
        scorer: Attack scorer configuration applied to attack outputs.
        score_dict: Runtime score payload collected during attack execution.
    """

    # Configuration fields
    attack_type: str = "art.attacks.evasion.HopSkipJump"
    attack_params: dict = field(
        default_factory=dict,
        metadata={"help": "Parameters for the attack."},
    )
    attack_size: int = field(
        default=1000,
        metadata={"help": "Number of samples to use for the attack."},
    )
    targeted_attribute: str = field(
        default_factory=str,
        metadata={"help": "Targeted attribute for inference attacks."},
    )
    scorer: Union["AttackScorerConfig", None] = None
    alias: Union[str, None] = None
    plugins: list = field(default_factory=list)
    device: Union[str, None] = None
    mode: Literal["auto", "train", "test", "val"] = "auto"

    # Runtime state fields
    attack_time: Union[float, None] = None
    attack_prediction_time: Union[float, None] = None
    attack_score_time: Union[float, None] = None
    attack: Union[object, None] = None
    attack_predictions: Union[object, None] = None
    attacked_labels: Union[object, None] = None
    score_y_pred: Union[object, None] = None
    score_y_proba: Union[object, None] = None
    target_index: Union[int, None] = None
    _attack_type: Union[str, None] = None
    _attack_subtype: Union[str, None] = None
    score_dict: ScoreDict = field(default_factory=ScoreDict)
    _target_: Union[str, None] = None
    _plugin_objects: Union[list, None] = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __hash__(self):
        return super().__hash__()

    def __post_init__(self):
        """
        Initializes post-construction attributes for the class.

        Sets the internal attack attribute to None. If attack_params is not provided,
        initializes it as an empty dictionary.
        """
        self._target_ = "deckard.attack.AttackConfig"
        attack_scorer_cls = resolve_class(
            "deckard.score.attack.AttackScorerConfig",
        )
        # Coerce user-specified scorer, if provided
        if isinstance(self.scorer, str):
            if is_null_config_value(self.scorer):
                self.scorer = None
            elif is_default_config_value(self.scorer, include_best=False):
                self.scorer = attack_scorer_cls()
            else:
                self.scorer = load_class(self.scorer)
        elif isinstance(self.scorer, DictConfig):
            self.scorer = OmegaConf.to_container(self.scorer, resolve=True)
        # Otherwise, handle expected input values
        if self.scorer is None:
            self.scorer = attack_scorer_cls()
        elif isinstance(self.scorer, dict):
            scorer_spec = dict(self.scorer)
            scorer_target = scorer_spec.pop("_target_", scorer_spec.pop("name", None))
            if scorer_target is None:
                self.scorer = attack_scorer_cls(**scorer_spec)
            else:
                self.scorer = load_class(scorer_target, **scorer_spec)
        elif isinstance(self.scorer, type):
            self.scorer = self.scorer()

        if not hasattr(self.scorer, "_score"):
            raise TypeError(
                "AttackConfig scorer must expose a '_score' method.",
            )
        ensure_attack_runtime_contract(self)
        self.mode = str(self.mode or "auto").strip().lower()
        self._validate_poisoning_params()
        self.device = str(resolve_torch_device(self.device))

    def load_cached_attack_artifacts(
        self,
        attack_file: str | None,
        attack_predictions_file: str | None,
    ) -> None:
        """Load previously persisted attack runtime artifacts when available.

        Args:
            attack_file: Optional persisted attack object path.
            attack_predictions_file: Optional persisted predictions path.
        """
        if attack_file is not None and Path(attack_file).exists():
            loaded_self = self.load_object(
                attack_file,
                ignore_corrupt=True,
                delete_corrupt=True,
            )
            if loaded_self is not None:
                self.__dict__.update(loaded_self.__dict__)
        if (
            attack_predictions_file is not None
            and Path(attack_predictions_file).exists()
        ):
            try:
                self.attack_predictions = self.load_data(attack_predictions_file)
            except (ValueError, OSError) as exc:
                logger.warning(
                    "Failed to load cached attack predictions %s (%s). Recomputing predictions.",
                    attack_predictions_file,
                    exc,
                )
                Path(attack_predictions_file).unlink(missing_ok=True)

    def _validate_poisoning_params(self):
        """Validate poisoning-specific configuration parameters."""
        attack_type = (self.attack_family or "").lower()
        if attack_type != "poisoning":
            return

        if str(self.attack_type).endswith("PoisoningAttackSVM"):
            return

        required_keys = ("class_source", "class_target")
        missing_keys = [k for k in required_keys if k not in self.attack_params]
        if missing_keys:
            raise ValueError(
                "Poisoning attacks require attack_params to include "
                f"{required_keys}. Missing: {tuple(missing_keys)}",
            )

        class_source = int(self.attack_params["class_source"])
        class_target = int(self.attack_params["class_target"])
        if class_source == class_target:
            raise ValueError(
                "Poisoning attacks require class_source and class_target to differ.",
            )

    def set_mode(
        self,
        mode: Literal["auto", "train", "test", "val"],
    ) -> "AttackConfig":
        """Set attack scoring/evaluation split mode explicitly.

        Args:
            mode: Attack mode token.

        Returns:
            This AttackConfig instance.
        """
        self.mode = normalize_attack_mode(mode)
        return self

    def _run_attack_stage_hooks(
        self,
        when: str,
        stage: str,
        **kwargs: Any,
    ) -> None:
        event = str(when).strip().lower()
        if event not in {"before", "after"}:
            raise ValueError(f"Invalid attack hook event: {when}")
        canonical_stage = normalize_attack_stage(stage)
        hook_outputs = self._run_plugin_hook(
            f"{event}_attack_stage",
            stage=canonical_stage,
            **kwargs,
        )
        self._merge_plugin_scores(hook_outputs)

    def resolve_mode_for_attack_kind(
        self,
        attack_kind: Optional[str],
        *,
        attack_subtype: Optional[str] = None,
        split_override: Optional[str] = None,
    ) -> Literal["train", "test", "val"]:
        """Resolve active split mode from overrides, explicit mode, or auto defaults.

        Precedence order:
        1) Explicit split override (method arg or attack_params['split']).
        2) Explicit attack mode (train/test/val).
        3) Auto defaults inferred from attack family/subtype/kind.

        Args:
            attack_kind: Canonical attack kind token.
            attack_subtype: Optional attack subtype token.
            split_override: Optional split override token.

        Returns:
            Canonical split mode.

        Raises:
            ValueError: If configured mode or split override token is unsupported.
        """
        valid_modes = {"auto", "train", "test", "val"}
        mode_value = str(self.mode).strip().lower()
        if mode_value not in valid_modes:
            raise ValueError(
                f"Unsupported attack mode '{self.mode}'. Expected one of: {', '.join(sorted(valid_modes))}.",
            )

        requested_split = split_override
        if requested_split is None and isinstance(self.attack_params, dict):
            requested_split = self.attack_params.get("split")
        if requested_split is not None:
            canonical_split = str(requested_split).strip().lower()
            if canonical_split not in {"train", "test", "val"}:
                raise ValueError(
                    "Unsupported attack split override "
                    f"'{requested_split}'. Expected one of: train, test, val.",
                )
            return canonical_split

        if mode_value in {"train", "test", "val"}:
            return mode_value
        attack_family = (self.attack_family or "").lower()
        subtype = (attack_subtype or self.attack_subtype or "").lower()
        kind = (attack_kind or "").lower()

        if attack_family == "poisoning":
            return "train"
        if attack_family == "evasion":
            return "test"
        if attack_family == "extraction":
            return "test"
        if attack_family == "inference":
            if subtype in {
                "membership_inference",
                "attribute_inference",
                "reconstruction",
            }:
                return "train"
            if subtype == "model_inversion":
                return "test"

        if kind in {"membership", "attribute", "poisoning", "reconstruction"}:
            return "train"
        return "test"

    def _parse_attack_path(self) -> tuple[str, str]:
        parts = (self.attack_type or "").split("attacks.")[-1].split(".")
        attack_type = parts[0] if len(parts) > 0 else ""
        attack_subtype = parts[1] if len(parts) > 1 else ""
        return attack_type, attack_subtype

    def _instantiate_plugin(self, plugin_spec: Any):
        return instantiate_plugin_spec(
            plugin_spec,
            loader=load_class,
        )

    def _get_plugins(self) -> list:
        if self._plugin_objects is None:
            plugin_specs = normalize_plugin_specs(self.plugins)
            self._plugin_objects = [
                self._instantiate_plugin(spec) for spec in plugin_specs
            ]
        return self._plugin_objects

    def _run_plugin_hook(self, hook_name: str, **kwargs) -> list[Any]:
        """Execute one plugin hook across all instantiated plugins.

        Args:
            hook_name: Hook method name to invoke when present on a plugin.
            **kwargs: Hook-specific keyword arguments.

        Returns:
            Ordered list of hook return values.
        """
        hook_outputs = []
        for plugin in self._get_plugins():
            hook = getattr(plugin, hook_name, None)
            if callable(hook):
                hook_outputs.append(hook(self, **kwargs))
        return hook_outputs

    def _merge_plugin_scores(self, hook_outputs):
        if self.score_dict is None:
            self.score_dict = ScoreDict()
        for output in hook_outputs:
            if isinstance(output, dict):
                self.score_dict.update(ScoreDict.from_payload(output))

    def _resolve_runtime_attack_mixins(
        self,
        attack_type: str,
        attack_subtype: str,
    ) -> tuple[type, ...]:
        mixins: list[type] = []
        attack_type_lower = (attack_type or "").lower()
        attack_subtype_lower = (attack_subtype or "").lower()

        if attack_type_lower == "evasion":
            from .evasion import EvasionAttackMixin

            mixins.append(EvasionAttackMixin)
        elif attack_type_lower == "poisoning":
            from .poisoning import PoisoningAttackMixin

            mixins.append(PoisoningAttackMixin)
        elif attack_type_lower == "extraction":
            from .extraction import ExtractionAttackMixin

            mixins.append(ExtractionAttackMixin)
        elif attack_type_lower == "inference":
            if attack_subtype_lower == "reconstruction":
                from .reconstruction import ReconstructionAttackMixin

                mixins.append(ReconstructionAttackMixin)
            else:
                from .inference import InferenceAttackMixin

                mixins.append(InferenceAttackMixin)

        plugin_outputs = self._run_plugin_hook(
            "resolve_attack_mixins",
            attack_type=attack_type,
            attack_subtype=attack_subtype,
            default_mixins=tuple(mixins),
        )
        for output in plugin_outputs:
            if isinstance(output, type):
                mixins.append(output)
            elif isinstance(output, (tuple, list)):
                for item in output:
                    if isinstance(item, type):
                        mixins.append(item)

        deduped: list[type] = []
        for mixin in mixins:
            if mixin not in deduped:
                deduped.append(mixin)
        return tuple(deduped)

    def _resolve_attack_handler(self, attack_type: str, attack_subtype: str):
        mixins = self._resolve_runtime_attack_mixins(attack_type, attack_subtype)
        default_handler = None
        for mixin in mixins:
            if isinstance(mixin, type) and issubclass(mixin, AttackMixin):
                default_handler = mixin(self)
                break

        hook_outputs = self._run_plugin_hook(
            "resolve_attack_handler",
            attack_type=attack_type,
            attack_subtype=attack_subtype,
            default_handler=default_handler,
            default_mixins=mixins,
        )
        for output in hook_outputs:
            if callable(output):
                return output
            if isinstance(output, type) and issubclass(output, AttackMixin):
                return output(self)

        return default_handler

    def _with_attack_context(self, attack_type: str, attack_subtype: str):
        mixins = self._resolve_runtime_attack_mixins(
            attack_type=attack_type,
            attack_subtype=attack_subtype,
        )
        if len(mixins) == 0:
            return self

        runtime_cls = type(
            f"RuntimeAttackContext_{attack_type}_{attack_subtype}_{self.__class__.__name__}",
            (*mixins, self.__class__),
            {},
        )
        runtime = copy.copy(self)
        runtime.__class__ = runtime_cls
        return runtime

    @property
    def attack_family(self) -> Optional[str]:
        """Return canonical attack family resolved from runtime attack declaration.

        Returns:
            Canonical attack family token when available.
        """
        if self._attack_type:
            return self._attack_type
        attack_type, _ = self._parse_attack_path()
        return attack_type or None

    @property
    def attack_subtype(self) -> Optional[str]:
        """Return canonical attack subtype resolved from runtime attack declaration.

        Returns:
            Canonical attack subtype token when available.
        """
        if self._attack_subtype:
            return self._attack_subtype
        _, attack_subtype = self._parse_attack_path()
        return attack_subtype or None

    @property
    def attack_kind(self) -> Optional[str]:
        """Return normalized scoring attack kind token for the configured attack.

        Returns:
            Normalized attack scoring kind token.
        """
        attack_type = (self.attack_family or "").lower()
        subtype = (self.attack_subtype or "").lower()

        if attack_type == "evasion":
            return "evasion"
        if attack_type == "inference" and "membership" in subtype:
            return "membership"
        if attack_type == "inference" and "attribute" in subtype:
            return "attribute"
        return None

    @staticmethod
    def _infer_task_is_classification(
        data: DataConfig,
        model: ModelConfig | EstimatorLike | BaseEstimator,
    ) -> Optional[bool]:
        """Infer task type from model first, then data config as fallback."""
        if isinstance(model, ModelConfig) and model.classifier is not None:
            return bool(model.classifier)
        if isinstance(model, RegressorMixin) and not isinstance(
            model,
            ClassifierMixin,
        ):
            return False
        if isinstance(model, ClassifierMixin):
            return True
        if hasattr(data, "classifier") and getattr(data, "classifier") is not None:
            return bool(getattr(data, "classifier"))
        return None

    def _validate_attack_task_compatibility(
        self,
        data: DataConfig,
        model: ModelConfig | EstimatorLike | BaseEstimator,
    ):
        """Fail fast for known unsupported task/attack combinations."""
        attack_type = (self.attack_family or "").lower()
        task_is_classification = self._infer_task_is_classification(data, model)
        if attack_type == "evasion" and task_is_classification is False:
            raise ValueError(
                "Evasion attacks are not supported for regression models in the current sklearn+ART integration.",
            )

    def _initialize_attack(
        self,
        model: ModelConfig | EstimatorLike | BaseEstimator,
        data: DataConfig,
    ):
        """Initialize attack runtime objects for the provided model/data context.

        Args:
            model: Model payload or model config to attack.
            data: Runtime dataset used for ART wrapping and subset extraction.

        Returns:
            Tuple of initialized attack object, ART model, attack family, and subtype.

        Raises:
            ValueError: If attack type/model type is unsupported.
        """
        attack_type = self.attack_family or ""
        attack_subtype = self.attack_subtype or ""

        art_model = None
        if isinstance(model, ModelConfig):
            runtime_model = getattr(model, "_model", None)
            if attack_type == "extraction" and is_torch_model(runtime_model):
                # Extraction expects a neural-network ART classifier; build directly
                # from the underlying torch module when available.
                art_model = build_torch_art_model(model=runtime_model, data=data)
            else:
                art_model = model.get_art_model(data)
        elif is_torch_model(model):
            art_model = build_torch_art_model(model=model, data=data)
        else:
            check_is_fitted(model)

        # Validate attack type
        if attack_type not in [
            "evasion",
            "poisoning",
            "extraction",
            "inference",
        ]:
            raise ValueError(f"Unsupported attack type: {attack_type}")

        if attack_type == "poisoning":
            self._validate_poisoning_params()

        attack_class = resolve_class(self.attack_type)
        if art_model is None:
            sklearn_dict = _get_sklearn_dict()
            if isinstance(model, _get_supported_models()):
                art_model = model
            elif (
                isinstance(model, BaseEstimator)
                and type(model).__name__ in sklearn_dict
            ):
                assert isinstance(
                    model,
                    ClassifierMixin,
                ), f"Model must be a ClassifierMixin, got {type(model)}"
                model_alias = type(model).__name__
                art_cls = sklearn_dict[model_alias]
                try:
                    check_is_fitted(model)
                except NotFittedError as e:
                    logger.debug(e)
                    model.fit(data.X_train, data.y_train)
                art_model = art_cls(model)
            elif isinstance(model, BaseEstimator):
                try:
                    check_is_fitted(model)
                except NotFittedError:
                    model.fit(data.X_train, data.y_train)
                # Wrap models that require sensitive_features in predict (e.g. ThresholdOptimizer)
                import inspect

                predict_sig = inspect.signature(model.predict)
                if "sensitive_features" in predict_sig.parameters:
                    sensitive = getattr(data, "_sensitive_test", None)
                    if sensitive is None:
                        sensitive = getattr(data, "_sensitive_train", None)
                    if sensitive is not None:
                        model = SensitiveFeaturesWrapper(model, sensitive)
                if isinstance(model, RegressorMixin) and not isinstance(
                    model,
                    ClassifierMixin,
                ):
                    art_model = sklearn_dict["sklearn-regressor"](model)
                else:
                    art_model = sklearn_dict["sklearn-classifier"](model)
                if art_model.input_shape is None:
                    art_model._input_shape = (data.X_train.shape[1],)
                nb = getattr(art_model, "nb_classes", None)
                if nb is None or nb <= 0:
                    art_model.nb_classes = len(
                        np.unique(np.asarray(data.y_train).flatten()),
                    )
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")
        # Convert targeted attribute to index if necessary
        if len(self.targeted_attribute) > 0 and isinstance(
            self.targeted_attribute,
            str,
        ):
            feature_name = self.targeted_attribute
            assert isinstance(
                data.X_train,
                pd.DataFrame,
            ), f"Expected Dataframe got {type(data.X_train)}"
            if not hasattr(self, "target_index"):
                if feature_name not in data.X_train.columns:
                    cols = [
                        col
                        for col in data.X_train.columns
                        if feature_name.split("_")[0] in col
                    ]
                    raise ValueError(
                        f"{feature_name} not found. Did you mean one of these: {cols}?",
                    )
                self.target_index = data.X_train.columns.get_loc(feature_name)
                self.attack_params["attack_feature"] = self.target_index
                assert (
                    "attack_feature" in self.attack_params
                ), "attack_feature must be specified in attack_params for attribute inference attacks"
        # TODO: Set labels to distinguish targeted attacks from non-targeted attacks
        if "attack_model" in self.attack_params:
            attack_model = self.attack_params["attack_model"]
            if isinstance(attack_model, DictConfig):
                dict_ = OmegaConf.to_container(attack_model)
                cfg = ModelConfig(**dict_)
                cfg(data)
                attack_model = cfg.get_art_model(data)
            elif isinstance(attack_model, ModelConfig):
                attack_model._load_or_train_model(data)
                attack_model = attack_model.get_art_model(data)
            elif isinstance(attack_model, str):
                assert Path(
                    attack_model,
                ).exists(), f"attack_model path {attack_model} does not exist"
                with open(attack_model, "rb") as f:
                    attack_model = pickle.load(f)
                    assert isinstance(
                        attack_model,
                        ModelConfig,
                    ), "Loaded attack_model must be a ModelConfig instance"
                    attack_model = attack_model.get_art_model(data)
            else:
                raise ValueError(
                    f"attack_model must be a ModelConfig instance. Got {type(attack_model)}",
                )
            self.attack_params["attack_model"] = attack_model
        attack_init_params = copy.deepcopy(self.attack_params)
        if attack_type == "poisoning":
            # Internal orchestration fields are not constructor args for ART attacks.
            for key in (
                "class_source",
                "class_target",
                "trigger_index",
                "poison_fit_params",
                "num_workers",
            ):
                attack_init_params.pop(key, None)
            if self._is_poisoning_svm_attack(attack_class):
                self._ensure_poisoning_svm_clip_values(art_model, data)
                attack_init_params.update(
                    self._build_poisoning_svm_init_params(data),
                )
        if attack_type == "inference" and attack_subtype == "model_inversion":
            for key in (
                "split",
                "targets",
                "initialization",
                "x_init",
            ):
                attack_init_params.pop(key, None)
        if attack_type == "inference" and attack_subtype == "reconstruction":
            for key in (
                "split",
                "missing_index",
            ):
                attack_init_params.pop(key, None)
        attack = attack_class(art_model, **attack_init_params)
        self._attack_type = attack_type
        self._attack_subtype = attack_subtype
        return attack, art_model, attack_type, attack_subtype

    def initialize_attack(
        self,
        model: ModelConfig | BaseEstimator | EstimatorLike,
        data: DataConfig,
    ) -> tuple[AttackLike, EstimatorLike, str, str]:
        """Public entry-point for attack initialisation. Delegates to _initialize_attack().

        Args:
            model: Runtime model payload.
            data: Runtime data payload.

        Returns:
            Initialized attack, ART model, attack family, and attack subtype.
        """
        return self._initialize_attack(model, data)

    @staticmethod
    def _is_poisoning_svm_attack(attack_class) -> bool:
        return "PoisoningAttackSVM" in getattr(attack_class, "__name__", "")

    def _build_poisoning_svm_init_params(self, data: DataConfig) -> dict:
        y_train = self._target_to_class_labels(getattr(data, "y_train"))
        x_val = getattr(data, "X_val", None)
        y_val = getattr(data, "y_val", None)
        if x_val is None or y_val is None:
            x_val = getattr(data, "X_test")
            y_val = getattr(data, "y_test")
        nb_classes = int(np.max(y_train)) + 1
        return {
            "x_train": self._prepare_features_for_art(getattr(data, "X_train")),
            "y_train": self._one_hot_encode(y_train, nb_classes=nb_classes),
            "x_val": self._prepare_features_for_art(x_val),
            "y_val": self._one_hot_encode(
                self._target_to_class_labels(y_val),
                nb_classes=nb_classes,
            ),
        }

    def _ensure_poisoning_svm_clip_values(
        self,
        art_model: EstimatorLike,
        data: DataConfig,
    ) -> None:
        if getattr(art_model, "clip_values", None) is not None:
            return
        x_train = self._prepare_features_for_art(getattr(data, "X_train"))
        lower = float(np.min(x_train))
        upper = float(np.max(x_train))
        if hasattr(art_model, "_clip_values"):
            art_model._clip_values = (lower, upper)
        else:
            art_model.clip_values = (lower, upper)

    def __call__(
        self,
        data: DataConfig,
        model: ModelConfig | BaseEstimator | EstimatorLike,
        files: dict[str, str | None] | None = None,
        attack_file: Union[str, None] = None,
        attack_predictions_file: Union[str, None] = None,
        score_file: Union[str, None] = None,
    ) -> ScoreDict:
        """
        Executes the specified attack on the provided model using the given data.

        Args:
            data: Input data used for attack execution.
            model: Model payload to be attacked.
            files: Optional runtime file alias mapping.
            attack_file: Optional path for persisted attack object.
            attack_predictions_file: Optional path for persisted attack predictions.
            score_file: Optional path for persisted scores.

        Returns:
            A score payload containing attack scores and timing information.

        Raises:
            ValueError: If attack type/subtype/model wiring is unsupported.
            NotImplementedError: If selected attack runtime is not implemented.
            AssertionError: If runtime outputs fail payload assertions.
        """
        files = dict(files or {})
        if attack_file is None:
            attack_file = files.get("attack_file")
        if attack_predictions_file is None:
            attack_predictions_file = files.get("attack_predictions_file")
        if score_file is None:
            score_file = files.get("score_file")

        self.load_cached_attack_artifacts(
            attack_file=attack_file,
            attack_predictions_file=attack_predictions_file,
        )
        ensure_attack_runtime_contract(self)
        self._validate_attack_task_compatibility(data, model)

        attack, art_model, attack_type, attack_subtype = self._initialize_attack(
            model,
            data,
        )
        runtime = self._with_attack_context(
            attack_type=attack_type,
            attack_subtype=attack_subtype,
        )
        handler = runtime._resolve_attack_handler(
            attack_type=attack_type,
            attack_subtype=attack_subtype,
        )
        if handler is None:
            raise NotImplementedError(
                f"Attack type {attack_type} subtype {attack_subtype} has no registered runtime handler.",
            )

        before_outputs = runtime._run_plugin_hook(
            "before_attack_dispatch",
            data=data,
            model=model,
            attack=attack,
            art_model=art_model,
            attack_type=attack_type,
            attack_subtype=attack_subtype,
            runtime=runtime,
            handler=handler,
        )
        runtime._merge_plugin_scores(before_outputs)

        attack_execution_order = "post-defense"
        runtime._run_attack_stage_hooks(
            "before",
            "pre-attack",
            data=data,
            model=model,
            attack=attack,
            art_model=art_model,
            attack_type=attack_type,
            attack_subtype=attack_subtype,
            execution_order=attack_execution_order,
        )

        scores = handler(
            data=data,
            model=model,
            art_model=art_model,
            attack=attack,
            attack_type=attack_type,
            attack_subtype=attack_subtype,
        )

        self.__dict__.update(runtime.__dict__)
        self._run_attack_stage_hooks(
            "after",
            "post-attack",
            data=data,
            model=model,
            attack=attack,
            art_model=art_model,
            attack_type=attack_type,
            attack_subtype=attack_subtype,
            execution_order=attack_execution_order,
        )
        after_outputs = self._run_plugin_hook(
            "after_attack_dispatch",
            data=data,
            model=model,
            attack=attack,
            art_model=art_model,
            attack_type=attack_type,
            attack_subtype=attack_subtype,
            scores=scores,
        )
        self._merge_plugin_scores(after_outputs)
        assert isinstance(scores, dict), "Scores should be a dictionary"
        assert isinstance(
            self.attack_time,
            float,
        ), f"Attack time should be a float, got {type(self.attack_time)}"
        assert isinstance(
            self.attack_prediction_time,
            float,
        ), "Attack prediction time should be a float"
        assert isinstance(
            self.attack_score_time,
            float,
        ), "Attack score time should be a float"
        times = {
            "attack_generation_time": self.attack_time,
            "attack_prediction_time": self.attack_prediction_time,
            "attack_score_time": self.attack_score_time,
        }
        score_dict = {
            **scores,
            **times,
            "attack_stage": normalize_attack_stage("post-attack"),
            "attack_execution_order": attack_execution_order,
        }
        self.score_dict = ScoreDict.from_payload(score_dict)

        self.merge_runtime_files(
            {
                "attack_file": attack_file,
                "attack_predictions_file": attack_predictions_file,
                "score_file": score_file,
            },
        )

        # Save attack, predictions, and scores if file paths are provided
        if attack_file is not None and not Path(attack_file).exists():
            try:
                self.save_object(self, attack_file)
            except (pickle.PicklingError, AttributeError, TypeError) as exc:
                logger.warning(
                    "Failed to cache attack object %s (%s). Continuing without cache.",
                    attack_file,
                    exc,
                )
                Path(attack_file).unlink(missing_ok=True)
        if attack_predictions_file is not None:
            self.save_data(self.attack_predictions, attack_predictions_file)
        self.score_dict = ScoreDict.from_payload(
            self.merge_and_persist_scores(self.score_dict, score_file),
        )
        return ScoreDict.from_payload(score_dict)

    def _get_benign_preds(
        self,
        data: DataConfig,
        art_model: EstimatorLike,
        train: bool = False,
    ):
        """
        Generate benign predictions and corresponding labels for a subset of data.

        Depending on the `train` flag, selects either the training or test set, obtains predictions
        from the provided ART model, and returns the predicted labels along with the corresponding
        data subset and true labels.

        Args:
            data: Data runtime providing train and test splits.
            art_model: ART-compatible model with a ``predict`` method.
            train: When ``True``, score against the test split; otherwise use the training split.

        Returns:
            Tuple of subset size, benign prediction labels, feature subset, and
            true-label subset.
        """
        n = self.attack_size
        if train is True:
            ben_preds = art_model.predict(
                self._prepare_features_for_art(data.X_test),
            )
            ben_pred_labels = ben_preds.argmax(axis=1)
            n, X_subset, y_subset = self.get_attack_subset(data, test=True)
        else:
            ben_preds = art_model.predict(
                self._prepare_features_for_art(data.X_train),
            )
            ben_preds = tensor_to_numpy(ben_preds, dtype=ART_NUMPY_DTYPE)
            ben_pred_labels = ben_preds.argmax(axis=1)
            n, X_subset, y_subset = self.get_attack_subset(data, test=False)
        y_subset = tensor_to_numpy(y_subset, dtype=ART_NUMPY_DTYPE)
        assert isinstance(
            ben_pred_labels,
            np.ndarray,
        ), f"ben_pred_labels should be np.ndarray, got {type(ben_pred_labels)}"
        assert isinstance(
            X_subset,
            np.ndarray,
        ), f"X_subset should be np.ndarray, got {type(X_subset)}"
        assert isinstance(
            y_subset,
            np.ndarray,
        ), f"y_subset should be np.ndarray, got {type(y_subset)}"
        return n, ben_pred_labels, X_subset, y_subset

    def _get_feature_vector_preds(
        self,
        data: DataConfig,
        targeted_attribute: str | list[str],
        train: bool = False,
    ):
        """
        Extracts a subset of feature vectors, labels, and attributes from the provided data for either training or testing.

        Args:
            data: Data runtime providing train and test splits.
            targeted_attribute: Attribute column or columns to recover.
            train: When ``True``, extract from the training split; otherwise use the test split.

        Returns:
            Tuple of subset size, feature subset, label subset, and attribute subset.

        Raises:
            AssertionError: If the extracted feature, label, and attribute lengths diverge.
        """
        n = self.attack_size
        if train is False:
            X_train = data.X_train
            y_train = data.y_train
            a_train = data.X_train[targeted_attribute]
            X_test = data.X_test
            y_test = data.y_test
            a_test = data.X_test[targeted_attribute]
            X_train = X_train.drop(columns=[targeted_attribute])
            X_test = X_test.drop(columns=[targeted_attribute])
            assert (
                len(X_test) == len(y_test) == len(a_test)
            ), "X_test, y_test, and a_test must have the same length, but got lengths: {}, {}, {}".format(
                len(X_test),
                len(y_test),
                len(a_test),
            )
            X_subset = X_test[:n]
            y_subset = y_test[:n]
            a_subset = a_test[:n]
        else:

            assert (
                len(X_train) == len(y_train) == len(a_train)
            ), "X_train, y_train, and a_train must have the same length, but got lengths: {}, {}, {}".format(
                len(X_train),
                len(y_train),
                len(a_train),
            )
            X_subset = X_train[:n]
            y_subset = y_train[:n]
            a_subset = a_train[:n]
        return n, X_subset, y_subset, a_subset

    def _score_attack(self, ben_pred_labels, adv_pred_labels, y_test_numeric):
        """
        Computes and logs various performance metrics for adversarial attack predictions.

        Calculates the following metrics for the adversarial predictions:
            - Accuracy
            - Precision
            - Recall
            - F1-score
            - Success rate (agreement between benign and adversarial predictions)

        Args:
            ben_pred_labels: Predicted labels from the benign model.
            adv_pred_labels: Predicted labels from the adversarially perturbed model.
            y_test_numeric: True numeric labels for the evaluation subset.

        Returns:
            ``None``. The method updates ``self.score_dict`` with computed metrics.
        """
        score_dict = self._score(
            attack_kind="evasion",
            y_true=y_test_numeric,
            y_pred=adv_pred_labels,
            ben_pred_labels=ben_pred_labels,
        )
        logger.info(
            f"Attack scoring took {self.attack_score_time} seconds for {len(adv_pred_labels)} samples and {len(self.score_dict)} scores.",
        )
        self.score_dict = {**self.score_dict, **score_dict}
        for score in self.score_dict:
            logger.info(f"{score}: {self.score_dict[score]}")

    def _score_comparison(
        self,
        *,
        y_true,
        y_pred,
        stage: str,
        prefix: str,
        is_classification: bool,
        y_proba=None,
        mode: str | None = None,
        ben_pred_labels=None,
        sensitive_features=None,
    ) -> ScoreDict:
        """Score one comparison branch and namespace outputs by prefix.

        Used by attack runtimes to emit pairs like benign/adversarial metrics
        without relying on caller-side key rewriting.
        """
        if not is_classification:
            return ScoreDict()
        attack_kind = (self.attack_kind or "evasion").lower()
        raw_scores = self._score(
            attack_kind=attack_kind,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            stage=stage,
            mode=mode,
            ben_pred_labels=(y_pred if ben_pred_labels is None else ben_pred_labels),
            sensitive_features=sensitive_features,
        )
        normalized_scores: dict[str, Any] = {}
        attack_prefix = f"{attack_kind}_"
        for key, value in raw_scores.items():
            metric_key = str(key)
            if metric_key.startswith(attack_prefix):
                metric_key = metric_key[len(attack_prefix) :]
            if metric_key == "f1-score":
                metric_key = "f1"
            normalized_scores[f"{prefix}_{metric_key}"] = value
        return ScoreDict.from_payload(normalized_scores)

    def _score(
        self,
        attack_kind: str,
        y_true,
        y_pred=None,
        *args,
        **kwargs,
    ) -> ScoreDict:
        """Dispatch attack scoring through the configured AttackScorerConfig."""
        if self.scorer is None:
            raise ValueError(
                "AttackConfig.scorer must be configured with an AttackScorerConfig instance",
            )
        if y_pred is None:
            y_pred = self.score_y_pred
        y_proba = kwargs.pop("y_proba", None)
        if y_proba is None:
            y_proba = self.score_y_proba

        score_kwargs = {
            "attack_kind": attack_kind,
            "y_true": y_true,
            "y_pred": y_pred,
            "attack_size": self.attack_size,
            "mode": self.resolve_mode_for_attack_kind(
                attack_kind,
                attack_subtype=self.attack_subtype,
            ),
            **kwargs,
        }
        if y_proba is not None:
            import inspect

            signature = inspect.signature(self.scorer._score)
            accepts_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in signature.parameters.values()
            )
            if accepts_var_kwargs or "y_proba" in signature.parameters:
                score_kwargs["y_proba"] = y_proba

        score_dict = self.scorer._score(
            *args,
            **score_kwargs,
        )
        self.attack_score_time = score_dict.get("attack_score_time")
        return ScoreDict.from_payload(score_dict)

    @staticmethod
    def _is_regression_prediction_output(y_true, predictions) -> bool:
        """Infer whether attack predictions represent regression outputs."""
        preds = np.asarray(predictions)
        labels = np.asarray(y_true)
        if preds.ndim > 1 and preds.shape[1] > 1:
            return False
        if preds.ndim > 1 and preds.shape[1] == 1:
            return True
        if preds.dtype.kind == "f" and labels.dtype.kind == "f":
            return True
        return False

    @staticmethod
    def _to_numpy_array(value, dtype=None, flatten: bool = False) -> np.ndarray:
        """Normalize array-like inputs (tensor/pandas/list/ndarray) to numpy arrays. Raises error for inconsistent shapes and provides clear error messages."""
        if is_tensor(value):
            arr = tensor_to_numpy(value, dtype=dtype)
        elif isinstance(value, pd.DataFrame):
            arr = value.values
            if dtype is not None:
                arr = arr.astype(dtype)
        elif isinstance(value, pd.Series):
            arr = value.values
            if dtype is not None:
                arr = arr.astype(dtype)
        elif isinstance(value, np.ndarray):
            arr = value.astype(dtype) if dtype is not None else value
        elif isinstance(value, (list, tuple)):
            shapes = [np.shape(v) for v in value]
            if len(set(shapes)) > 1:
                raise ValueError(
                    f"Inconsistent shapes in input list/tuple: {shapes}. All elements must have the same shape for conversion to numpy array.",
                )
            try:
                arr = np.asarray(value, dtype=dtype)
            except Exception as e:
                raise ValueError(
                    f"Failed to convert list/tuple to numpy array due to shape inconsistency or unsupported types: {e}",
                )
        else:
            try:
                arr = np.asarray(value, dtype=dtype)
            except Exception as e:
                raise ValueError(f"Failed to convert input to numpy array: {e}")

        arr = np.asarray(arr)
        return arr.reshape(-1) if flatten else arr

    def _prepare_features_for_attack(self, value):
        """Prepare feature inputs for attack APIs.

        Subclasses can override this to preserve framework-native tensors.
        """
        if is_dataloader(value):
            x_subset, _ = collect_subset_from_dataloader(value, n=len(value.dataset))
            return tensor_to_numpy(x_subset, dtype=ART_NUMPY_DTYPE)
        try:
            from torch.utils.data import DataLoader, Dataset, Subset
        except ImportError:  # pragma: no cover
            DataLoader = Dataset = Subset = ()
        if isinstance(value, (Dataset, Subset)) and not isinstance(value, DataLoader):
            loader = DataLoader(value, batch_size=len(value), shuffle=False)
            batch = next(iter(loader))
            features = batch[0] if isinstance(batch, (tuple, list)) else batch
            return tensor_to_numpy(features, dtype=ART_NUMPY_DTYPE)
        if is_tensor(value):
            return tensor_to_numpy(value, dtype=ART_NUMPY_DTYPE)
        if isinstance(value, pd.DataFrame):
            return value.values
        if isinstance(value, pd.Series):
            return value.values
        return value

    def _prepare_labels_for_attack(self, value):
        """Prepare label inputs for attack APIs.

        Subclasses can override this to preserve framework-native tensors.
        """
        if is_tensor(value):
            return tensor_to_numpy(value, dtype=ART_NUMPY_DTYPE)
        if isinstance(value, pd.DataFrame):
            return value.values
        if isinstance(value, pd.Series):
            return value.values
        return value

    def _prepare_features_for_art(self, value):
        """Prepare feature inputs specifically for ART model/attack boundaries."""
        prepared = self._prepare_features_for_attack(value)
        arr = self._to_numpy_array(prepared)
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(ART_NUMPY_DTYPE, copy=False)
        return arr

    def _prepare_labels_for_art(self, value):
        """Prepare labels specifically for ART model/attack boundaries."""
        prepared = self._prepare_labels_for_attack(value)
        return self._to_numpy_array(prepared)

    @classmethod
    def _labels_from_classifier_predictions(cls, predictions) -> np.ndarray:
        """Convert classifier outputs (labels/logits/probabilities) to class labels."""
        arr = cls._to_numpy_array(predictions)
        if arr.ndim == 1:
            if np.issubdtype(arr.dtype, np.floating):
                unique_vals = np.unique(arr)
                if np.all(np.isin(unique_vals, [0.0, 1.0])):
                    return arr.astype(int)
                return (arr >= 0.5).astype(int)
            return arr.astype(int)
        if arr.ndim == 2:
            if arr.shape[1] == 1:
                col = arr.reshape(-1)
                return (col >= 0.5).astype(int)
            return np.argmax(arr, axis=1).astype(int)
        return arr.reshape(-1).astype(int)

    @classmethod
    def _prediction_to_labels(cls, predictions, is_regression: bool = False):
        """Convert model/attack prediction outputs into score-ready labels."""
        arr = cls._to_numpy_array(predictions)
        if is_regression:
            return arr.reshape(-1)
        return cls._labels_from_classifier_predictions(arr)

    @classmethod
    def _normalize_ground_truth(cls, y_true, is_regression: bool = False):
        """Normalize y_true into a consistent 1D numpy representation."""
        if isinstance(y_true, pd.Series):
            if is_regression:
                return y_true.astype(float).values
            return y_true.astype("category").cat.codes.values
        if isinstance(y_true, pd.DataFrame):
            series = y_true.iloc[:, 0]
            if is_regression:
                return series.astype(float).values
            return series.astype("category").cat.codes.values
        arr = cls._to_numpy_array(y_true)
        if not is_regression and arr.ndim == 2 and arr.shape[1] > 1:
            return np.argmax(arr, axis=1)
        return arr.reshape(-1)

    @classmethod
    def _target_to_class_labels(cls, y) -> np.ndarray:
        """Convert labels/targets to 1D class-index labels."""
        arr = cls._to_numpy_array(y)
        if arr.ndim == 1:
            return arr.astype(int)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr.reshape(-1).astype(int)
        if arr.ndim == 2 and arr.shape[1] > 1:
            return np.argmax(arr, axis=1).astype(int)
        raise ValueError(f"Unsupported target shape for class labels: {arr.shape}")

    @staticmethod
    def _one_hot_encode(labels, nb_classes: int) -> np.ndarray:
        """One-hot encode class-index labels using ART default dtype."""
        labels = np.asarray(labels).reshape(-1).astype(int)
        one_hot = np.zeros((len(labels), int(nb_classes)), dtype=ART_NUMPY_DTYPE)
        one_hot[np.arange(len(labels)), labels] = 1.0
        return one_hot

    @classmethod
    def _normalize_inferred_output(cls, inferred, reference=None):
        """Normalize inferred outputs and align dimensions with reference labels."""
        arr = cls._to_numpy_array(inferred)
        if reference is None:
            return arr
        ref = cls._to_numpy_array(reference)
        if ref.ndim > arr.ndim:
            return pd.get_dummies(arr).values
        if arr.ndim > ref.ndim:
            return np.argmax(arr, axis=1)
        return arr

    @staticmethod
    def _looks_like_probabilities(pred) -> bool:
        pred = np.asarray(pred)
        if pred.ndim != 2 or pred.shape[1] <= 1:
            return False
        if not np.all(np.isfinite(pred)):
            return False
        if np.min(pred) < -1e-12 or np.max(pred) > 1.0 + 1e-12:
            return False
        row_sums = pred.sum(axis=1)
        return np.allclose(row_sums, 1.0, atol=1e-4)

    @staticmethod
    def _select_extraction_scorer(benign_pred, extracted_pred):
        """Use full classifier metrics when probabilities are available, else label-only metrics."""
        preds = [np.asarray(benign_pred), np.asarray(extracted_pred)]
        has_probabilities = all(
            AttackConfig._looks_like_probabilities(pred) for pred in preds
        )
        if has_probabilities:
            return DefaultClassifierScorerDictConfig(), True
        full_classifier = DefaultClassifierScorerDictConfig()
        label_only = {
            name: scorer
            for name, scorer in full_classifier.scorers.items()
            if not scorer.needs_proba
        }
        return ScorerDictConfig(scorers=label_only), False

    def _score_attack_legacy(
        self,
        ben_pred_labels,
        adv_pred_labels,
        y_test_numeric,
    ):
        """Backward-compatible alias retained for older call sites."""
        return self._score_attack(
            ben_pred_labels,
            adv_pred_labels,
            y_test_numeric,
        )

    def get_attack_subset(
        self,
        data: DataConfig,
        test: bool = True,
    ) -> tuple[int, MatrixLike | ArrayLike, ArrayLike | None]:
        """Return attack-size limited feature/label subset from train or test split.

        Args:
            data: Data runtime containing train/test splits.
            test: Use test split when True, otherwise train split.

        Returns:
            Subset size, feature payload, and label payload.

        Raises:
            ValueError: If selected split payload cannot be converted to a supported subset type.
        """
        n = self.attack_size
        if test is True:
            x_ = data.X_test
            y_ = data.y_test
        else:
            x_ = data.X_train
            y_ = data.y_train
        from torch.utils.data import Dataset, DataLoader, Subset

        # Accept Subset/Dataset and convert to tensor
        if isinstance(x_, (pd.Series, np.ndarray, pd.DataFrame)) or is_tensor(x_):
            x_subset = x_[:n]
            y_subset = y_[:n]
        elif isinstance(x_, (Dataset, Subset)):
            # Convert to tensor
            loader = DataLoader(x_, batch_size=n, shuffle=False)
            batch = next(iter(loader))
            if isinstance(batch, (tuple, list)):
                x_subset = batch[0]
                y_subset = batch[1]
            else:
                x_subset = batch
                y_subset = None
        elif is_dataloader(x_):
            x_subset, y_subset = collect_subset_from_dataloader(x_, n=n)
        else:
            raise ValueError(
                f"Expected data.X_test to be a pd.Series, np.ndarray, torch Tensor, torch DataLoader, or torch Dataset/Subset. Got: {type(data.X_test)}",
            )
        # Do not flatten x_subset; preserve original shape for torch/ART models
        if is_tensor(y_subset) and y_subset.ndim > 1:
            y_subset = y_subset.view(-1)
        return n, x_subset, y_subset
