# Standard library imports
import copy
import importlib
import pickle
import logging

from pathlib import Path
import pandas as pd

# Typing imports
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional, Union

# Sklearn and numpy imports
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import numpy as np

# ART imports
try:
    from art.config import ART_NUMPY_DTYPE
except Exception:
    # Keep non-ART workflows importable (for example score-only test scopes)
    # when optional ART/Torch stacks are unavailable or fail at import time.
    ART_NUMPY_DTYPE = np.float32

from omegaconf import DictConfig, OmegaConf

from ..artifacts import ScoreDict
from ..data import DataConfig
from ..model import ModelConfig
from ..frameworks.types import ArrayLike, EstimatorLike, MatrixLike
from ..model.defense.base import _get_art_symbols
from ..utils import (
    BaseConfig,
    is_default_config_value,
    is_null_config_value,
    instantiate_plugin_spec,
    load_class,
    normalize_plugin_specs,
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
from ..orchestration import ScoreOrchestratorMixin
from ..score.attack import AttackScorerConfig

logger = logging.getLogger(__name__)

AttackFamily = Literal[
    "evasion",
    "poisoning",
    "inference",
    "extraction",
    "reconstruction",
]
AttackSubFamily = str
PLUGIN_EVASION_ATTACK_NAMESPACES = frozenset({"textattack", "openattack"})


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


def _get_sklearn_dict() -> dict[str, Any]:
    return _get_art_symbols()["sklearn_dict"]


def _get_supported_models() -> tuple[type, ...]:
    return tuple(_get_sklearn_dict().values())


def _resolve_plugin_root_from_attack_name(attack_name: str) -> str | None:
    """Infer plugin root token from canonical attack name.

    Example:
        ``textattack.attack_recipes...`` -> ``textattack``
    """
    normalized_name = str(attack_name or "").strip().lower()
    if normalized_name == "":
        return None
    parts = normalized_name.split(".")
    if len(parts) < 2:
        return None
    root = str(parts[0]).strip()
    if root in {"", "deckard", "art"}:
        return None
    return root


def _resolve_plugin_attack_family(attack_name: str) -> tuple[str, str] | None:
    """Resolve first-party plugin family/sub-family tokens from attack name.

    Text-oriented third-party integrations currently expose evasion-style
    perturbation attacks only.
    """
    plugin_root = _resolve_plugin_root_from_attack_name(attack_name)
    if plugin_root in PLUGIN_EVASION_ATTACK_NAMESPACES:
        return "evasion", plugin_root
    return None


def _resolve_plugin_runtime_config_type(attack_name: str) -> type | None:
    """Resolve plugin runtime config class for a canonical attack declaration.

    This avoids hard-coded attack-name prefixes by resolving plugin runtime
    handlers from plugin module declarations at runtime.
    """
    plugin_root = _resolve_plugin_root_from_attack_name(attack_name)
    if plugin_root in {None, "", "art", "deckard"}:
        return None
    module_name = f"deckard.plugins.{plugin_root}.attack"
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None

    candidates: list[type] = []
    for value in vars(module).values():
        if not isinstance(value, type):
            continue
        if value is AttackConfig:
            continue
        if not issubclass(value, AttackConfig):
            continue
        if value.__module__ != module.__name__:
            continue
        candidates.append(value)
    if not candidates:
        return None
    candidates.sort(key=lambda cls: cls.__name__)
    return candidates[0]


@dataclass(eq=False, kw_only=True)
class AttackConfig(ScoreOrchestratorMixin, BaseConfig):
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
        name: Fully-qualified attack class path used to resolve attack class.
        attack_params: Constructor and runtime parameters for attack execution.
        attack_size: Number of samples used for attack execution.
        plugins: Runtime attack plugins used for dispatch/hook extension.
        scorer: Attack scorer configuration applied to attack outputs.
        score_dict: Runtime score payload collected during attack execution.
    """

    # Configuration fields
    name: str = "art.attacks.evasion.HopSkipJump"
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
    plugins: list = field(
        default_factory=list,
        metadata={"help": "Resolved attack plugins attached to this runtime config."},
    )
    device: Union[str, None] = None
    mode: Literal["auto", "train", "test", "val"] = "auto"

    # Runtime state fields
    attack_time: Union[float, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={
            "help": "Elapsed time in seconds for generating adversarial examples.",
        },
    )
    attack_prediction_time: Union[float, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={
            "help": "Elapsed time in seconds for model predictions on attacked inputs.",
        },
    )
    attack_score_time: Union[float, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Elapsed time in seconds for scoring the attack outputs."},
    )
    attack: Union[object, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Instantiated attack object used for the current runtime."},
    )
    attack_predictions: Union[object, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Predictions produced on attacked samples."},
    )
    attacked_labels: Union[object, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Labels or targets associated with attacked samples."},
    )
    score_y_pred: Union[object, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Prediction payload forwarded into attack scoring."},
    )
    score_y_proba: Union[object, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Probability payload forwarded into attack scoring."},
    )
    target_index: Union[int, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={
            "help": "Selected target-class index for targeted attack workflows.",
        },
    )
    _attack_family: Union[str, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Resolved canonical attack family for runtime dispatch."},
    )
    _attack_sub_family: Union[str, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Resolved canonical attack subtype for runtime dispatch."},
    )
    score_dict: ScoreDict = field(
        default_factory=ScoreDict,
        init=False,
        repr=False,
        metadata={
            "help": "Attack score payload accumulated during runtime evaluation.",
        },
    )
    _target_: Union[str, None] = field(
        default="deckard.attack.base.AttackConfig",
        init=True,
        repr=True,
        metadata={"help": "Hydra target path used to rehydrate this attack config."},
    )
    _plugin_objects: Union[list, None] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        metadata={
            "help": "Instantiated plugin objects cached for attack hook dispatch.",
        },
    )

    def __hash__(self):
        return super().__hash__()

    def __post_init__(self):
        """
        Initializes post-construction attributes for the class.

        Sets the internal attack attribute to None. If attack_params is not provided,
        initializes it as an empty dictionary.
        """
        attack_name = str(self.name).strip()
        if attack_name == "":
            raise ValueError("AttackConfig.name must be a non-empty attack class path")

        self.name = attack_name
        if self._target_ in {None, ""}:
            self._target_ = "deckard.attack.base.AttackConfig"
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
        attack_family = (self.attack_family or "").lower()
        if attack_family != "poisoning":
            return

        if str(self.resolve_name(default="") or "").endswith("PoisoningAttackSVM"):
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
        attack_sub_family: Optional[str] = None,
        split_override: Optional[str] = None,
    ) -> Literal["train", "test", "val"]:
        """Resolve active split mode from overrides, explicit mode, or auto defaults.

        Precedence order:
        1) Explicit split override (method arg or attack_params['split']).
        2) Explicit attack mode (train/test/val).
        3) Auto defaults inferred from attack family/subtype/kind.

        Args:
            attack_kind: Canonical attack kind token.
            attack_sub_family: Optional attack sub-family token.
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
        subtype = (attack_sub_family or self.attack_sub_family or "").lower()
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
        attack_path = str(self.resolve_name(default="") or "")
        plugin_family = _resolve_plugin_attack_family(attack_path)
        if plugin_family is not None:
            return plugin_family
        parts = attack_path.split("attacks.")[-1].split(".")
        attack_family = parts[0] if len(parts) > 0 else ""
        attack_sub_family = parts[1] if len(parts) > 1 else ""
        return attack_family, attack_sub_family

    def _attack_target_token(self) -> str:
        """Resolve a stable target token used in emitted attack metric labels."""
        attack_params = (
            self.attack_params if isinstance(self.attack_params, dict) else {}
        )
        target_token: Any = attack_params.get("class_target")

        if target_token in {None, ""}:
            target_token = self.targeted_attribute

        if target_token in {None, ""}:
            target_token = (
                "targeted"
                if bool(attack_params.get("targeted", False))
                else "untargeted"
            )

        normalized = "".join(
            ch if ch.isalnum() else "_" for ch in str(target_token).strip().lower()
        ).strip("_")
        return normalized or "untargeted"

    def _with_targeted_attack_labels(
        self,
        scores: Mapping[str, Any],
        attack_family: str,
    ) -> ScoreDict:
        """Add `<target>_evasion_<metric>` aliases for evasion and poisoning metrics."""
        payload = dict(scores)
        if attack_family not in {"evasion", "poisoning"}:
            return ScoreDict.from_payload(payload)

        target_token = self._attack_target_token()
        aliases: dict[str, Any] = {}
        for key, value in payload.items():
            token = str(key)
            metric: str | None = None
            if token.startswith("evasion_"):
                metric = token[len("evasion_") :]
            elif token.startswith("poisoned_"):
                metric = token[len("poisoned_") :]
            elif token.startswith("benign_"):
                metric = f"benign_{token[len('benign_') :]}"

            if metric is not None and metric != "":
                aliases[f"{target_token}_evasion_{metric}"] = value

        if aliases:
            payload.update(aliases)
        return ScoreDict.from_payload(payload)

    def _finalize_attack_state(
        self,
        *,
        attack: Any = None,
        attack_predictions: Any = None,
        attacked_labels: Any = None,
        score_dict: Any = None,
        score_y_pred: Any = None,
        score_y_proba: Any = None,
    ) -> ScoreDict:
        """Write common runtime attack outputs to this config instance."""
        if attack is not None:
            self.attack = attack
        if attack_predictions is not None:
            self.attack_predictions = attack_predictions
        if attacked_labels is not None:
            self.attacked_labels = attacked_labels
        if score_y_pred is not None:
            self.score_y_pred = score_y_pred
        if score_y_proba is not None:
            self.score_y_proba = score_y_proba
        if score_dict is not None:
            payload = score_dict
            if isinstance(payload, ScoreDict):
                payload = dict(payload)
            self.score_dict = ScoreDict.from_payload({**self.score_dict, **payload})
        return ScoreDict.from_payload(self.score_dict)

    def _combine_attack_scores(
        self,
        *,
        benign_scores: Any,
        attack_scores: Any,
        attack_kind: str | None = None,
        extra_scores: Any = None,
    ) -> ScoreDict:
        """Merge shared benign and attack score payloads."""
        merged_payload: dict[str, Any] = {}
        if benign_scores:
            merged_payload.update(dict(benign_scores))
        if attack_scores:
            merged_payload.update(dict(attack_scores))
        if extra_scores:
            merged_payload.update(dict(extra_scores))
        merged_scores = ScoreDict.from_payload(merged_payload)
        if attack_kind:
            merged_scores = self._with_targeted_attack_labels(
                merged_scores,
                attack_kind,
            )
        return merged_scores

    def _dispatch_attack_scores(
        self,
        *,
        benign_scores: Any,
        attack_scores: Any,
        attack_kind: str | None = None,
        extra_scores: Any = None,
    ) -> ScoreDict:
        """Shared scoring dispatch for attack family modes."""
        return self._combine_attack_scores(
            benign_scores=benign_scores,
            attack_scores=attack_scores,
            attack_kind=attack_kind,
            extra_scores=extra_scores,
        )

    def _instantiate_plugin(self, plugin_spec: Any):
        """Instantiate one attack plugin specification.

        Args:
            plugin_spec: Plugin declaration payload or runtime plugin object.

        Returns:
            Instantiated plugin object.
        """
        return instantiate_plugin_spec(
            plugin_spec,
            loader=load_class,
        )

    def _get_plugins(self) -> list:
        """Resolve and cache attack plugins for this config instance.

        Returns:
            Ordered list of instantiated plugin objects.
        """
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
        """Merge plugin hook outputs into the runtime score dictionary.

        Args:
            hook_outputs: Iterable of plugin hook return payloads.
        """
        if self.score_dict is None:
            self.score_dict = ScoreDict()
        for output in hook_outputs:
            if isinstance(output, dict):
                self.score_dict.update(ScoreDict.from_payload(output))

    def _resolve_runtime_attack_config(
        self,
        attack_family: str,
        attack_sub_family: str,
    ) -> tuple[type, ...]:
        mixins: list[type] = []
        attack_name = str(self.resolve_name(default="") or "")
        attack_family_lower = (attack_family or "").lower()
        attack_sub_family_lower = (attack_sub_family or "").lower()
        plugin_root = _resolve_plugin_root_from_attack_name(attack_name)

        plugin_runtime_config = _resolve_plugin_runtime_config_type(attack_name)
        if plugin_runtime_config is not None:
            mixins.append(plugin_runtime_config)
        elif plugin_root not in {None, "", "art", "deckard"}:
            raise ValueError(
                "No plugin runtime config is registered for attack namespace "
                f"'{plugin_root}'. Expected a deckard plugin runtime module at "
                f"'deckard.plugins.{plugin_root}.attack'.",
            )
        elif attack_family_lower == "evasion":
            from .evasion import EvasionAttackConfig

            mixins.append(EvasionAttackConfig)
        elif attack_family_lower == "poisoning":
            from .poisoning import PoisoningAttackConfig

            mixins.append(PoisoningAttackConfig)
        elif attack_family_lower == "extraction":
            from .extraction import ExtractionAttackConfig

            mixins.append(ExtractionAttackConfig)
        elif attack_family_lower == "inference":
            if attack_sub_family_lower == "reconstruction":
                from .reconstruction import ReconstructionAttackConfig

                mixins.append(ReconstructionAttackConfig)
            else:
                from .inference import InferenceAttackConfig

                mixins.append(InferenceAttackConfig)

        plugin_outputs = self._run_plugin_hook(
            "resolve_attack_mixins",
            attack_family=attack_family,
            attack_sub_family=attack_sub_family,
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

    def _resolve_attack_handler(
        self,
        attack_family: str,
        attack_sub_family: str,
    ):
        mixins = self._resolve_runtime_attack_config(
            attack_family,
            attack_sub_family,
        )
        default_handler = None
        for mixin in mixins:
            if not isinstance(mixin, type):
                continue
            if mixin in type(self).mro():
                default_handler = self
                break
            if issubclass(mixin, AttackConfig):
                # Config-first dispatch: resolve one concrete runtime config class
                # instead of composing multiple *Config types into a synthetic class.
                runtime_handler = copy.copy(self)
                runtime_handler.__class__ = mixin
                default_handler = runtime_handler
                break

        hook_outputs = self._run_plugin_hook(
            "resolve_attack_handler",
            attack_family=attack_family,
            attack_sub_family=attack_sub_family,
            default_handler=default_handler,
            default_mixins=mixins,
        )
        for output in hook_outputs:
            if callable(output):
                return output

        return default_handler

    def _with_attack_context(
        self,
        *,
        attack_family: str,
        attack_sub_family: str,
    ) -> "AttackConfig":
        """Return an attack runtime view bound to the resolved family context."""
        runtime: AttackConfig = self
        runtime._attack_family = attack_family
        runtime._attack_sub_family = attack_sub_family

        for mixin in runtime._resolve_runtime_attack_config(
            attack_family=attack_family,
            attack_sub_family=attack_sub_family,
        ):
            if not isinstance(mixin, type):
                continue
            if mixin in type(runtime).mro():
                break
            if issubclass(mixin, AttackConfig):
                # Preserve config-first behavior without mutating this instance type.
                runtime = copy.copy(runtime)
                runtime.__class__ = mixin
                runtime._attack_family = attack_family
                runtime._attack_sub_family = attack_sub_family
                break

        return runtime

    def resolve_runtime_attack_config(
        self,
        attack_family: str,
        attack_sub_family: str,
    ) -> tuple[type, ...]:
        """Public wrapper for config-first runtime attack config resolution.

        Args:
            attack_family: Canonical attack family token.
            attack_sub_family: Canonical attack subtype token.

        Returns:
            Tuple of runtime mixin/config classes used for attack dispatch.
        """
        return self._resolve_runtime_attack_config(
            attack_family=attack_family,
            attack_sub_family=attack_sub_family,
        )

    def resolve_runtime_attack_handler(
        self,
        attack_family: str,
        attack_sub_family: str,
    ) -> Callable[..., ScoreDict] | None:
        """Public wrapper for config-first runtime attack handler resolution.

        Args:
            attack_family: Canonical attack family token.
            attack_sub_family: Canonical attack subtype token.

        Returns:
            Callable runtime handler when available, else None.
        """
        return self._resolve_attack_handler(
            attack_family=attack_family,
            attack_sub_family=attack_sub_family,
        )

    def run(
        self,
        data: DataConfig,
        model: ModelConfig | BaseEstimator | EstimatorLike,
        files: dict[str, str | None] | None = None,
        attack_file: Union[str, None] = None,
        attack_predictions_file: Union[str, None] = None,
        score_file: Union[str, None] = None,
    ) -> ScoreDict:
        """Public execution alias for attack runtime orchestration.

        Args:
            data: Runtime data config used by the attack flow.
            model: Runtime model payload or model config.
            files: Optional runtime file payload overrides.
            attack_file: Optional attack artifact path override.
            attack_predictions_file: Optional attack predictions artifact path.
            score_file: Optional score artifact path override.

        Returns:
            Attack score payload generated by runtime orchestration.
        """
        return self(
            data=data,
            model=model,
            files=files,
            attack_file=attack_file,
            attack_predictions_file=attack_predictions_file,
            score_file=score_file,
        )

    def load(
        self,
        attack_file: str | None = None,
        attack_predictions_file: str | None = None,
    ) -> "AttackConfig":
        """Load cached runtime attack artifacts onto this config object.

        Args:
            attack_file: Optional attack artifact file path.
            attack_predictions_file: Optional predictions artifact path.

        Returns:
            This attack config instance for fluent chaining.
        """
        self.load_cached_attack_artifacts(
            attack_file=attack_file,
            attack_predictions_file=attack_predictions_file,
        )
        return self

    def score(
        self,
        *,
        attack_kind: str,
        y_true: Any,
        y_pred: Any = None,
        **kwargs: Any,
    ) -> ScoreDict:
        """Public scoring wrapper around the configured attack scorer.

        Args:
            attack_kind: Normalized attack scoring kind token.
            y_true: Ground-truth labels/targets.
            y_pred: Predicted labels/targets.
            **kwargs: Additional scorer kwargs.

        Returns:
            Score dictionary emitted by the configured scorer payload.
        """
        return self._score(
            attack_kind=attack_kind,
            y_true=y_true,
            y_pred=y_pred,
            **kwargs,
        )

    @property
    def attack_family(self) -> Optional[str]:
        """Return canonical attack family resolved from runtime attack declaration.

        Returns:
            Canonical attack family token when available.
        """
        if self._attack_family:
            return self._attack_family
        attack_family, _ = self._parse_attack_path()
        return attack_family or None

    @attack_family.setter
    def attack_family(self, value: Optional[str]) -> None:
        """Set cached runtime attack family token.

        Args:
            value: Attack family token to cache.
        """
        self._attack_family = value

    @property
    def attack_sub_family(self) -> Optional[str]:
        """Return canonical attack subtype resolved from runtime attack declaration.

        Returns:
            Canonical attack subtype token when available.
        """
        if self._attack_sub_family:
            return self._attack_sub_family
        _, attack_sub_family = self._parse_attack_path()
        return attack_sub_family or None

    @property
    def attack_kind(self) -> Optional[str]:
        """Return normalized scoring attack kind token for the configured attack.

        Returns:
            Normalized attack scoring kind token.
        """
        attack_family = (self.attack_family or "").lower()
        subtype = (self.attack_sub_family or "").lower()

        if attack_family == "evasion":
            return "evasion"
        if attack_family == "inference" and "membership" in subtype:
            return "membership"
        if attack_family == "inference" and "attribute" in subtype:
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
        attack_family = (self.attack_family or "").lower()
        task_is_classification = self._infer_task_is_classification(data, model)
        if attack_family == "evasion" and task_is_classification is False:
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
        attack_family = self.attack_family or ""
        attack_sub_family = self.attack_sub_family or ""
        attack_name = self.resolve_name(default=None)
        if attack_name is None or str(attack_name).strip() == "":
            raise ValueError(
                "AttackConfig.name must be set before attack initialization",
            )
        attack_name = str(attack_name)
        self.name = attack_name

        plugin_family = _resolve_plugin_attack_family(attack_name)
        if plugin_family is not None:
            attack_family, attack_sub_family = plugin_family
            self._attack_family = attack_family
            self._attack_sub_family = attack_sub_family
            # TextAttack/OpenAttack runtime handlers construct and execute
            # concrete attack objects directly from canonical plugin names.
            return None, None, attack_family, attack_sub_family

        art_model = None
        if isinstance(model, ModelConfig):
            runtime_model = getattr(model, "_model", None)
            if attack_family == "extraction" and is_torch_model(runtime_model):
                # Extraction expects a neural-network ART classifier; build directly
                # from the underlying torch module when available.
                art_model = build_torch_art_model(model=runtime_model, data=data)
            else:
                art_model = model.get_art_model(data)
        elif is_torch_model(model):
            art_model = build_torch_art_model(model=model, data=data)
        else:
            check_is_fitted(model)

        # Validate attack family
        if attack_family not in [
            "evasion",
            "poisoning",
            "extraction",
            "inference",
        ]:
            raise ValueError(f"Unsupported attack family: {attack_family}")

        if attack_family == "poisoning":
            self._validate_poisoning_params()

        attack_class = resolve_class(attack_name)
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
        if attack_family == "poisoning":
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
        if attack_family == "inference" and attack_sub_family == "model_inversion":
            for key in (
                "split",
                "targets",
                "initialization",
                "x_init",
            ):
                attack_init_params.pop(key, None)
        if attack_family == "inference" and attack_sub_family == "reconstruction":
            for key in (
                "split",
                "missing_index",
            ):
                attack_init_params.pop(key, None)
        attack = attack_class(art_model, **attack_init_params)
        self._attack_family = attack_family
        self._attack_sub_family = attack_sub_family
        return attack, art_model, attack_family, attack_sub_family

    def initialize_attack(
        self,
        model: ModelConfig | BaseEstimator | EstimatorLike,
        data: DataConfig,
    ) -> tuple[BaseEstimator | EstimatorLike | None, EstimatorLike | None, str, str]:
        """Public entry-point for attack initialisation. Delegates to _initialize_attack().

        Args:
            model: Runtime model payload.
            data: Runtime data payload.

        Returns:
            Initialized attack, ART model, attack family, and attack subtype.

            For plugin-managed text attacks (for example TextAttack/OpenAttack),
            attack and ART model placeholders may be ``None`` because concrete
            runtime handlers construct their own execution objects.
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

    def _sync_runtime_state_from(self, runtime: "AttackConfig") -> None:
        """Copy runtime attack outputs from a resolved handler config to this config."""
        runtime_fields = (
            "attack_time",
            "attack_prediction_time",
            "attack_score_time",
            "attack",
            "attack_predictions",
            "attacked_labels",
            "score_y_pred",
            "score_y_proba",
            "score_dict",
            "_attack_family",
            "_attack_sub_family",
        )
        for field_name in runtime_fields:
            setattr(self, field_name, getattr(runtime, field_name))

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
            ValueError: If attack family/sub-family/model wiring is unsupported.
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

        attack, art_model, attack_family, attack_sub_family = self._initialize_attack(
            model,
            data,
        )
        runtime = self._with_attack_context(
            attack_family=attack_family,
            attack_sub_family=attack_sub_family,
        )
        handler = runtime._resolve_attack_handler(
            attack_family=attack_family,
            attack_sub_family=attack_sub_family,
        )
        if handler is None:
            raise NotImplementedError(
                f"Attack type {attack_family} subtype {attack_sub_family} has no registered runtime handler.",
            )

        before_outputs = runtime._run_plugin_hook(
            "before_attack_dispatch",
            data=data,
            model=model,
            attack=attack,
            art_model=art_model,
            attack_family=attack_family,
            attack_sub_family=attack_sub_family,
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
            attack_family=attack_family,
            attack_sub_family=attack_sub_family,
            execution_order=attack_execution_order,
        )

        scores = handler(
            data=data,
            model=model,
            art_model=art_model,
            attack=attack,
            attack_family=attack_family,
            attack_sub_family=attack_sub_family,
        )
        runtime._run_attack_stage_hooks(
            "after",
            "post-attack",
            data=data,
            model=model,
            attack=attack,
            art_model=art_model,
            attack_family=attack_family,
            attack_sub_family=attack_sub_family,
            execution_order=attack_execution_order,
        )
        after_outputs = runtime._run_plugin_hook(
            "after_attack_dispatch",
            data=data,
            model=model,
            attack=attack,
            art_model=art_model,
            attack_family=attack_family,
            attack_sub_family=attack_sub_family,
            scores=scores,
        )
        runtime._merge_plugin_scores(after_outputs)
        assert isinstance(scores, dict), "Scores should be a dictionary"
        assert isinstance(
            runtime.attack_time,
            float,
        ), f"Attack time should be a float, got {type(runtime.attack_time)}"
        assert isinstance(
            runtime.attack_prediction_time,
            float,
        ), "Attack prediction time should be a float"
        assert isinstance(
            runtime.attack_score_time,
            float,
        ), "Attack score time should be a float"
        times = {
            "attack_generation_time": runtime.attack_time,
            "attack_prediction_time": runtime.attack_prediction_time,
            "attack_score_time": runtime.attack_score_time,
        }
        score_dict = {
            **scores,
            **times,
            "attack_stage": normalize_attack_stage("post-attack"),
            "attack_execution_order": attack_execution_order,
        }
        runtime.score_dict = ScoreDict.from_payload(score_dict)
        self._sync_runtime_state_from(runtime)

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
                attack_sub_family=self.attack_sub_family,
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
            return tensor_to_numpy(x_subset)
        try:
            from torch.utils.data import DataLoader, Dataset, Subset
        except ImportError:  # pragma: no cover
            DataLoader = Dataset = Subset = ()
        if isinstance(value, (Dataset, Subset)) and not isinstance(value, DataLoader):
            loader = DataLoader(value, batch_size=len(value), shuffle=False)
            batch = next(iter(loader))
            features = batch[0] if isinstance(batch, (tuple, list)) else batch
            return tensor_to_numpy(features)
        if is_tensor(value):
            return tensor_to_numpy(value)
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
