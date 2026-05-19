"""Abstract framework contracts for Deckard framework-specific configs.

These contracts define a consistent lifecycle for framework-backed config
objects:

- every config is a dataclass
- ``__post_init__`` runs declared private lifecycle steps in order
- ``__call__`` runs declared public lifecycle steps in order
- loading, persistence, and context-aware behavior are reusable mixins
"""

from __future__ import annotations

import importlib
import inspect

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Sequence

if TYPE_CHECKING:
    from ..attack.base import AttackConfig
    from ..data.base import DataConfig
    from ..data.sample import BaseSampler
    from ..model.base import ModelConfig


class RuntimeValue(Protocol):
    """Marker protocol for framework runtime payloads."""


class MatrixLike(Protocol):
    """Structural protocol for matrix-like payloads.

    Examples include DataFrame, ndarray, tensors, datasets, and dataloaders.
    """

    def __len__(self) -> int:
        """Return row or batch count when available."""

    def __iter__(self) -> object:
        """Yield rows, batches, or records."""


class ArrayLike(Protocol):
    """Structural protocol for array-like payloads.

    Examples include ndarray, list, tensor, datasets, and dataloaders.
    """

    def __len__(self) -> int:
        """Return element count."""

    def __iter__(self) -> object:
        """Yield elements, batches, or records."""


class EstimatorLike(Protocol):
    """Structural protocol for framework estimator runtime objects.

    Examples include sklearn estimators, torch modules, and ART wrappers.
    """

    def __len__(self) -> int:
        """Return size metadata when available."""


def _resolve_runtime_type(module_name: str, qualname: str) -> type[object] | None:
    """Resolve an optional runtime class from an installed dependency."""
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None

    value: object = module
    for part in qualname.split("."):
        if not hasattr(value, part):
            return None
        value = getattr(value, part)
    if isinstance(value, type):
        return value
    return None


def _collect_runtime_types(specs: tuple[tuple[str, str], ...]) -> list[type[object]]:
    """Collect installed runtime classes for payload checks."""
    resolved: list[type[object]] = []
    for module_name, qualname in specs:
        runtime_type = _resolve_runtime_type(module_name, qualname)
        if runtime_type is not None and runtime_type not in resolved:
            resolved.append(runtime_type)
    return resolved


_MATRIX_RUNTIME_TYPE_SPECS: tuple[tuple[str, str], ...] = (
    ("numpy", "ndarray"),
    ("pandas", "DataFrame"),
    ("torch", "Tensor"),
    ("torch.utils.data", "DataLoader"),
    ("torch.utils.data", "Dataset"),
    ("torch.utils.data", "Subset"),
)

_ARRAY_RUNTIME_TYPE_SPECS: tuple[tuple[str, str], ...] = (
    ("numpy", "ndarray"),
    ("pandas", "Series"),
    ("torch", "Tensor"),
    ("torch.utils.data", "DataLoader"),
    ("torch.utils.data", "Dataset"),
    ("torch.utils.data", "Subset"),
)

_ESTIMATOR_RUNTIME_TYPE_SPECS: tuple[tuple[str, str], ...] = (
    ("sklearn.base", "BaseEstimator"),
    ("torch.nn", "Module"),
    ("art.estimators.classification.scikitlearn", "ScikitlearnClassifier"),
    ("art.estimators.regression.scikitlearn", "ScikitlearnRegressor"),
    ("art.estimators.classification", "PyTorchClassifier"),
    ("art.estimators.regression", "PyTorchRegressor"),
)

_MATRIX_RUNTIME_TYPES: list[type[object]] = _collect_runtime_types(
    _MATRIX_RUNTIME_TYPE_SPECS,
)
_ARRAY_RUNTIME_TYPES: list[type[object]] = _collect_runtime_types(
    _ARRAY_RUNTIME_TYPE_SPECS,
)
_ESTIMATOR_RUNTIME_TYPES: list[type[object]] = _collect_runtime_types(
    _ESTIMATOR_RUNTIME_TYPE_SPECS,
)


def register_matrix_like_type(runtime_type: type[object]) -> None:
    """Register a framework-specific matrix-like runtime type."""
    if runtime_type not in _MATRIX_RUNTIME_TYPES:
        _MATRIX_RUNTIME_TYPES.append(runtime_type)


def register_array_like_type(runtime_type: type[object]) -> None:
    """Register a framework-specific array-like runtime type."""
    if runtime_type not in _ARRAY_RUNTIME_TYPES:
        _ARRAY_RUNTIME_TYPES.append(runtime_type)


def register_estimator_like_type(runtime_type: type[object]) -> None:
    """Register a framework-specific estimator-like runtime type."""
    if runtime_type not in _ESTIMATOR_RUNTIME_TYPES:
        _ESTIMATOR_RUNTIME_TYPES.append(runtime_type)


def get_matrix_like_types() -> tuple[type[object], ...]:
    """Return currently configured matrix-like runtime classes."""
    return tuple(_MATRIX_RUNTIME_TYPES)


def get_array_like_types() -> tuple[type[object], ...]:
    """Return currently configured array-like runtime classes."""
    return tuple(_ARRAY_RUNTIME_TYPES)


def get_estimator_like_types() -> tuple[type[object], ...]:
    """Return currently configured estimator-like runtime classes."""
    return tuple(_ESTIMATOR_RUNTIME_TYPES)


def is_matrix_like(value: object) -> bool:
    """Return True when *value* matches configured matrix-like types."""
    configured = get_matrix_like_types()
    if configured and isinstance(value, configured):
        return True
    return hasattr(value, "__len__") and hasattr(value, "__iter__")


def is_array_like(value: object) -> bool:
    """Return True when *value* matches configured array-like types."""
    configured = get_array_like_types()
    if configured and isinstance(value, configured):
        return True
    return hasattr(value, "__len__") and hasattr(value, "__iter__")


def is_estimator_like(value: object) -> bool:
    """Return True when *value* matches configured estimator-like types."""
    configured = get_estimator_like_types()
    if configured and isinstance(value, configured):
        return True
    return hasattr(value, "fit") or hasattr(value, "predict")


LifecycleResults = dict[str, RuntimeValue]
LifecycleStepNames = tuple[str, ...]


@dataclass(eq=False, kw_only=True)
class DeclarativeConfigContract(ABC):
    """Base lifecycle contract for core config abstractions."""

    @classmethod
    def post_init_steps(cls) -> tuple[str, ...]:
        """Return ordered private methods executed by ``__post_init__``."""
        return ()

    @classmethod
    def execution_steps(cls) -> tuple[str, ...]:
        """Return ordered public methods executed by ``__call__``."""
        return ()

    def __post_init__(self) -> None:
        self._run_private_lifecycle_steps(type(self).post_init_steps())

    def run_declared_execution(self, *args: Any, **kwargs: Any) -> LifecycleResults:
        """Execute the declared public lifecycle steps in order."""
        return self._run_public_lifecycle_steps(
            type(self).execution_steps(),
            *args,
            **kwargs,
        )

    def _run_private_lifecycle_steps(self, step_names: Sequence[str]) -> None:
        self._validate_declared_step_names(step_names, phase="post_init")
        for step_name in step_names:
            step = self._resolve_declared_step(step_name, require_private=True)
            self._invoke_declared_step(step)

    def _run_public_lifecycle_steps(
        self,
        step_names: Sequence[str],
        *args: Any,
        **kwargs: Any,
    ) -> LifecycleResults:
        self._validate_declared_step_names(step_names, phase="execution")
        results: LifecycleResults = {}
        for step_name in step_names:
            step = self._resolve_declared_step(step_name, require_private=False)
            results[step_name] = self._invoke_declared_step(step, *args, **kwargs)
        return results

    @staticmethod
    def _validate_declared_step_names(
        step_names: Sequence[str],
        *,
        phase: str,
    ) -> None:
        duplicate_steps = {
            step_name for step_name in step_names if step_names.count(step_name) > 1
        }
        if duplicate_steps:
            duplicates = ", ".join(sorted(duplicate_steps))
            raise ValueError(
                f"Duplicate lifecycle step declarations in {phase}: {duplicates}",
            )

    def _resolve_declared_step(
        self,
        step_name: str,
        *,
        require_private: bool,
    ) -> Any:
        is_private = step_name.startswith("_")
        if require_private and not is_private:
            raise ValueError(
                f"Post-init step '{step_name}' must be a private method.",
            )
        if not require_private and is_private:
            raise ValueError(
                f"Execution step '{step_name}' must be a public method.",
            )

        step = getattr(self, step_name, None)
        if not callable(step):
            raise AttributeError(
                f"Declared lifecycle step '{step_name}' is not callable on {type(self).__name__}.",
            )
        return step

    @staticmethod
    def _invoke_declared_step(step: Any, *args: Any, **kwargs: Any) -> Any:
        signature = inspect.signature(step)
        parameters = tuple(signature.parameters.values())

        accepts_var_positional = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters
        )
        accepts_var_keyword = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )

        if accepts_var_positional:
            call_args = args
        else:
            positional_slots = sum(
                parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                for parameter in parameters
            )
            call_args = args[:positional_slots]

        if accepts_var_keyword:
            call_kwargs = kwargs
        else:
            accepted_keywords = {
                parameter.name
                for parameter in parameters
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
            call_kwargs = {
                key: value for key, value in kwargs.items() if key in accepted_keywords
            }

        return step(*call_args, **call_kwargs)


@dataclass(eq=False, kw_only=True)
class LoadableConfigMixin(DeclarativeConfigContract, ABC):
    """Reusable contract for configs that load defaults, disk state, or weights."""

    @classmethod
    def loading_post_init_steps(cls) -> tuple[str, ...]:
        """Return private loading hooks executed during ``__post_init__``."""
        return ("_initialize_loading_behavior",)

    @classmethod
    def loading_steps(cls) -> tuple[str, ...]:
        """Return public loading lifecycle steps for this contract."""
        return ("load_defaults", "load", "load_cached")

    def _initialize_loading_behavior(self) -> None:
        """Prepare loading-related runtime state before execution."""

    @abstractmethod
    def load_defaults(self) -> RuntimeValue:
        """Resolve default config state for this runtime."""

    @abstractmethod
    def load(self) -> RuntimeValue:
        """Load persisted config or artifact state from disk."""

    @abstractmethod
    def load_cached(self) -> RuntimeValue:
        """Load pre-trained weights or precomputed state when available."""


@dataclass(eq=False, kw_only=True)
class PersistableConfigMixin(DeclarativeConfigContract, ABC):
    """Reusable contract for configs that persist runtime outputs."""

    @classmethod
    def persistence_post_init_steps(cls) -> tuple[str, ...]:
        """Return private persistence hooks executed during ``__post_init__``."""
        return ("_initialize_persistence_behavior",)

    @classmethod
    def persistence_steps(cls) -> tuple[str, ...]:
        """Return public persistence lifecycle steps for this contract."""
        return ("save",)

    def _initialize_persistence_behavior(self) -> None:
        """Prepare persistence-related runtime state before execution."""

    @abstractmethod
    def save(self) -> RuntimeValue:
        """Persist runtime outputs or artifacts for this config."""


@dataclass(eq=False, kw_only=True)
class ScoreableConfigMixin(DeclarativeConfigContract, ABC):
    """Reusable contract for configs that expose scoring behavior."""

    @classmethod
    def scoring_post_init_steps(cls) -> tuple[str, ...]:
        """Return private scoring hooks executed during ``__post_init__``."""
        return ("_initialize_scoring_behavior",)

    @classmethod
    def scoring_steps(cls) -> tuple[str, ...]:
        """Return public scoring lifecycle steps for this contract."""
        return ("score",)

    def _initialize_scoring_behavior(self) -> None:
        """Prepare scoring-related runtime state before execution."""

    @abstractmethod
    def score(self, *args: Any, **kwargs: Any) -> dict[str, RuntimeValue]:
        """Compute score or metric output for this config.

        Args:
            *args: Positional runtime values required by the concrete scorer.
            **kwargs: Keyword runtime values required by the concrete scorer.

        Returns:
            Score or metric outputs keyed by metric name.
        """


@dataclass(eq=False, kw_only=True)
class ContextAwareConfigMixin(DeclarativeConfigContract, ABC):
    """Reusable contract for configs that derive behavior from runtime context."""

    @classmethod
    def context_post_init_steps(cls) -> tuple[str, ...]:
        """Return private context hooks executed during ``__post_init__``."""
        return ("_initialize_context_behavior",)

    @classmethod
    def context_steps(cls) -> tuple[str, ...]:
        """Return public context lifecycle steps for this contract."""
        return ("resolve_context",)

    def _initialize_context_behavior(self) -> None:
        """Prepare context-resolution state before execution."""

    @abstractmethod
    def resolve_context(self, **context: RuntimeValue) -> dict[str, RuntimeValue]:
        """Resolve context-aware runtime state from external inputs.

        Args:
            **context: External runtime values used to derive context state.

        Returns:
            Context-derived runtime values keyed by context name.
        """


@dataclass(eq=False, kw_only=True)
class FrameworkDataConfig(
    LoadableConfigMixin,
    PersistableConfigMixin,
    ScoreableConfigMixin,
    ContextAwareConfigMixin,
    ABC,
):
    """Framework-level data config contract."""

    @classmethod
    def post_init_steps(cls) -> tuple[str, ...]:
        """Return private data-contract hooks executed during ``__post_init__``."""
        return (
            *cls.loading_post_init_steps(),
            *cls.persistence_post_init_steps(),
            *cls.scoring_post_init_steps(),
            *cls.context_post_init_steps(),
            "_validate_data_contract",
        )

    @classmethod
    def execution_steps(cls) -> tuple[str, ...]:
        """Return ordered public lifecycle steps for data configs."""
        return (
            *cls.loading_steps(),
            *cls.context_steps(),
            "load_data",
            "sample_data",
            "fit_presample",
            "fit_X",
            "fit_y",
            "fit_Xy",
            *cls.scoring_steps(),
            *cls.persistence_steps(),
        )

    def _validate_data_contract(self) -> None:
        """Validate that the data contract can execute in the declared order."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> LifecycleResults:
        """Execute the declared data-config lifecycle.

        Args:
            *args: Positional runtime values passed through to execution steps.
            **kwargs: Keyword runtime values passed through to execution steps.

        Returns:
            Lifecycle step outputs keyed by step name.
        """

    @abstractmethod
    def load_data(self) -> tuple[MatrixLike, ArrayLike]:
        """Return feature and target objects for the configured dataset."""

    @abstractmethod
    def sample_data(
        self,
        X: MatrixLike,
        y: ArrayLike,
    ) -> tuple[MatrixLike, MatrixLike, ArrayLike, ArrayLike]:
        """Return train/test and optional validation splits.

        Args:
            X: Feature payload to split.
            y: Target payload aligned with ``X``.

        Returns:
            Train features, test features, train targets, and test targets.
        """

    @abstractmethod
    def fit_Xy(self, X: MatrixLike, y: ArrayLike) -> tuple[MatrixLike, ArrayLike]:
        """Fit a transformer on a dataset and its targets.

        Args:
            X: Feature payload to fit.
            y: Target payload aligned with ``X``.

        Returns:
            Transformed features and aligned targets.
        """

    @abstractmethod
    def fit_y(self, X: MatrixLike, y: ArrayLike) -> tuple[MatrixLike, ArrayLike]:
        """Fit a transformer on targets.

        Args:
            X: Feature payload carried through the operation.
            y: Target payload to transform.

        Returns:
            Features and transformed targets.
        """

    @abstractmethod
    def fit_X(self, X: MatrixLike, y: ArrayLike) -> tuple[MatrixLike, ArrayLike]:
        """Fit a transformer on a dataset without mutating targets.

        Args:
            X: Feature payload to transform.
            y: Target payload carried through the operation.

        Returns:
            Transformed features and unchanged targets.
        """

    @abstractmethod
    def fit_presample(
        self,
        X: MatrixLike,
        y: ArrayLike,
    ) -> tuple[MatrixLike, ArrayLike]:
        """Run any fit step that must happen before sampling.

        Args:
            X: Feature payload to prepare before splitting.
            y: Target payload to prepare before splitting.

        Returns:
            Prepared features and targets.
        """


@dataclass(eq=False, kw_only=True)
class FrameworkModelConfig(
    LoadableConfigMixin,
    PersistableConfigMixin,
    ScoreableConfigMixin,
    ContextAwareConfigMixin,
    ABC,
):
    """Framework-level model config contract."""

    @classmethod
    def post_init_steps(cls) -> tuple[str, ...]:
        """Return private model-contract hooks executed during ``__post_init__``."""
        return (
            *cls.loading_post_init_steps(),
            *cls.persistence_post_init_steps(),
            *cls.context_post_init_steps(),
            *cls.scoring_post_init_steps(),
            "_validate_model_contract",
        )

    @classmethod
    def execution_steps(cls) -> tuple[str, ...]:
        """Return ordered public lifecycle steps for model configs."""
        return (
            *cls.loading_steps(),
            *cls.context_steps(),
            "init_model",
            "fit_model",  # if pretrained, set this to None
            *cls.scoring_steps(),
            *cls.persistence_steps(),
        )

    def _validate_model_contract(self) -> None:
        """Validate that the model contract can execute in the declared order."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> LifecycleResults:
        """Execute the declared model-config lifecycle.

        Args:
            *args: Positional runtime values passed through to execution steps.
            **kwargs: Keyword runtime values passed through to execution steps.

        Returns:
            Lifecycle step outputs keyed by step name.
        """

    @abstractmethod
    def init_model(self, data: "DataConfig") -> RuntimeValue:
        """Construct a model instance using data-derived context.

        Args:
            data: Runtime data context used to construct the model.

        Returns:
            Model runtime object.
        """

    @abstractmethod
    def fit_model(self, data: "DataConfig") -> RuntimeValue:
        """Train and return the model runtime object.

        Args:
            data: Runtime data context used for training.

        Returns:
            Trained model runtime object.
        """


@dataclass(eq=False, kw_only=True)
class FrameworkAttackConfig(
    LoadableConfigMixin,
    PersistableConfigMixin,
    ScoreableConfigMixin,
    ContextAwareConfigMixin,
    ABC,
):
    """Framework-level attack config contract."""

    @classmethod
    def post_init_steps(cls) -> tuple[str, ...]:
        """Return private attack-contract hooks executed during ``__post_init__``."""
        return (
            *cls.loading_post_init_steps(),
            *cls.persistence_post_init_steps(),
            *cls.scoring_post_init_steps(),
            *cls.context_post_init_steps(),
            "_validate_attack_contract",
        )

    @classmethod
    def execution_steps(cls) -> tuple[str, ...]:
        """Return ordered public lifecycle steps for attack configs."""
        return (
            *cls.loading_steps(),
            *cls.context_steps(),
            "build_attack",
            *cls.scoring_steps(),
            *cls.persistence_steps(),
        )

    def _validate_attack_contract(self) -> None:
        """Validate that the attack contract can execute in the declared order."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> LifecycleResults:
        """Execute the declared attack-config lifecycle.

        Args:
            *args: Positional runtime values passed through to execution steps.
            **kwargs: Keyword runtime values passed through to execution steps.

        Returns:
            Lifecycle step outputs keyed by step name.
        """

    @abstractmethod
    def build_attack(self, model: "ModelConfig", data: "DataConfig") -> RuntimeValue:
        """Construct attack runtime using model and data context.

        Args:
            model: Runtime model object targeted by the attack.
            data: Runtime data context used to configure the attack.

        Returns:
            Attack runtime object.
        """


@dataclass(eq=False, kw_only=True)
class FrameworkDetectorConfig(
    LoadableConfigMixin,
    PersistableConfigMixin,
    ScoreableConfigMixin,
    ContextAwareConfigMixin,
    ABC,
):
    """Framework-level detector config contract."""

    @classmethod
    def post_init_steps(cls) -> tuple[str, ...]:
        """Return private detector-contract hooks executed during ``__post_init__``."""
        return (
            *cls.loading_post_init_steps(),
            *cls.persistence_post_init_steps(),
            *cls.scoring_post_init_steps(),
            *cls.context_post_init_steps(),
            "_validate_detector_contract",
        )

    @classmethod
    def execution_steps(cls) -> tuple[str, ...]:
        """Return ordered public lifecycle steps for detector configs."""
        return (
            *cls.loading_steps(),
            *cls.context_steps(),
            "build_detector",
            *cls.scoring_steps(),
            *cls.persistence_steps(),
        )

    def _validate_detector_contract(self) -> None:
        """Validate that the detector contract can execute in the declared order."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> LifecycleResults:
        """Execute the declared detector-config lifecycle.

        Args:
            *args: Positional runtime values passed through to execution steps.
            **kwargs: Keyword runtime values passed through to execution steps.

        Returns:
            Lifecycle step outputs keyed by step name.
        """

    @abstractmethod
    def build_detector(
        self,
        model: "ModelConfig",
        attack: "AttackConfig",
    ) -> RuntimeValue:
        """Construct detector runtime for model/attack outputs.

        Args:
            model: Runtime model object used for detection.
            attack: Runtime attack object or outputs under inspection.

        Returns:
            Detector runtime object.
        """


@dataclass(eq=False, kw_only=True)
class FrameworkExperimentConfig(
    LoadableConfigMixin,
    PersistableConfigMixin,
    ScoreableConfigMixin,
    ContextAwareConfigMixin,
    ABC,
):
    """Framework-level experiment orchestration contract."""

    @classmethod
    def post_init_steps(cls) -> tuple[str, ...]:
        """Return private experiment-contract hooks executed during ``__post_init__``."""
        return (
            *cls.loading_post_init_steps(),
            *cls.persistence_post_init_steps(),
            *cls.scoring_post_init_steps(),
            *cls.context_post_init_steps(),
            "_validate_experiment_contract",
        )

    @classmethod
    def execution_steps(cls) -> tuple[str, ...]:
        """Return ordered public lifecycle steps for experiment configs."""
        return (
            *cls.loading_steps(),
            *cls.context_steps(),
            "run_experiment",
            *cls.scoring_steps(),
            *cls.persistence_steps(),
        )

    def _validate_experiment_contract(self) -> None:
        """Validate that the experiment contract can execute in the declared order."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> LifecycleResults:
        """Execute the declared experiment-config lifecycle.

        Args:
            *args: Positional runtime values passed through to execution steps.
            **kwargs: Keyword runtime values passed through to execution steps.

        Returns:
            Lifecycle step outputs keyed by step name.
        """

    @abstractmethod
    def run_experiment(self) -> dict[str, RuntimeValue]:
        """Execute experiment and return score or metric output."""


@dataclass(eq=False, kw_only=True)
class FrameworkScorerConfig(
    LoadableConfigMixin,
    PersistableConfigMixin,
    ScoreableConfigMixin,
    ContextAwareConfigMixin,
    ABC,
):
    """Framework-level scorer contract."""

    @classmethod
    def post_init_steps(cls) -> tuple[str, ...]:
        """Return private scorer-contract hooks executed during ``__post_init__``."""
        return (
            *cls.loading_post_init_steps(),
            *cls.persistence_post_init_steps(),
            *cls.scoring_post_init_steps(),
            *cls.context_post_init_steps(),
            "_validate_scorer_contract",
        )

    @classmethod
    def execution_steps(cls) -> tuple[str, ...]:
        """Return ordered public lifecycle steps for scorer configs."""
        return (
            *cls.loading_steps(),
            *cls.context_steps(),
            *cls.scoring_steps(),
            *cls.persistence_steps(),
        )

    def _validate_scorer_contract(self) -> None:
        """Validate that the scorer contract can execute in the declared order."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> LifecycleResults:
        """Execute the declared scorer-config lifecycle.

        Args:
            *args: Positional runtime values passed through to execution steps.
            **kwargs: Keyword runtime values passed through to execution steps.

        Returns:
            Lifecycle step outputs keyed by step name.
        """

    @abstractmethod
    def score(
        self,
        ind: ArrayLike,
        dep: ArrayLike,
        *args: Any,
        data: "DataConfig" | None = None,
        model: "ModelConfig" | None = None,
        attack: "AttackConfig" | None = None,
        **kwargs: Any,
    ) -> dict[str, RuntimeValue]:
        """Compute and return metric dictionary from runtime context.

        Args:
            ind: Array-like independent/reference scoring payload (previously ``y_true``).
            dep: Array-like dependent/prediction scoring payload (previously ``y_pred``).
            *args: Additional positional runtime scoring payloads.
            data: Optional runtime data context for metric computation.
            model: Optional runtime model context for metric computation.
            attack: Optional runtime attack context for metric computation.
            **kwargs: Additional runtime keyword scoring payloads.

        Returns:
            Metric outputs keyed by metric name.
        """


@dataclass(eq=False, kw_only=True)
class FrameworkDataScorer(FrameworkScorerConfig, ABC):
    """Framework-level scorer contract specialized for data-style scoring.

    This scorer variant expects a matrix-like independent payload and an
    array-like dependent payload.
    """

    @abstractmethod
    def score(
        self,
        ind: MatrixLike,
        dep: ArrayLike,
        *args: Any,
        data: "DataConfig" | None = None,
        model: "ModelConfig" | None = None,
        attack: "AttackConfig" | None = None,
        **kwargs: Any,
    ) -> dict[str, RuntimeValue]:
        """Compute and return metric dictionary from matrix/array inputs.

        Args:
            ind: Matrix-like independent/reference scoring payload.
            dep: Array-like dependent/prediction scoring payload.
            *args: Additional positional runtime scoring payloads.
            data: Optional runtime data context for metric computation.
            model: Optional runtime model context for metric computation.
            attack: Optional runtime attack context for metric computation.
            **kwargs: Additional runtime keyword scoring payloads.

        Returns:
            Metric outputs keyed by metric name.
        """


@dataclass(eq=False, kw_only=True)
class FrameworkModelDefenseConfig(
    PersistableConfigMixin,
    ScoreableConfigMixin,
    ContextAwareConfigMixin,
    ABC,
):
    """Framework-level contract for model defense runtime adapters."""

    @classmethod
    def post_init_steps(cls) -> tuple[str, ...]:
        """Return private defense-contract hooks executed during ``__post_init__``."""
        return (
            *cls.persistence_post_init_steps(),
            *cls.scoring_post_init_steps(),
            *cls.context_post_init_steps(),
            "_validate_model_defense_contract",
        )

    @classmethod
    def execution_steps(cls) -> tuple[str, ...]:
        """Return ordered public lifecycle steps for model-defense configs."""
        return (
            *cls.context_steps(),
            "apply_to",
            *cls.scoring_steps(),
            *cls.persistence_steps(),
        )

    def _validate_model_defense_contract(self) -> None:
        """Validate that defense operations are declared in execution order."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> LifecycleResults:
        """Execute the declared model-defense lifecycle.

        Args:
            *args: Positional runtime values passed through to execution steps.
            **kwargs: Keyword runtime values passed through to execution steps.

        Returns:
            Lifecycle step outputs keyed by step name.
        """

    @abstractmethod
    def apply_to(self, estimator: EstimatorLike, data: "DataConfig") -> RuntimeValue:
        """Apply one defense runtime to a fitted estimator.

        Args:
            estimator: Framework estimator runtime object targeted by the defense.
            data: Runtime data context used by the defense.

        Returns:
            Updated estimator runtime object.
        """


@dataclass(eq=False, kw_only=True)
class FrameworkDataPipelineConfig(FrameworkDataConfig, ABC):
    """Framework-level contract for data pipeline orchestration runtimes."""

    @classmethod
    def execution_steps(cls) -> tuple[str, ...]:
        """Return ordered public lifecycle steps for data-pipeline configs."""
        return (
            *cls.loading_steps(),
            *cls.context_steps(),
            "load_data",
            "sample_data",
            "fit_presample",
            "fit_X",
            "fit_y",
            "fit_Xy",
            "build_pipeline",
            "run_pipeline",
            *cls.scoring_steps(),
            *cls.persistence_steps(),
        )

    def _validate_data_contract(self) -> None:
        """Validate that the data-pipeline contract declares required stages."""

    @abstractmethod
    def build_pipeline(self) -> RuntimeValue:
        """Build and return pipeline runtime components."""

    @abstractmethod
    def run_pipeline(self, pipeline: RuntimeValue | None = None) -> RuntimeValue:
        """Attach or execute a pipeline runtime object against current data state.

        Args:
            pipeline: Optional pre-built pipeline runtime to execute or attach.

        Returns:
            Pipeline runtime output.
        """


@dataclass(eq=False, kw_only=True)
class FrameworkDataSamplerContract(ABC):
    """Framework-neutral contract for sampler runtimes used by data configs."""

    @abstractmethod
    def __call__(
        self,
        config: "BaseSampler",
    ) -> tuple[RuntimeValue, RuntimeValue, RuntimeValue]:
        """Return train/test/validation index arrays for a data runtime.

        Args:
            config: Base sampler configuration object used to produce splits.

        Returns:
            Train, test, and validation index payloads.
        """


__all__ = [
    "RuntimeValue",
    "MatrixLike",
    "ArrayLike",
    "EstimatorLike",
    "register_matrix_like_type",
    "register_array_like_type",
    "register_estimator_like_type",
    "get_matrix_like_types",
    "get_array_like_types",
    "get_estimator_like_types",
    "is_matrix_like",
    "is_array_like",
    "is_estimator_like",
    "LifecycleResults",
    "LifecycleStepNames",
    "DeclarativeConfigContract",
    "LoadableConfigMixin",
    "PersistableConfigMixin",
    "ScoreableConfigMixin",
    "ContextAwareConfigMixin",
    "FrameworkDataConfig",
    "FrameworkModelConfig",
    "FrameworkAttackConfig",
    "FrameworkDetectorConfig",
    "FrameworkExperimentConfig",
    "FrameworkScorerConfig",
    "FrameworkDataScorer",
    "FrameworkModelDefenseConfig",
    "FrameworkDataPipelineConfig",
    "FrameworkDataSamplerContract",
]
