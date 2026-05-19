"""Framework-side compatibility adapters for core config contracts.

These mixins bridge existing core runtime objects to the abstract framework
contracts without changing execution semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Union, cast

from .core import ArrayLike, EstimatorLike, MatrixLike, RuntimeValue

if TYPE_CHECKING:
    from ..attack.base import AttackConfig
    from ..data.base import DataConfig
    from ..data.sample import BaseSampler
    from ..model.base import ModelConfig


ScoreMap = dict[str, RuntimeValue]
ContextMap = dict[str, RuntimeValue]


class BaseContractMixin:
    """Shared lifecycle helpers used by framework contract adapters."""

    def load_defaults(self) -> RuntimeValue | None:
        """Load framework defaults when available.

        Args:
            None.

        Returns:
            Optional framework-specific defaults payload.
        """
        return None

    def load_cached(self) -> RuntimeValue | None:
        """Load pretrained state when available.

        Args:
            None.

        Returns:
            Optional framework-specific pretrained payload.
        """
        return None

    def resolve_context(self, **context: RuntimeValue) -> ContextMap:
        """Normalize runtime context kwargs into a dictionary.

        Args:
            **context: Runtime context fields provided by callers.

        Returns:
            Context mapping forwarded to downstream execution.
        """
        return dict(context)

    def load(self, filepath: str | None = None) -> BaseContractMixin | RuntimeValue:
        """Load adapter state from disk when a path is provided.

        Args:
            filepath: Optional path to load state from.

        Returns:
            The adapter instance when no path is provided, otherwise the
            downstream load payload.
        """
        if filepath is None:
            return self
        super_load = getattr(super(), "load", None)
        if callable(super_load):
            return super_load(filepath)
        raise AttributeError(
            f"{type(self).__name__} must define load(filepath) on a parent class.",
        )

    def save(self, filepath: str | None = None) -> RuntimeValue | None:
        """Persist adapter state when a target path is provided.

        Args:
            filepath: Optional path to write state to.

        Returns:
            `None` when no path is provided, otherwise the downstream save
            payload.
        """
        if filepath is None:
            return None
        super_save = getattr(super(), "save", None)
        if callable(super_save):
            return super_save(filepath)
        raise AttributeError(
            f"{type(self).__name__} must define save(filepath) on a parent class.",
        )


class DataContractMixin(BaseContractMixin):
    """Adapter methods for ``FrameworkDataConfig`` compliance."""

    dataset_name: str
    data_params: dict[str, RuntimeValue]
    test_size: Union[float, int, None]
    train_size: Union[float, int, None]
    val_size: Union[float, int, None]
    split: Union[int, None]
    sample: BaseSampler
    random_state: int
    stratify: Union[None, str, bool]
    classifier: Union[bool, str]
    target: Union[str, None]
    drop: list[str]
    keep: list[str]
    score_mode: Literal["train", "test", "val", "pre-sample"]
    score_dict: dict[str, Any]

    X: MatrixLike
    y: ArrayLike
    X_train: MatrixLike
    y_train: ArrayLike
    X_test: MatrixLike
    y_test: ArrayLike
    X_val: MatrixLike
    y_val: ArrayLike

    def fit_presample(
        self,
        X: MatrixLike,
        y: ArrayLike,
    ) -> tuple[MatrixLike, ArrayLike]:
        """Data-level default for pre-sample transform stage."""
        return X, y

    def fit_X(self, X: MatrixLike, y: ArrayLike) -> tuple[MatrixLike, ArrayLike]:
        """Data-level default for X-only transform stage."""
        return X, y

    def fit_y(self, X: MatrixLike, y: ArrayLike) -> tuple[MatrixLike, ArrayLike]:
        """Data-level default for y-only transform stage."""
        return X, y

    def fit_Xy(self, X: MatrixLike, y: ArrayLike) -> tuple[MatrixLike, ArrayLike]:
        """Data-level default for joint X/y transform stage."""
        return X, y

    def load_data(
        self,
        filepath: str | None = None,
    ) -> RuntimeValue | tuple[MatrixLike, ArrayLike]:
        """Load or materialize runtime data.

        Args:
            filepath: Optional path forwarded to the underlying loader.

        Returns:
            Loader output when filepath is provided, otherwise `(X, y)`.
        """
        if filepath is not None:
            super_load_data = getattr(super(), "load_data", None)
            if callable(super_load_data):
                return super_load_data(filepath)
            raise AttributeError(
                f"{type(self).__name__} must define load_data(filepath) on a parent class.",
            )
        load_raw_data = getattr(self, "load_raw_data", None)
        if not callable(load_raw_data):
            raise AttributeError(
                f"{type(self).__name__} must define load_raw_data() for data contract compliance.",
            )
        load_raw_data()
        return self.X, self.y

    def sample_data(
        self,
        X: MatrixLike | None = None,
        y: ArrayLike | None = None,
    ) -> tuple[MatrixLike, MatrixLike, ArrayLike, ArrayLike]:
        """Split data into train/test payloads.

        Args:
            X: Optional matrix-like feature payload override.
            y: Optional array-like target payload override.

        Returns:
            Tuple of `(X_train, X_test, y_train, y_test)`.
        """
        if X is not None:
            self.X = X
        if y is not None:
            self.y = y
        split_data = getattr(self, "split_data", None)
        if not callable(split_data):
            raise AttributeError(
                f"{type(self).__name__} must define split_data() for data contract compliance.",
            )
        split_data()
        return self.X_train, self.X_test, self.y_train, self.y_test

    def score(self, *args: RuntimeValue, **kwargs: RuntimeValue) -> ScoreMap:
        """Compute or return cached data-scoring outputs.

        Args:
            *args: Positional payloads for score computation.
            **kwargs: Keyword payloads for score computation.

        Returns:
            Score mapping from metric names to runtime values.
        """
        if args or kwargs:
            compute_score = getattr(self, "compute_score", None)
            if callable(compute_score):
                return cast(ScoreMap, compute_score(*args, **kwargs))
            raise AttributeError(
                f"{type(self).__name__} must define compute_score(*args, **kwargs) for scoring.",
            )
        return dict(getattr(self, "score_dict", {}) or {})


class DataSamplerContractMixin(BaseContractMixin):
    """Adapter methods for ``FrameworkDataSamplerContract`` compliance."""

    test_size: float
    train_size: float
    val_size: float
    random_state: int
    sample: Literal["split", "fold", "shuffle"]
    stratify: bool

    def __call__(
        self,
        config: RuntimeValue,
    ) -> tuple:
        """Dispatch sampler execution through `sample_data` when available."""
        sample_data = getattr(self, "sample_data", None)
        if callable(sample_data):
            return cast(
                tuple[RuntimeValue, RuntimeValue, RuntimeValue],
                sample_data(config),
            )
        raise AttributeError(
            f"{type(self).__name__} must define sample_data(config) for sampler contract compliance.",
        )


class DataPipelineContractMixin(BaseContractMixin):
    """Adapter methods for ``FrameworkDataPipelineConfig`` compliance."""

    pipeline: dict[str, RuntimeValue]
    pre_sample_transform: bool

    def normalize_step_hooks(self, raw_hooks: RuntimeValue) -> list[str]:
        """Normalize `plugin_hook` declarations from pipeline step config."""
        if raw_hooks is None:
            return []
        if isinstance(raw_hooks, str):
            hooks = [raw_hooks]
        elif isinstance(raw_hooks, (list, tuple, set)):
            hooks = list(raw_hooks)
        else:
            raise TypeError(
                f"plugin_hook must be None, str, or list-like. Got {type(raw_hooks)}",
            )

        normalized = []
        for hook in hooks:
            text = str(hook).strip().lower()
            if text:
                normalized.append(text)
        return normalized

    def pipeline_declares_hook(self, hook_name: str) -> bool:
        """Return True when any pipeline step declares the requested hook."""
        target_hook = str(hook_name).strip().lower()
        for _, step_config in getattr(self, "pipeline", {}).items():
            hooks = self.normalize_step_hooks(step_config.get("plugin_hook", None))
            if target_hook in hooks:
                return True
        return False

    def declares_hook(self, hook_name: str) -> bool:
        """Expose pipeline hook declaration checks for plugin dispatch."""
        return self.pipeline_declares_hook(hook_name)

    def build_pipeline(self) -> RuntimeValue:
        """Build a pipeline object through a framework hook.

        Args:
            None.

        Returns:
            Constructed pipeline payload.

        Raises:
            AttributeError: If no callable pipeline factory is available.
        """
        create_pipeline = getattr(self, "create_pipeline", None)
        if callable(create_pipeline):
            return create_pipeline()
        raise AttributeError(
            f"{type(self).__name__} must define create_pipeline() for data pipeline contract compliance.",
        )

    def fit_presample(
        self,
        X: MatrixLike,
        y: ArrayLike,
    ) -> tuple[MatrixLike, ArrayLike]:
        """Pre-fit stage passthrough for pre-sample transforms.

        Args:
            X: Matrix-like feature payload.
            y: Array-like target payload.

        Returns:
            Unmodified `(X, y)` tuple.
        """
        return X, y

    def fit_X(self, X: MatrixLike, y: ArrayLike) -> tuple[MatrixLike, ArrayLike]:
        """Fit X-only pipeline stage.

        Args:
            X: Matrix-like feature payload.
            y: Array-like target payload.

        Returns:
            Unmodified `(X, y)` tuple.
        """
        return X, y

    def fit_y(self, X: MatrixLike, y: ArrayLike) -> tuple[MatrixLike, ArrayLike]:
        """Fit y-only pipeline stage.

        Args:
            X: Matrix-like feature payload.
            y: Array-like target payload.

        Returns:
            Unmodified `(X, y)` tuple.
        """
        return X, y

    def fit_Xy(self, X: MatrixLike, y: ArrayLike) -> tuple[MatrixLike, ArrayLike]:
        """Fit joint feature/target pipeline stage.

        Args:
            X: Matrix-like feature payload.
            y: Array-like target payload.

        Returns:
            Unmodified `(X, y)` tuple.
        """
        return X, y

    def run_pipeline(self, pipeline: RuntimeValue | None = None) -> RuntimeValue:
        """Execute or passthrough a prepared pipeline payload.

        Args:
            pipeline: Pipeline payload to execute.

        Returns:
            Executed pipeline result when a hook exists, else the original
            pipeline payload.
        """
        apply_pipeline = getattr(self, "apply_pipeline", None)
        if callable(apply_pipeline):
            return apply_pipeline(pipeline)
        return pipeline


class ModelContractMixin(BaseContractMixin):
    """Adapter methods for ``FrameworkModelConfig`` compliance."""

    model_type: Union[str, None]
    classifier: Union[bool, str]
    model_params: dict[str, RuntimeValue]
    probability: bool
    score_mode: Literal["train", "test", "val"]
    score_dict: dict[str, RuntimeValue]
    training_predictions: RuntimeValue
    predictions: RuntimeValue
    val_predictions: RuntimeValue
    training_probabilities: RuntimeValue
    probabilities: RuntimeValue
    val_probabilities: RuntimeValue

    def init_model(self, data: DataConfig | None = None) -> RuntimeValue:
        """Initialize and return a model payload.

        Args:
            data: Optional data payload retained for signature compatibility.

        Returns:
        """
        initialize_model = getattr(self, "initialize_model", None)
        if not callable(initialize_model):
            raise AttributeError(
                f"{type(self).__name__} must define initialize_model(data) for model contract compliance.",
            )
        initialize_model(data)

        get_model = getattr(self, "get_model", None)
        if callable(get_model):
            return get_model()
        raise AttributeError(
            f"{type(self).__name__} must define get_model() for model contract compliance.",
        )

    def fit_model(self, data: DataConfig) -> RuntimeValue:
        """Train model payload using data with split attributes.

        Args:
            data: Runtime payload expected to expose `X_train` and `y_train`
                when training is required.

        Returns:
            Model payload from `get_model()`.
        """
        if data is not None and hasattr(data, "X_train") and hasattr(data, "y_train"):
            train = getattr(self, "train", None)
            if callable(train):
                train(data.X_train, data.y_train)
            else:
                raise AttributeError(
                    f"{type(self).__name__} must define train(X, y) for model contract compliance.",
                )

        get_model = getattr(self, "get_model", None)
        if callable(get_model):
            return get_model()
        raise AttributeError(
            f"{type(self).__name__} must define get_model() for model contract compliance.",
        )

    def score(self, *args: RuntimeValue, **kwargs: RuntimeValue) -> ScoreMap:
        """Compute or return cached model-scoring outputs.

        Args:
            *args: Positional payloads for score computation.
            **kwargs: Keyword payloads for score computation.

        Returns:
            Score mapping from metric names to runtime values.
        """
        if args or kwargs:
            compute_score = getattr(self, "compute_score", None)
            if callable(compute_score):
                return cast(ScoreMap, compute_score(*args, **kwargs))
            raise AttributeError(
                f"{type(self).__name__} must define compute_score(*args, **kwargs) for scoring.",
            )
        return dict(getattr(self, "score_dict", {}) or {})


class ModelDefenseContractMixin(BaseContractMixin):
    """Adapter methods for ``FrameworkModelDefenseConfig`` compliance."""

    model_type: Union[str, None]
    classifier: Union[bool, str, None]
    model_params: dict[str, RuntimeValue]
    probability: bool
    defense_name: Union[str, None]
    defense_params: dict[str, RuntimeValue]
    init_params: dict[str, RuntimeValue]
    score_dict: dict[str, RuntimeValue]

    def apply_to(self, estimator: EstimatorLike, data: DataConfig) -> RuntimeValue:
        """Apply defense payloads against an estimator.

        Args:
            estimator: Estimator payload to register with the defense.
            data: Runtime data payload used by the defense hook.

        Returns:
            Defense application result from `apply_defense`.

        Raises:
            AttributeError: If no defense hook is available.
        """
        set_estimator = getattr(self, "set_estimator", None)
        if not callable(set_estimator):
            raise AttributeError(
                f"{type(self).__name__} must define set_estimator(estimator) for defense contract compliance.",
            )
        set_estimator(estimator)
        apply_defense = getattr(self, "apply_defense", None)
        if callable(apply_defense):
            return apply_defense(data)
        raise AttributeError(
            f"{type(self).__name__} must define apply_to(estimator, data) or apply_defense(data).",
        )

    def score(self, *args: RuntimeValue, **kwargs: RuntimeValue) -> ScoreMap:
        """Return cached defense score outputs.

        Args:
            *args: Unused positional payloads retained for compatibility.
            **kwargs: Unused keyword payloads retained for compatibility.

        Returns:
            Score mapping from metric names to runtime values.
        """
        _ = (args, kwargs)
        return dict(getattr(self, "score_dict", {}) or {})


class AttackContractMixin(BaseContractMixin):
    """Adapter methods for ``FrameworkAttackConfig`` compliance."""

    attack_type: str
    attack_params: dict[str, RuntimeValue]
    init_params: dict[str, RuntimeValue]
    attack_size: int
    targeted_attribute: str
    mode: Literal["auto", "train", "test", "val"]
    score_dict: dict[str, RuntimeValue]

    def build_attack(self, model: ModelConfig, data: DataConfig) -> RuntimeValue:
        """Initialize and return attack payloads.

        Args:
            model: Model payload the attack targets.
            data: Data payload required for attack initialization.

        Returns:
            Attack payload from `initialize_attack`.
        """
        initialize_attack = getattr(self, "initialize_attack", None)
        if not callable(initialize_attack):
            raise AttributeError(
                f"{type(self).__name__} must define initialize_attack(model, data) for attack contract compliance.",
            )
        attack_result = cast(
            tuple[RuntimeValue, RuntimeValue, RuntimeValue, RuntimeValue],
            initialize_attack(model, data),
        )
        attack, _, _, _ = attack_result
        return attack

    def score(self, *args: RuntimeValue, **kwargs: RuntimeValue) -> ScoreMap:
        """Return cached attack score outputs.

        Args:
            *args: Unused positional payloads retained for compatibility.
            **kwargs: Unused keyword payloads retained for compatibility.

        Returns:
            Score mapping from metric names to runtime values.
        """
        _ = (args, kwargs)
        return dict(getattr(self, "score_dict", {}) or {})


class DetectorContractMixin(BaseContractMixin):
    """Adapter methods for ``FrameworkDetectorConfig`` compliance."""

    detector_type: str
    detector_params: dict[str, RuntimeValue]
    fit_params: dict[str, RuntimeValue]
    detector_model: RuntimeValue
    detector: RuntimeValue
    score_dict: dict[str, RuntimeValue]

    def build_detector(
        self,
        model: ModelConfig,
        attack: AttackConfig,
    ) -> RuntimeValue | None:
        """Build or fetch detector payload.

        Args:
            model: Model payload retained for compatibility.
            attack: Attack payload retained for compatibility.

        Returns:
            Detector payload when available, otherwise `None`.
        """
        _ = (model, attack)
        return getattr(self, "detector", None)

    def score(self, *args: RuntimeValue, **kwargs: RuntimeValue) -> ScoreMap:
        """Return cached detector score outputs.

        Args:
            *args: Unused positional payloads retained for compatibility.
            **kwargs: Unused keyword payloads retained for compatibility.

        Returns:
            Score mapping from metric names to runtime values.
        """
        _ = (args, kwargs)
        return dict(getattr(self, "score_dict", {}) or {})


class ExperimentContractMixin(BaseContractMixin):
    """Adapter methods for ``FrameworkExperimentConfig`` compliance."""

    experiment_name: str
    data: RuntimeValue
    model: RuntimeValue
    defense: RuntimeValue
    attack: RuntimeValue
    detector: RuntimeValue
    files: RuntimeValue
    random_state: int
    classifier: Union[str, bool]
    evaluation_mode: Literal["standard", "tuning", "report"]

    def run_experiment(self) -> ScoreMap:
        """Run experiment orchestration through the config call entrypoint.

        Args:
            None.

        Returns:
            Experiment result mapping.
        """
        run = getattr(self, "__call__", None)
        if callable(run):
            return cast(ScoreMap, run())
        raise AttributeError(
            f"{type(self).__name__} must define __call__() for experiment contract compliance.",
        )

    def score(self, *args: RuntimeValue, **kwargs: RuntimeValue) -> ScoreMap:
        """Return cached experiment score outputs.

        Args:
            *args: Unused positional payloads retained for compatibility.
            **kwargs: Unused keyword payloads retained for compatibility.

        Returns:
            Score mapping from metric names to runtime values.
        """
        _ = (args, kwargs)
        return dict(getattr(self, "score_dict", {}) or {})


class ScorerContractMixin(BaseContractMixin):
    """Adapter methods for ``FrameworkScorerConfig`` compliance."""

    scorers: dict[str, RuntimeValue]

    def score(
        self,
        ind: ArrayLike,
        dep: ArrayLike,
        *args: RuntimeValue,
        data: DataConfig | None = None,
        model: ModelConfig | None = None,
        attack: AttackConfig | None = None,
        **kwargs: RuntimeValue,
    ) -> ScoreMap:
        """Execute scorer orchestration through the config call entrypoint.

        Args:
            ind: Array-like independent/reference scoring payload.
            dep: Array-like dependent/prediction scoring payload.
            *args: Additional positional payloads for scoring.
            data: Optional runtime data context for scoring.
            model: Optional runtime model context for scoring.
            attack: Optional runtime attack context for scoring.
            **kwargs: Additional keyword payloads for scoring.

        Returns:
            Score mapping produced by the scorer config.
        """
        run = getattr(self, "__call__", None)
        if not callable(run):
            raise AttributeError(
                f"{type(self).__name__} must define __call__(...) for scorer contract compliance.",
            )

        return cast(
            ScoreMap,
            run(
                ind,
                dep,
                *args,
                data=data,
                model=model,
                attack=attack,
                **kwargs,
            ),
        )


__all__ = [
    "BaseContractMixin",
    "DataContractMixin",
    "DataSamplerContractMixin",
    "DataPipelineContractMixin",
    "ModelContractMixin",
    "ModelDefenseContractMixin",
    "AttackContractMixin",
    "DetectorContractMixin",
    "ExperimentContractMixin",
    "ScorerContractMixin",
]
