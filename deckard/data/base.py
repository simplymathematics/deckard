# Imports
import os
import pandas as pd
import time
import logging
from pathlib import Path

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Union, cast
from omegaconf import DictConfig, ListConfig

import numpy as np

# deckard
from ..utils import (
    BaseConfig,
    RuntimeSerializable,
    load_class,
    coerce_to_list,
    merge_list_of_dicts,
)
from ..types import (
    ArrayLike,
    DatasetLike,
    EstimatorLike,
    IndexLike,
    MatrixLike,
    StringifiedClass,
    TabularLike,
)
from ..orchestration import (
    ScoreOrchestratorMixin,
    resolve_data_split_payload,
    stage_hook_token,
)
from ..artifacts import ArtifactLoaderMixin, ScoreDict, SerializableValue
from .canon import (
    CANONICAL_DATA_LOAD_FILETYPES,
    CANONICAL_DATA_SAVE_FILETYPES,
    CANONICAL_DATASET_LOAD_FILETYPES,
    CANONICAL_DATA_STAGES,
    DEFAULT_DATA_SCORE_STAGE,
    CANONICAL_DATA_TIMES,
    DataFiles,
    ensure_data_runtime_contract,
    ensure_canonical_times,
    merge_data_files,
    normalize_data_score_mode,
    normalize_data_score_stage,
    resolve_runtime_files,
)

# Setup logger
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .pipeline.base import DataPipeline
    from .sample import BaseSampler


AUTO_SCORER = "auto"
DECKARD_TEST_MAX_SAMPLES_ENV = "DECKARD_TEST_MAX_SAMPLES"


def _coerce_scorer_config(*args, **kwargs):
    """Lazy import scorer coercion to avoid data<->score import cycles at module import time."""
    from ..score.base import coerce_scorer_config as _coerce

    return _coerce(*args, **kwargs)


def _is_data_scorer_instance(scorer: Any) -> bool:
    """Return True when *scorer* is explicitly a data-profile scorer."""
    from ..score.base import _DataScorerMarker

    if isinstance(scorer, _DataScorerMarker):
        return True
    if callable(scorer):
        return True
    return str(getattr(scorer, "scoring_type", "")).strip().lower() == "data"


def _load_optuna_studies_dataframe(
    *,
    storage: Any,
    study_name: str | None,
    schema: Any,
    **kwargs: Any,
):
    from ..optuna_callback import load_optuna_studies_dataframe

    return load_optuna_studies_dataframe(
        storage=storage,
        study_name=study_name,
        schema=schema,
        **kwargs,
    )


@dataclass(eq=False, kw_only=True)
class DataConfig(ScoreOrchestratorMixin, BaseConfig):
    """DataConfig runtime class.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    f"""
    Configuration and utility class for loading, preprocessing, and splitting datasets for machine learning tasks.

    Attributes
    ----------
    name : DatasetLike
        Canonical dataset identifier to load or path to a data file.
    data_params : dict
        Additional parameters for data loading or generation.
    split : Union[int, None]
        Which split index to use as the validation set when ``sampler`` performs
        cross-validation or shuffle splitting
        (e.g. :class:`~deckard.data.sample.KFoldSampler` or
        :class:`~deckard.data.sample.ShuffleSampler`).  Defaults to ``0``.
    sampler : Union[BaseSampler, Literal["split", "shuffle", "fold"], dict, None]
        Optional pluggable sampler spec. Resolution/configuration/execution is
        centralized in :class:`~deckard.data.sample.BaseSampler`. Accepted forms
        include alias strings (``split``/``shuffle``/``fold``), instantiated
        sampler objects, sampler classes, and Hydra-style dicts with
        ``name``/``_target_``.
    classifier: bool
        Whether the task is classification (True) or regression (False).
    drop: list
        List of columns to drop from the dataset.
    target: Union[str, None]
        Name of the target column in the dataset (if applicable).
    keep: list
        List of columns to keep in the dataset.
    plugins : list
        Optional data plugin specifications executed during load/sample/score hooks.
    _X : pd.DataFrame
        Loaded feature matrix.
    _y : pd.Series
        Loaded target vector.
    data_load_time : float
        Time taken to load the data.
    data_sample_time : float
        Time taken to sample/split the data.
    train_n : int
        Number of training samples.
    test_n : int
        Number of testing samples.
    val_n : int
        Number of validation samples (set only when a sampler is used).
    alias: str
        Optional alias for the dataset configuration.
    train_indices : list
        Indices for training samples.
    test_indices : list
        Indices for testing samples.
    val_indices : list
        Indices for validation samples (set only when a sampler is used).
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    X_test : pd.DataFrame
        Testing feature matrix.
    y_test : pd.Series
        Testing target vector.
    X_val : pd.DataFrame
        Validation feature matrix (set only when a sampler is used).
    y_val : pd.Series
        Validation target vector (set only when a sampler is used).
    score_dict : dict
        Dictionary to store scores or metrics.
    _target_ : str
        Internal identifier for the class.

    Parameter layers
    ----------------
    data_params : dict
        Dataset loader/generator kwargs consumed by the selected data source
        (for example sklearn loaders, OpenML fetchers, synthetic generators,
        or file readers).
    plugins : list
        Plugin specs resolved at runtime and invoked through hook names.
        For :class:`DataHookPlugin`, initialization fields are:
        ``hook_name``, ``method_name``, ``method_kwargs``, and ``init_params``
        (metadata only).
    init_params (plugin metadata)
        Plugin declaration metadata (class/type/library docs). This is not
        interpreted by DataConfig orchestration logic directly.

    Family-specific parameter semantics
    ------------------------------------
    sklearn loader datasets
        ``data_params`` are forwarded to sklearn dataset loader callables
        (for example ``as_frame=True``).
    OpenML/fetch datasets
        ``data_params`` typically include source-identifying keys (for example
        ``name``, ``version``, ``as_frame``) and are passed to fetch helpers.
    synthetic generators
        ``data_params`` are generation controls (for example
        ``n_samples``, ``n_features``, ``n_classes``).
    file-backed datasets
        ``data_params`` may include read-time options used by pandas/file
        loading utilities.

        Canonical dataset-load filetypes:
        {", ".join(CANONICAL_DATASET_LOAD_FILETYPES)}

        Canonical artifact save filetypes:
        {", ".join(CANONICAL_DATA_SAVE_FILETYPES)}

        Canonical artifact load filetypes:
        {", ".join(CANONICAL_DATA_LOAD_FILETYPES)}

    Plugin hook runtime params
    --------------------------
    Hooks are orchestrated by ``_run_plugin_hook(hook_name, **kwargs)``.
    Core hook names used by DataConfig runtime are:
    ``before_load_data``, ``after_load_data``, ``before_sample``, ``before_pipeline``, ``after_pipeline``
    ``after_sample``, ``before_score``, and ``after_score``.


    Methods
    -------
    __post_init__()
        Post-initialization method to validate parameters and initialize internal attributes.
    __hash__()
        Computes a hash value for the instance based on non-private attributes.
    _get_stratify_col()
        Returns the stratification array (or None) based on ``self.stratify``.
    load_dataset()
        Loads the dataset payload into runtime features/targets from defaults or file-backed datasets.
    sample()
        Splits the loaded dataset into training, testing, and optionally validation sets.
    score(*args, mode=None, **kwargs)
        Scores the selected data split via the configured data scorer.
    __call__(filepath=None)
        Loads and samples the dataset, splits it into training and testing sets, and returns the corresponding features and labels.
    save(filepath)
        Saves the current state of the DataConfig instance to a file.
    load(filepath=None)
        Loads a cached DataConfig object from pickle-based artifact storage.

    Raises:
        ValueError: For invalid parameter values or missing data.
        NotImplementedError: For unsupported datasets or file types.

    Examples
    --------
    >>> config = DataConfig(name="adult", **kwargs)
    >>> config()
    >>> X_train = config.X_train
    >>> y_train = config.y_train
    >>> X_test = config.X_test
    >>> y_test = config.y_test
    >>> score_dict = config.score_dict

    Using a pluggable sampler for 3-way splits::

        from deckard.data.sample import SplitSampler
        config = DataConfig(
            name="digits",
            sampler=SplitSampler(test_size=0.2, val_size=0.1),
        )
        config()
        X_val, y_val = config.X_val, config.y_val
    """

    # Configuration fields
    name: DatasetLike = "adult"
    data_params: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Keyword arguments forwarded to the dataset loader."},
    )
    split: Union[int, None] = None
    sampler: Union["BaseSampler", Literal["split", "shuffle", "fold"], dict, None] = (
        "split"
    )
    classifier: Union[bool, str] = True
    target: Union[str, None] = None
    drop: list[str] = field(
        default_factory=list,
        metadata={"help": "Feature columns dropped before scoring or sampling."},
    )
    keep: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Optional allowlist of feature columns retained from the dataset.",
        },
    )
    plugins: list[Any] = field(
        default_factory=list,
        metadata={"help": "Resolved data plugins attached to this runtime config."},
    )
    alias: Union[str, None] = None
    scorer: Any = AUTO_SCORER
    score_mode: str = "test"
    score_stage: str = DEFAULT_DATA_SCORE_STAGE
    pipeline: Any = None
    files: DataFiles = field(
        default_factory=lambda: {},
        metadata={
            "help": "Declared input and output file paths for this dataset runtime.",
        },
    )

    # Runtime state fields
    score_dict: ScoreDict = field(
        default_factory=ScoreDict,
        init=False,
        repr=False,
        metadata={"help": "Dataset-level score payload accumulated during runtime."},
    )
    times: dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
        metadata={
            "help": "Timing measurements collected for dataset load and sample stages.",
        },
    )
    data_load_time: Union[float, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Elapsed time in seconds for the dataset load stage."},
    )
    data_sample_time: Union[float, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Elapsed time in seconds for the dataset sampling stage."},
    )
    _X: Union[TabularLike, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Cached full feature matrix for the loaded dataset."},
    )
    _y: Union[pd.Series, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Cached full target vector for the loaded dataset."},
    )
    train_indices: Union[IndexLike, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Indices selected for the training split."},
    )
    test_indices: Union[IndexLike, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Indices selected for the test split."},
    )
    val_indices: Union[IndexLike, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Indices selected for the validation split."},
    )
    _X_train: Union[TabularLike, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Cached training feature matrix."},
    )
    _y_train: Union[pd.Series, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Cached training target vector."},
    )
    _X_test: Union[TabularLike, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Cached test feature matrix."},
    )
    _y_test: Union[pd.Series, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Cached test target vector."},
    )
    _X_val: Union[TabularLike, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Cached validation feature matrix."},
    )
    _y_val: Union[pd.Series, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Cached validation target vector."},
    )
    train_n: Union[int, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Number of samples in the training split."},
    )
    test_n: Union[int, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Number of samples in the test split."},
    )
    val_n: Union[int, None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Number of samples in the validation split."},
    )
    _target_: Union[str, None] = field(
        default="deckard.data.base.DataConfig",
        init=True,
        repr=True,
        metadata={"help": "Hydra target path used to rehydrate this data config."},
    )
    _plugin_objects: Union[list[Any], None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={
            "help": "Instantiated plugin objects cached for runtime hook dispatch.",
        },
    )
    _sampler_obj: Union[Callable[..., Any], None] = field(
        default=None,
        init=False,
        repr=False,
        metadata={
            "help": "Resolved sampler object cached for repeated split generation.",
        },
    )
    pipeline: Any = field(
        default=None,
        init=True,
        repr=True,
        metadata={
            "help": "Resolved runtime pipeline object built from pipeline config.",
        },
    )
    _score_orchestration_active: bool = field(
        default=False,
        init=False,
        repr=False,
        metadata={
            "help": "Internal guard indicating score orchestration is currently active.",
        },
    )

    def _normalize_score_mode(self, mode: str) -> str:
        return normalize_data_score_mode(mode)

    def _normalize_score_stage(
        self,
        stage: str | None,
        *,
        allow_all_auto: bool = True,
    ) -> str:
        normalized = normalize_data_score_stage(
            stage or DEFAULT_DATA_SCORE_STAGE,
        )
        if normalized in {"all", "auto"} and not allow_all_auto:
            raise ValueError(
                "Unknown data score stage "
                f"'{stage}'. Must be one of {list(CANONICAL_DATA_STAGES)}",
            )
        return normalized

    def _stage_hook_token(self, stage: str) -> str:
        return stage_hook_token(stage)

    def _validate_init(self):
        """
        Post-initialization method for setting up data-related attributes.

        Validates that `test_size` is between 0 and 1, then initializes training size and internal attributes
        for data loading, sampling, parameters, and train/test splits.

        Raises:
            ValueError: If `test_size` is not between 0 and 1.
        """

        self.data_params = self.data_params if self.data_params is not None else {}

        def _is_declared_dataset(token: str) -> bool:
            from .declarations import build_loader_registry

            try:
                declared = build_loader_registry(self)
            except Exception:
                return False
            return token in declared

        def _is_supported_dataset_token(token: str) -> bool:
            normalized = token.strip()
            if normalized == "":
                return False
            if normalized.lower() in {"adult", "torch_mnist"}:
                return True
            if ":" in normalized:
                # Supports module:path declaration specs (e.g. torch_fairness_dataset.py:CelebASmileDataset).
                return True
            if _is_declared_dataset(normalized):
                return True
            if Path(normalized).suffix in CANONICAL_DATASET_LOAD_FILETYPES:
                return True
            return False

        canonical_name = str(self.name or "").strip()
        if canonical_name == "":
            raise ValueError(
                "DataConfig.name must be non-empty and provided as a DatasetLike "
                "(str or pathlib.Path).",
            )
        self.name = canonical_name
        if not _is_supported_dataset_token(canonical_name):
            logger.debug(
                "DataConfig.name=%r is not a declared dataset token or supported file extension; "
                "deferring validation to runtime loader resolution.",
                canonical_name,
            )
        self.score_stage = self._normalize_score_stage(
            getattr(self, "score_stage", DEFAULT_DATA_SCORE_STAGE)
            or DEFAULT_DATA_SCORE_STAGE,
        )
        self.score_mode = self._normalize_score_mode(
            getattr(self, "score_mode", "test") or "test",
        )

        self.files = merge_data_files(getattr(self, "files", None), None)
        self.drop = [] if not hasattr(self, "drop") or self.drop is None else self.drop
        self.keep = [] if not hasattr(self, "keep") or self.keep is None else self.keep
        ensure_data_runtime_contract(self)
        for attr in ["data_load_time", "data_sample_time", "split"]:
            if not hasattr(self, attr):
                setattr(self, attr, None)
        self.times = ensure_canonical_times(getattr(self, "times", None))
        for key in CANONICAL_DATA_TIMES:
            if hasattr(self, key):
                self.times[key] = getattr(self, key)
            setattr(self, key, self.times.get(key))

        if not hasattr(self, "_target_") or self._target_ in {None, ""}:
            self._target_ = "deckard.data.base.DataConfig"
        if not self.data_params:
            self.data_params = {}

    def _set_time(self, key: str, value: float | None) -> None:
        self.times[key] = value
        setattr(self, key, value)

    def _sync_canonical_time_state(self) -> None:
        """Backfill canonical timing fields from attrs, times map, and score payload.

        This is especially important for cached runs where older serialized
        artifacts may have partial timing state spread across ``self.times``,
        top-level attributes, or ``score_dict``.
        """
        self.times = ensure_canonical_times(getattr(self, "times", None))
        score_payload = ScoreDict.from_payload(getattr(self, "score_dict", {}) or {})
        for key in CANONICAL_DATA_TIMES:
            attr_value = getattr(self, key, None)
            if attr_value is not None:
                self.times[key] = attr_value
                continue

            time_value = self.times.get(key)
            if time_value is None:
                score_value = score_payload.get(key)
                if score_value is not None:
                    time_value = cast(float | None, score_value)

            self.times[key] = time_value
            setattr(self, key, time_value)

    def _resolve_max_samples(self, dataset_len: int) -> Union[int, None]:
        """Resolve an optional dataset cap from the test-only environment variable."""
        max_samples_text = os.environ.get(DECKARD_TEST_MAX_SAMPLES_ENV)
        if max_samples_text in [None, ""]:
            return None
        assert max_samples_text is not None
        try:
            max_samples = int(max_samples_text)
        except (TypeError, ValueError):
            raise ValueError(
                f"{DECKARD_TEST_MAX_SAMPLES_ENV} must be an integer, got {max_samples_text}",
            )
        if max_samples <= 0:
            return None
        return min(max_samples, dataset_len)

    def _apply_max_samples(self):
        """Truncate loaded tabular data to the configured test-only max sample cap."""
        if self._X is None or self._y is None:
            return

        sample_cap = self._resolve_max_samples(len(self._y))
        if sample_cap is None or sample_cap >= len(self._y):
            return

        if isinstance(self._X, pd.DataFrame):
            self._X = self._X.iloc[:sample_cap].copy()
        elif isinstance(self._X, pd.Series):
            self._X = self._X.iloc[:sample_cap].copy()
        else:
            raise TypeError(
                f"Unsupported _X type for {DECKARD_TEST_MAX_SAMPLES_ENV}: {type(self._X)}",
            )

        self._y = self._y.iloc[:sample_cap].copy()

    def _copy_runtime_state_to(self, target) -> None:
        runtime_fields = [
            "score_dict",
            "times",
            "files",
            "data_load_time",
            "data_sample_time",
            "data_pipeline_time",
            "data_score_time",
            "_X",
            "_y",
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
            "pipeline_y_fit_time",
            "pipeline_y_transform_n",
            "pipeline_y_transform_time",
        ]
        for attr in runtime_fields:
            if hasattr(self, attr):
                setattr(target, attr, getattr(self, attr, None))

    def __post_init__(self):
        self._validate_init()
        self._coerce_pipeline_runtime()
        self.scorer = _coerce_scorer_config(
            self.scorer,
            default_factory=lambda: load_class(
                (
                    "deckard.score.data.DefaultDataClassificationScorerDictConfig"
                    if self.classifier
                    else "deckard.score.data.DefaultDataRegressionScorerDictConfig"
                ),
            ),
        )
        if (
            self.scorer is not None
            and str(
                getattr(self.scorer, "scoring_type", ""),
            ).strip()
            == ""
        ):
            try:
                setattr(self.scorer, "scoring_type", "data")
            except Exception:
                pass
        if self.scorer is not None and not _is_data_scorer_instance(self.scorer):
            scoring_type = (
                str(getattr(self.scorer, "scoring_type", "")).strip().lower()
            )
            if scoring_type in {"model", "attack"}:
                raise TypeError(
                    "DataConfig requires a data-profile scorer configuration. "
                    "Model/attack scorers are not valid for data-split scoring.",
                )
        self._initialize_runtime_components()

    def _coerce_pipeline_runtime(self) -> None:
        """Normalize legacy pipeline config payloads to a runtime DataPipeline."""
        raw_pipeline = getattr(self, "pipeline", None)
        if raw_pipeline is None:
            return

        from .pipeline.base import DataPipeline

        if isinstance(raw_pipeline, DataPipeline):
            self.pipeline = raw_pipeline
            return
        if isinstance(raw_pipeline, (dict, DictConfig)):
            pipeline_dict = {str(k): v for k, v in dict(raw_pipeline).items()}
            self.pipeline = DataPipeline(pipeline=pipeline_dict)
            return
        if isinstance(raw_pipeline, (list, ListConfig)):
            merged = merge_list_of_dicts(coerce_to_list(raw_pipeline))
            self.pipeline = DataPipeline(pipeline=dict(merged))
            return
        raise TypeError(
            "DataConfig.pipeline must be DataPipeline | dict | list | None, "
            f"got {type(raw_pipeline)}",
        )

    def _init_pipeline(self):
        """Resolve and return the configured runtime X-stage pipeline."""
        self._coerce_pipeline_runtime()
        pipeline_runtime = getattr(self, "pipeline", None)
        if pipeline_runtime is None:
            return None, None

        from .pipeline.base import DataPipeline

        if isinstance(pipeline_runtime, DataPipeline):
            pipeline = pipeline_runtime._build_x_pipeline(
                pipeline_runtime._collect_x_steps(stage="X"),
            )
        else:
            pipeline = pipeline_runtime
        return pipeline, pipeline_runtime

    def fit_transform(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        pipeline,
    ):
        """Fit the provided runtime pipeline and transform train/test features."""
        from .pipeline.base import DataPipeline

        if pipeline is None:
            return X_train, X_test, y_train, y_test
        if isinstance(pipeline, DataPipeline):
            pipeline = pipeline._build_x_pipeline(
                pipeline._collect_x_steps(stage="X"),
            )
        assert pipeline is not None
        fit_start = time.process_time()
        if y_train is not None:
            pipeline.fit(X_train, y_train)
        else:
            pipeline.fit(X_train)
        self.pipeline_fit_time = time.process_time() - fit_start
        transform_start = time.process_time()
        X_train_t = pipeline.transform(X_train)
        X_test_t = pipeline.transform(X_test)
        self.pipeline_transform_time = time.process_time() - transform_start
        return X_train_t, X_test_t, y_train, y_test

    def _initialize_runtime_components(self) -> None:
        """Instantiate runtime-bound plugin and sampler objects eagerly."""
        from .sample import BaseSampler

        self._plugin_objects = self._get_plugins()
        self._configure_score_orchestration_plugins()
        self._sampler_obj = BaseSampler.resolve(self)

    @property
    def scores(self) -> ScoreDict:
        """Canonical score container alias (backed by ``score_dict``).

        Returns:
            Canonical score payload container.
        """
        return self.score_dict

    @scores.setter
    def scores(self, value: ScoreDict) -> None:
        """Set canonical score payload from mapping input.

        Args:
            value: Score payload mapping.
        """
        self.score_dict = ScoreDict.from_payload(value)

    @property
    def y(self) -> pd.Series | None:
        """Convenience alias for the loaded target vector.

        Returns:
            Loaded target payload.
        """
        return self._y

    @y.setter
    def y(self, value: pd.Series | None) -> None:
        """Set the loaded target vector.

        Args:
            value: Loaded target payload.
        """
        self._y = value

    @property
    def X(self) -> TabularLike | None:
        """Convenience alias for the loaded feature matrix.

        Returns:
            Loaded feature payload.
        """
        return self._X

    @X.setter
    def X(self, value: TabularLike | None) -> None:
        """Set the loaded feature matrix.

        Args:
            value: Loaded feature payload.
        """
        self._X = value

    @property
    def X_train(self) -> TabularLike | None:
        """Convenience alias for the loaded feature matrix.

        Returns:
            Loaded feature payload.
        """
        return self._X_train

    @X_train.setter
    def X_train(self, value: TabularLike | None) -> None:
        """Set the loaded feature matrix.

        Args:
            value: Loaded feature payload.
        """
        self._X_train = value

    @property
    def X_test(self) -> TabularLike | None:
        """Convenience alias for the loaded feature matrix.

        Returns:
            Loaded feature payload.
        """
        return self._X_test

    @X_test.setter
    def X_test(self, value: TabularLike | None) -> None:
        """Set the loaded feature matrix.

        Args:
            value: Loaded feature payload.
        """
        self._X_test = value

    @property
    def y_train(self) -> pd.Series | None:
        """Convenience alias for the loaded feature matrix.

        Returns:
            Loaded feature payload.
        """
        return self._y_train

    @y_train.setter
    def y_train(self, value: pd.Series | None) -> None:
        """Set the loaded feature matrix.

        Args:
            value: Loaded feature payload.
        """
        self._y_train = value

    @property
    def y_test(self) -> pd.Series | None:
        """Convenience alias for the loaded feature matrix.

        Returns:
            Loaded feature payload.
        """
        return self._y_test

    @y_test.setter
    def y_test(self, value: pd.Series | None) -> None:
        """Set the loaded feature matrix.

        Args:
            value: Loaded feature payload.
        """
        self._y_test = value

    @property
    def X_val(self) -> TabularLike | None:
        """Convenience alias for cached validation features."""
        return self._X_val

    @X_val.setter
    def X_val(self, value: TabularLike | None) -> None:
        """Set cached validation feature payload."""
        self._X_val = value

    @property
    def y_val(self) -> pd.Series | None:
        """Convenience alias for cached validation labels."""
        return self._y_val

    @y_val.setter
    def y_val(self, value: pd.Series | None) -> None:
        """Set cached validation label payload."""
        self._y_val = value

    def load(
        self,
        filepath: Union[str, None] = None,
    ) -> "ArtifactLoaderMixin | ScoreDict | MatrixLike | ArrayLike | EstimatorLike | SerializableValue | None":
        """Load a cached DataConfig object from pickle artifact storage.

        This method does not materialize datasets. Use :meth:`load_dataset` for
        runtime dataset loading from defaults/files.

        Args:
            filepath: Persisted DataConfig artifact path.

        Returns:
            Loaded payload or updated DataConfig instance.

        Raises:
            ValueError: If filepath is not provided.
        """
        if filepath is None:
            raise ValueError("filepath is required for DataConfig.load()")
        loaded = super().load(filepath)
        if isinstance(loaded, DataConfig) and loaded is not self:
            self.__dict__.update(loaded.__dict__)
            return self
        return loaded

    def save(
        self,
        payload: (
            EstimatorLike
            | MatrixLike
            | ArrayLike
            | ScoreDict
            | SerializableValue
            | None
        ) = None,
        filepath: str | None = None,
    ) -> None:
        """Save this DataConfig object as a pickle cache artifact.

        Args:
            payload: Payload to persist.
            filepath: Target file path.

        Raises:
            ValueError: If filepath is not provided.
        """
        target_path = filepath
        if target_path is None and isinstance(payload, (str, Path)):
            target_path = str(payload)
            payload = self
        if payload is None:
            payload = self
        if target_path is None:
            raise ValueError("filepath is required for DataConfig.save()")
        super().save(payload=payload, filepath=target_path)

    def load_raw_data(self) -> tuple[TabularLike | None, pd.Series | None]:
        """Compatibility alias for runtime dataset loading.

        Returns:
            Loaded feature and target payload tuple.
        """
        self.load_dataset()
        return self._X, self._y

    def load_default_dataset(
        self,
        dataset: StringifiedClass,
        **loader_params: Any,
    ) -> MatrixLike | ArrayLike | None:
        """Public default dataset load entry-point delegated to declarations.

        Args:
            dataset: Dataset identifier.
            **loader_params: Dataset-loader specific kwargs.

        Returns:
            Optional loaded payload from declaration loader.
        """
        from .declarations import load_default_dataset

        return load_default_dataset(self, dataset_name=dataset, **loader_params)

    def sample(self, run_hooks: bool = True) -> "DataConfig":
        """Materialize train/test/(optional val) splits for this dataset.

        Args:
            run_hooks: Whether to execute plugin hooks.

        Returns:
            This DataConfig instance.

        Raises:
            TypeError: If configured pipeline runtime is not a DataPipeline object.
        """
        self.load_dataset()
        has_split_payload = all(
            getattr(self, attr, None) is not None
            for attr in ("X_train", "y_train", "X_test", "y_test")
        )
        if (
            not hasattr(self, "data_sample_time")
            or self.data_sample_time is None
            or not has_split_payload
        ):
            self._split_loaded_data(run_hooks=run_hooks)
        pipeline_runtime = getattr(self, "pipeline", None)
        if pipeline_runtime is not None:
            from .pipeline.base import DataPipeline

            if not isinstance(pipeline_runtime, DataPipeline):
                raise TypeError(
                    "DataConfig.pipeline must be a DataPipeline runtime object",
                )
            if getattr(self, "data_pipeline_time", None) is None:
                pipeline_start = time.process_time()
                pipeline_runtime(self)
                self._set_time(
                    "data_pipeline_time",
                    time.process_time() - pipeline_start,
                )
        return self

    def score(
        self,
        *args,
        mode: str | None = None,
        stage: str | None = None,
        **kwargs,
    ) -> ScoreDict:
        """Thin pass-through scoring call delegated to the configured scorer.

        Args:
            *args: Positional scorer args.
            mode: Scoring mode token.
            stage: Optional scoring stage token.
            **kwargs: Additional scorer kwargs.

        Returns:
            Canonical score payload.

        Raises:
            TypeError: If configured scorer is not callable.
        """
        if self.scorer is None:
            return ScoreDict()
        if not callable(self.scorer):
            raise TypeError(
                f"DataConfig.scorer must be callable or None, got {type(self.scorer)}",
            )
        resolved_mode = self._normalize_score_mode(mode or self.score_mode)
        mode_token = str(resolved_mode).strip().lower().replace("_", "-")
        stage_token = str(stage).strip().lower().replace("_", "-")
        y, X = self._resolve_score_payload(mode_token, stage_token)

        scorer_kwargs = {
            "mode": resolved_mode,
            "data": self,
            **kwargs,
        }
        if y is not None and X is not None:
            scorer_kwargs["y"] = y
            scorer_kwargs["X"] = X
        if stage is not None:
            scorer_kwargs["stage"] = stage
        scorer_output = self.scorer(*args, **scorer_kwargs)
        if isinstance(scorer_output, dict):
            return ScoreDict.from_payload(scorer_output)
        return ScoreDict.from_payload({"value": scorer_output})

    @staticmethod
    def _coerce_split_payload_array(
        payload: Any,
        tuple_index: int | None = None,
    ) -> Any:
        if isinstance(payload, (pd.DataFrame, pd.Series)):
            return payload
        if isinstance(payload, np.ndarray):
            return payload
        if hasattr(payload, "__len__") and hasattr(payload, "__getitem__"):
            if len(payload) == 0:
                return np.asarray([])
            first = payload[0]
            if tuple_index is not None and isinstance(first, (tuple, list)):
                values = [sample[tuple_index] for sample in payload]
            else:
                values = list(payload)
            np_values = [np.asarray(value) for value in values]
            try:
                return np.stack(np_values)
            except ValueError:
                return np.asarray(np_values, dtype=object)
        return np.asarray(payload)

    @classmethod
    def _concat_split_arrays(
        cls,
        train_payload: Any,
        test_payload: Any,
        tuple_index: int | None = None,
    ) -> Any:
        if isinstance(train_payload, (pd.DataFrame, pd.Series)):
            return pd.concat([train_payload, test_payload], ignore_index=True)
        train_arr = cls._coerce_split_payload_array(train_payload, tuple_index)
        test_arr = cls._coerce_split_payload_array(test_payload, tuple_index)
        return np.concatenate([train_arr, test_arr])

    def _resolve_all_score_payload(self) -> tuple[Any, Any]:
        y_train = getattr(self, "y_train", None)
        y_test = getattr(self, "y_test", None)
        X_train = getattr(self, "X_train", None)
        X_test = getattr(self, "X_test", None)
        if y_train is None or y_test is None or X_train is None or X_test is None:
            return None, None
        X_all = self._concat_split_arrays(X_train, X_test, tuple_index=0)
        y_all = self._concat_split_arrays(y_train, y_test, tuple_index=1)
        return y_all, X_all

    def _resolve_score_payload(
        self,
        mode_token: str,
        stage_token: str,
    ) -> tuple[Any, Any]:
        if mode_token == "all" or stage_token == "post-pipeline":
            return self._resolve_all_score_payload()
        return resolve_data_split_payload(self, mode_token)

    def __hash__(self):
        return super().__hash__()

    def _encode_binary_series(
        self,
        series: pd.Series,
        mapping: dict[str, int],
    ) -> pd.Series:
        encoded = series.map(mapping)
        if encoded.isna().any():
            unique_values = [
                value for value in series.dropna().unique().tolist() if value != "nan"
            ]
            if len(unique_values) != 2:
                raise ValueError(
                    f"Expected a binary series, found values {sorted(unique_values)}",
                )
            fallback_mapping = {
                value: idx for idx, value in enumerate(sorted(unique_values))
            }
            encoded = series.map(fallback_mapping)
        return encoded.astype(int)

    def _split_loaded_data(
        self,
        run_hooks: bool = True,
    ):
        """
        Samples training, testing, and optionally validation indices from the loaded dataset.

        Delegates sampler resolution/configuration/execution to
        :class:`~deckard.data.sample.BaseSampler`, then materializes
        ``X_train``/``X_test`` and optional ``X_val`` splits.


        Raises:
            ValueError: If data is not loaded, if the specified stratify column
                is not found, or if ``stratify`` is invalid.

        Side Effects
        ------------
        Sets ``self.train_indices``, ``self.test_indices``, ``self.val_indices`` (may be
        ``None``), and ``self.data_sample_time``.
        Logs the time taken for sampling.
        """
        if run_hooks:
            self._run_plugin_hook("before_sample")
        if self._X is None or self._y is None:
            raise ValueError("Data not loaded. Cannot sample.")

        start_time = time.process_time()

        from .sample import BaseSampler

        train_idx, test_idx, val_idx = BaseSampler.execute(self)
        self.train_indices = np.asarray(train_idx, dtype=int).tolist()
        self.test_indices = np.asarray(test_idx, dtype=int).tolist()
        self.val_indices = np.asarray(val_idx, dtype=int).tolist()
        if self.val_indices is not None and len(self.val_indices) > 0:
            val_index = np.asarray(self.val_indices, dtype=int)
            self.X_val = self._X.iloc[val_index].reset_index(drop=True)
            self.y_val = self._y.iloc[val_index].reset_index(drop=True)
            assert self.X_val is not None
            self.val_n = len(self.X_val)

        end_time = time.process_time()
        self._set_time("data_sample_time", end_time - start_time)
        logger.info(f"Data sampled in {self.data_sample_time:.2f} seconds")

        assert self.train_indices is not None and self.test_indices is not None
        train_index = np.asarray(self.train_indices, dtype=int)
        test_index = np.asarray(self.test_indices, dtype=int)
        self._X_train = self._X.iloc[train_index].reset_index(drop=True)
        self._y_train = self._y.iloc[train_index].reset_index(drop=True)
        self._X_test = self._X.iloc[test_index].reset_index(drop=True)
        self._y_test = self._y.iloc[test_index].reset_index(drop=True)
        assert self.X_train is not None and self.X_test is not None
        self.train_n = len(self.X_train)
        self.test_n = len(self.X_test)
        assert isinstance(
            self.X_train,
            (pd.DataFrame, pd.Series),
        ), "X_train must be a DataFrame"
        assert isinstance(self.y_train, pd.Series), "y_train must be a Series"
        assert isinstance(
            self.X_test,
            (pd.DataFrame, pd.Series),
        ), "X_test must be a DataFrame"
        assert isinstance(self.y_test, pd.Series), "y_test must be a Series"
        assert (
            hasattr(self, "train_indices") and self.train_indices is not None
        ), "Train indices must be set after sampling"
        assert (
            hasattr(self, "test_indices") and self.test_indices is not None
        ), "Test indices must be set after sampling"
        assert isinstance(
            self.X_train,
            (pd.DataFrame, pd.Series),
        ), f"X_train must be a DataFrame or Series, got {type(self.X_train)}"
        assert isinstance(
            self.y_train,
            pd.Series,
        ), f"y_train must be a Series, got {type(self.y_train)}"
        assert isinstance(
            self.X_test,
            (pd.DataFrame, pd.Series),
        ), f"X_test must be a DataFrame or Series, got {type(self.X_test)}"
        assert isinstance(
            self.y_test,
            pd.Series,
        ), f"y_test must be a Series, got {type(self.y_test)}"
        if run_hooks:
            self._run_plugin_hook("after_sample")

    def _load_dataset_with_hooks(
        self,
        load_fn: Callable[[], None],
    ) -> "DataConfig":
        """Run canonical dataset-load hook orchestration around a loader callable."""
        if hasattr(self, "data_load_time") and self.data_load_time is not None:
            return self
        self._run_plugin_hook("before_load_data")
        load_fn()
        self._run_plugin_hook("after_load_data")
        return self

    def load_dataset(self) -> None:
        """Load dataset payload based on configured dataset source or file type.

        Supported datasets (without optional dependencies)
        --------------------------------------------------
        - adult
        - make_classification
        - make_regression
        - diabetes
        - digits
        - iris
        - wine
        - breast_cancer
        - california_housing
        - olivetti_faces
        - lfw_people
        - lfw_pairs
        - 20newsgroups
        - 20newsgroups_vectorized

        Raises:
            NotImplementedError: If dataset source type is unsupported.
            TypeError: If loaded dataset payload has unsupported structure.
        """
        f"""
        Loads dataset based on the provided dataset name or file type.

        Supported datasets (without optional dependencies)
        --------------------------------------------------
        - "adult"
        - "make_classification"
        - "make_regression"
        - "diabetes"
        - "digits"
        - "iris"
        - "wine"
        - "breast_cancer"
        - "california_housing"
        - "olivetti_faces"
        - "lfw_people"
        - "lfw_pairs"
        - "20newsgroups"
        - "20newsgroups_vectorized"
        - "covtype"
        - "kddcup99"
        - "rcv1"
        - "sample_image"
        - "make_blobs"
        - "make_moons"
        - "make_circles"
        - "make_multilabel_classification"
        - "make_hastie_10_2"
        - "make_friedman1"
        - "make_friedman2"
        - "make_friedman3"
        - "make_sparse_coded_signal"
        - "make_sparse_spd_matrix"
        - "make_spd_matrix"
        - "make_low_rank_matrix"
        - "make_s_curve"
        - "make_swiss_roll"
        - "fetch_20newsgroups"
        - "fetch_20newsgroups_vectorized"
        - "fetch_california_housing"
        - "fetch_covtype"
        - "fetch_kddcup99"
        - "fetch_lfw_people"
        - "fetch_lfw_pairs"
        - "fetch_olivetti_faces"
        - "fetch_rcv1"
        - "load_breast_cancer"
        - "load_diabetes"
        - "load_digits"
        - "load_iris"
        - "load_wine"

        Supported file types
        --------------------
                - Dataset/load dispatch (DataConfig):
                    {", ".join(CANONICAL_DATASET_LOAD_FILETYPES)}
                - Artifact save (ArtifactLoaderMixin.save_data):
                    {", ".join(CANONICAL_DATA_SAVE_FILETYPES)}
                - Artifact load (ArtifactLoaderMixin.load_data):
                    {", ".join(CANONICAL_DATA_LOAD_FILETYPES)}

        For built-in datasets, calls the corresponding loader method.
        Updates ``self._X``, ``self._y``, and ``self.data_load_time`` with loaded data and timing information.

        Raises:
            NotImplementedError: If dataset name or file type is unsupported.
            ValueError: If required runtime target/source fields are invalid.
        """
        if (
            hasattr(self, "data_load_time")
            and self.data_load_time is not None
            and getattr(self, "_X", None) is not None
            and getattr(self, "_y", None) is not None
        ):
            return
        self._run_plugin_hook("before_load_data")
        from .declarations import build_loader_registry, load_adult_income_data

        dataset_name = str(self.resolve_name(default="") or "")
        if dataset_name in ["adult", ""]:
            return load_adult_income_data(self, **self.data_params)
        supported_datasets = build_loader_registry(self)
        if dataset_name == "":
            raise ValueError("DataConfig.name must be set before loading data")
        self.name = dataset_name
        filetype = Path(dataset_name).suffix
        supported_filetypes = CANONICAL_DATASET_LOAD_FILETYPES
        is_optuna_source = (
            dataset_name.strip().lower() == "optuna"
            or filetype in {".db", ".sqlite3"}
            or "optuna_storage" in (self.data_params or {})
            or "study_name" in (self.data_params or {})
        )
        if (
            not is_optuna_source
            and filetype not in supported_filetypes
            and dataset_name not in supported_datasets
        ):
            raise NotImplementedError(
                f"Currently only {supported_filetypes} filetypes are supported for loading data. Cannot load {dataset_name}",
            )
        if is_optuna_source:
            start_time = time.process_time()
            self._load_from_optuna_storage()
            end_time = time.process_time()
            self._set_time("data_load_time", end_time - start_time)
        elif dataset_name in supported_datasets:
            start_time = time.process_time()
            self.load_default_dataset(dataset_name, **self.data_params)
        elif filetype == ".openml":
            start_time = time.process_time()
            dataset_base_name = Path(dataset_name).stem
            from .declarations import load_generic_openml

            load_generic_openml(
                self,
                dataset_name=dataset_base_name,
                **self.data_params,
            )
        elif filetype in supported_filetypes:
            start_time = time.process_time()
            self._load_from_csv(dataset_name=dataset_name, **self.data_params)
            end_time = time.process_time()
            self._set_time("data_load_time", end_time - start_time)
        else:
            raise NotImplementedError(
                f"Dataset {dataset_name} not implemented",
            )

        assert isinstance(
            self._X,
            (pd.DataFrame, pd.Series),
        ), "_X must be a DataFrame after loading data"
        assert isinstance(
            self._y,
            pd.Series,
        ), "_y must be a Series after loading data"
        self._apply_max_samples()
        self._run_plugin_hook("after_load_data")
        logger.info(
            f"Data loaded from {dataset_name} in {self.data_load_time:.2f} seconds",
        )

    def _load_from_csv(self, dataset_name: str | None = None):
        dataset_name = dataset_name or ""
        print(dataset_name)
        data = pd.DataFrame(cast(Any, self.load_data(dataset_name)))
        if self.target is None:
            raise ValueError(
                "CSV file must contain a 'target' column or specify the target column name in the 'target' attribute",
            )
        y = data.pop(self.target)
        if len(self.keep) > 1:
            data = data[self.keep]
        elif len(self.keep) == 1:
            data = data[self.keep[0]]
        for del_col in self.drop:
            assert len(self.keep) == 0, "Cannot specify both keep and drop columns"
            if del_col in data.columns:
                data = data.drop(columns=del_col)
        self._X = data
        self._y = y

    def _load_from_optuna_storage(self):
        dataset_name = str(self.resolve_name(default="") or "")
        data_params = dict(self.data_params or {})
        storage = data_params.pop("optuna_storage", None)
        if storage is None:
            storage = data_params.pop("storage", None)
        if storage is None and dataset_name.strip().lower() != "optuna":
            storage = dataset_name
        if storage is None:
            storage = "sqlite:///optuna.db"

        study_name = data_params.pop("study_name", None)
        schema = data_params.pop("schema", None)
        data = pd.DataFrame(
            _load_optuna_studies_dataframe(
                storage=storage,
                study_name=study_name,
                schema=schema,
                study_names=data_params.pop("study_names", None),
                columns=data_params.pop("columns", None),
                include_columns=data_params.pop("include_columns", None),
                exclude_columns=data_params.pop("exclude_columns", None),
                trial_numbers=data_params.pop("trial_numbers", None),
                trial_number_range=data_params.pop("trial_number_range", None),
                trial_states=data_params.pop("trial_states", None),
                row_slice=data_params.pop("row_slice", None),
                sort_by=data_params.pop("sort_by", None),
                ascending=bool(data_params.pop("ascending", True)),
                offset=int(data_params.pop("offset", 0)),
                limit=data_params.pop("limit", None),
            ),
        )
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Optuna study query returned no rows.")

        target_col = self.target or data_params.pop("target", None)
        if target_col is None:
            if "value" in data.columns:
                target_col = "value"
            else:
                value_cols = [c for c in data.columns if str(c).startswith("value")]
                if len(value_cols) > 0:
                    target_col = value_cols[0]

        if target_col is None or str(target_col) not in data.columns:
            raise ValueError(
                "Optuna-backed DataConfig requires a valid target column (set target=... or data_params.target).",
            )

        y = data.pop(str(target_col))
        if len(self.keep) > 1:
            data = data[self.keep]
        elif len(self.keep) == 1:
            data = data[self.keep[0]]
        for del_col in self.drop:
            assert len(self.keep) == 0, "Cannot specify both keep and drop columns"
            if del_col in data.columns:
                data = data.drop(columns=del_col)
        self._X = data
        self._y = y

    def apply_pipeline(self, pipeline: "DataPipeline | list | None") -> "DataConfig":
        """Attach a pipeline-like plugin object to this data config.

        Args:
            pipeline: Pipeline object or list of pipeline plugins.

        Returns:
            This DataConfig instance.
        """
        if pipeline is None:
            return self
        pipeline_plugins = (
            [pipeline] if not isinstance(pipeline, list) else list(pipeline)
        )
        existing_plugins = list(self.plugins or [])
        self.plugins = [*pipeline_plugins, *existing_plugins]
        return self

    def build_data_time_dict(self) -> dict:
        """Build timing/count metadata dictionary for data runtime outputs.

        Returns:
            Runtime timing/count metadata mapping.
        """
        self._sync_canonical_time_state()
        time_dict = dict(self.times)
        time_dict["train_n"] = self.train_n
        time_dict["test_n"] = self.test_n
        if self.val_n is not None:
            time_dict["val_n"] = self.val_n
        return time_dict

    def _prepare_files(self, files: DataFiles | None = None) -> bool:
        """Merge file aliases and optionally load existing cached runtime.

        Returns True when this run should persist runtime state to data_file.
        """
        self.files = merge_data_files(self.files, files)
        data_file = self.files.get("data_file")
        if data_file is None:
            return False
        data_path = Path(data_file)
        if data_path.exists():
            # Repeated split/fold runs intentionally mutate ``split`` and clear
            # sample timing/state between iterations; loading a cached artifact
            # here can restore stale timing fields and skip re-sampling.
            explicit_split = getattr(self, "split", None) not in [None, ""]
            resample_requested = getattr(self, "data_sample_time", None) is None
            if explicit_split or resample_requested:
                return True
            self.load(str(data_path))
            self._sync_canonical_time_state()
            return False
        data_path.parent.mkdir(parents=True, exist_ok=True)
        return True

    def resolve_call_files(
        self,
        kwargs: dict[str, Any],
        files: DataFiles | DictConfig | dict[str, Any] | None = None,
    ) -> DataFiles:
        """Resolve canonical runtime files payload from explicit and legacy kwargs.

        Args:
            kwargs: Mutable runtime kwargs payload.
            files: Optional explicit files mapping.

        Returns:
            Canonical runtime files payload.
        """
        files_payload = files if isinstance(files, (dict, DictConfig)) else None
        return resolve_runtime_files(kwargs, files_payload)

    def execute_data_runtime(
        self,
        *args: Any,
        files: DataFiles | DictConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Execute canonical DataConfig call flow with normalized files/pipeline runtime.

        Args:
            *args: Positional runtime payloads forwarded to ``__call__``.
            files: Optional files payload to resolve into canonical aliases.
            **kwargs: Keyword runtime payloads forwarded to ``__call__``.

        Returns:
            Runtime score payload.
        """
        runtime_files = self.resolve_call_files(kwargs, files=files)
        self._coerce_pipeline_runtime()
        return DataConfig.__call__(self, *args, files=runtime_files, **kwargs)

    def __call__(
        self,
        *args,
        files: DataFiles | None = None,
        **kwargs,
    ) -> ScoreDict:
        """
        Loads and samples the dataset, splits it into training and testing sets, and returns timing and scoring information.
        Strictly validates that all output values are flat and serializable.

        Args:
            *args: Positional scoring args.
            files: Optional runtime file aliases.
            **kwargs: Runtime scoring kwargs.

        Returns:
            Runtime score/timing payload dictionary.

        Raises:
            TypeError: If legacy file kwargs are passed outside files mapping.
        """

        if "data_file" in kwargs or "score_file" in kwargs:
            raise TypeError(
                "DataConfig.__call__ uses files-only persistence. "
                "Pass file aliases via files={data_file: ..., score_file: ...}.",
            )
        self.files = merge_data_files(self.files, files)
        save_flag = self._prepare_files(files=self.files)
        self._sync_canonical_time_state()
        score_file = self.files.get("score_file")
        data_file = self.files.get("data_file")
        scores = dict(ScoreDict.from_payload(getattr(self, "score_dict", {}) or {}))
        self._score_orchestration_active = True
        score_hook_elapsed = 0.0
        try:
            self.load_dataset()
            logger.info(f"Data loaded in {self.data_load_time:.2f} seconds")
            self.sample()

            score_hook_start = time.process_time()
            self._run_plugin_hook("after_pipeline", score_kwargs=kwargs)
            score_hook_elapsed = time.process_time() - score_hook_start
        finally:
            self._score_orchestration_active = False

        existing_score_time = getattr(self, "data_score_time", None)
        if existing_score_time is None and score_hook_elapsed > 0.0:
            self._set_time("data_score_time", score_hook_elapsed)

        time_dict = self.build_data_time_dict()
        assert self.X_train is not None and self.X_test is not None
        if self.X_val is not None:
            logger.info(
                f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}, "
                f"Val set size: {len(self.X_val)}",
            )
        else:
            logger.info(
                f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}",
            )
        data_scores = dict(
            ScoreDict.from_payload(getattr(self, "score_dict", {}) or {}),
        )
        if len(data_scores) == 0:
            score_start = time.process_time()
            data_scores = self.score(*args, **kwargs)
            score_elapsed = time.process_time() - score_start
            combined_elapsed = score_hook_elapsed + score_elapsed
            if existing_score_time is None or combined_elapsed > 0.0:
                self._set_time("data_score_time", combined_elapsed)

        self._sync_canonical_time_state()
        time_dict = self.build_data_time_dict()
        all_scores = {**scores, **data_scores, **time_dict}
        self.score_dict = ScoreDict.from_payload(all_scores)
        self.times.update({k: all_scores.get(k) for k in CANONICAL_DATA_TIMES})
        assert hasattr(self, "score_dict"), "score_dict must be set"
        self.merge_runtime_files(
            cast(dict[str, RuntimeSerializable], dict(self.files)),
            {
                "data_file": data_file,
                "score_file": score_file,
            },
        )
        all_scores = dict(ScoreDict.from_payload(self.score_dict))
        all_scores = self.merge_and_persist_scores(all_scores, score_file)
        self.score_dict = ScoreDict.from_payload(all_scores)
        if save_flag:
            self.save(filepath=data_file)
        return self.score_dict
