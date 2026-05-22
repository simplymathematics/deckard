# Imports
import os
import pandas as pd
import time
import logging
from pathlib import Path

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal,Union, Optional
from omegaconf import DictConfig, ListConfig

import numpy as np

# deckard
from ..utils import (
    ConfigBase,
    data_supported_filetypes,
    load_class,
    coerce_to_list,
    merge_list_of_dicts,
)
from ..frameworks.types import ArrayLike, MatrixLike
from ..plugins.base import PluginOrchestratorMixin
from .canon import (
    DEFAULT_DATA_SCORE_STAGE,
    CANONICAL_DATA_TIMES,
    DataFiles,
    ensure_data_runtime_contract,
    ensure_canonical_times,
    merge_data_files,
    normalize_data_score_mode,
    stage_hook_token,
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





@dataclass(eq=False, kw_only=True)
class DataConfig(PluginOrchestratorMixin, ConfigBase):
    """
    Configuration and utility class for loading, preprocessing, and splitting datasets for machine learning tasks.

    Attributes
    -------
    dataset_name : str
        Name of the dataset to load or path to a data file.
    data_params : dict
        Additional parameters for data loading or generation.
    test_size : float
        Proportion of the dataset to include in the test split (between 0 and 1).
    train_size : float
        Proportion or count of samples to include in the training split.
    val_size : Union[float, int, None]
        Proportion or count of samples to include in the validation split when a
        ``sampler`` is provided (e.g. :class:`~deckard.data.sample.SplitSampler` or
        :class:`~deckard.data.sample.ShuffleSampler`).  Unused in legacy mode.
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
    random_state : int
        Seed for random number generation to ensure reproducibility.
    stratify : Union[None, str, bool]
        Specifies stratification for sampling; can be None, True (use target), or a column name.
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
    ----------------------------------
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

    Plugin hook runtime params
    --------------------------
    Hooks are orchestrated by ``_run_plugin_hook(hook_name, **kwargs)``.
    Core hook names used by DataConfig runtime are:
    ``before_load_data``, ``after_load_data``, ``before_sample``, ``before_pipeline``, `after_pipeline``
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
    Raises
    ------
    ValueError
        For invalid parameter values or missing data.
    NotImplementedError
        For unsupported datasets or file types.

    Examples
    --------
    >>> config = DataConfig(dataset_name="adult", **kwargs)
    >>> config()
    >>> X_train = config.X_train
    >>> y_train = config.y_train
    >>> X_test = config.X_test
    >>> y_test = config.y_test
    >>> score_dict = config.score_dict

    Using a pluggable sampler for 3-way splits::

        from deckard.data.sample import SplitSampler
        config = DataConfig(
            dataset_name="digits",
            test_size=0.2,
            val_size=0.1,
            sampler=SplitSampler(),
        )
        config()
        X_val, y_val = config.X_val, config.y_val
    """

    # Configuration fields
    dataset_name: str = "adult"
    data_params: dict = None
    test_size: Union[float, int, None] = 0.2
    train_size: Union[float, int, None] = None
    val_size: Union[float, int, None] = None
    split: Union[int, None] = None
    sampler: Union["BaseSampler", Literal["split", "shuffle", "fold"], dict, None] = "split"
    random_state: Union[int, None] = 0
    stratify: Union[None, str, bool] = None
    classifier: Union[bool, str] = True
    target: Union[str, None] = None
    drop: list = None
    keep: list = None
    plugins: list = field(default_factory=list)
    alias: Union[str, None] = None
    scorer: Any = AUTO_SCORER
    score_split: str = "test"
    score_mode: str = DEFAULT_DATA_SCORE_STAGE
    pipeline: "DataPipeline | None" = None
    files: DataFiles = field(default_factory=lambda: {})

    # Runtime state fields
    score_dict: dict = field(default_factory=dict, init=False, repr=True)
    times: dict[str, Any] = field(default_factory=dict)
    data_load_time: Union[float, None] = None
    data_sample_time: Union[float, None] = None
    _X: Union[pd.DataFrame, pd.Series, None] = None
    _y: Union[pd.Series, None] = None
    train_indices: Union[list, None] = None
    test_indices: Union[list, None] = None
    val_indices: Union[list, None] = None
    X_train: Union[pd.DataFrame, pd.Series, None] = None
    y_train: Union[pd.Series, None] = None
    X_test: Union[pd.DataFrame, pd.Series, None] = None
    y_test: Union[pd.Series, None] = None
    X_val: Union[pd.DataFrame, pd.Series, None] = None
    y_val: Union[pd.Series, None] = None
    train_n: Union[int, None] = None
    test_n: Union[int, None] = None
    val_n: Union[int, None] = None
    _target_: Union[str, None] = None
    _plugin_objects: Union[list, None] = None
    _sampler_obj: Union[callable, None] = None
    _score_orchestration_active: bool = field(default=False, init=False, repr=False)

    def _normalize_score_mode(self, mode: str) -> str:
        return normalize_data_score_mode(mode)

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
        if self.train_size is None:
            if self.test_size is None:
                self.test_size = 0.2
                self.train_size = 0.8
            else:
                if isinstance(self.test_size, float):
                    if not (0 < self.test_size < 1):
                        raise ValueError("test_size must be between 0 and 1")
                    self.train_size = 1 - self.test_size
                elif isinstance(self.test_size, int):
                    self.train_size = None
                else:
                    raise ValueError("test_size must be a float or int")
        self.data_params = self.data_params if self.data_params is not None else {}
        self.score_mode = (
            getattr(self, "score_mode", DEFAULT_DATA_SCORE_STAGE)
            or DEFAULT_DATA_SCORE_STAGE
        )
        self.score_split = normalize_data_score_mode(
            getattr(self, "score_split", "test") or "test",
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

        if not hasattr(self, "_target_") or self._target_ is None:
            self._target_ = "deckard.data.DataConfig"
        if not self.data_params:
            self.data_params = {}

    def _set_time(self, key: str, value: float | None) -> None:
        self.times[key] = value
        setattr(self, key, value)

    def _resolve_max_samples(self, dataset_len: int) -> Union[int, None]:
        """Resolve an optional dataset cap from the test-only environment variable."""
        max_samples = os.environ.get(DECKARD_TEST_MAX_SAMPLES_ENV)
        if max_samples in [None, ""]:
            return None
        try:
            max_samples = int(max_samples)
        except (TypeError, ValueError):
            raise ValueError(
                f"{DECKARD_TEST_MAX_SAMPLES_ENV} must be an integer, got {max_samples}",
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
                    "deckard.score.data.DefaultDataClassificationConfig"
                    if self.classifier
                    else "deckard.score.data.DefaultDataRegressionConfig"
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
            return
        if isinstance(raw_pipeline, (dict, DictConfig)):
            self.pipeline = DataPipeline(pipeline=dict(raw_pipeline))
            return
        if isinstance(raw_pipeline, (list, ListConfig)):
            merged = merge_list_of_dicts(coerce_to_list(raw_pipeline))
            self.pipeline = DataPipeline(pipeline=dict(merged))
            return
        raise TypeError(
            "DataConfig.pipeline must be DataPipeline | dict | list | None, "
            f"got {type(raw_pipeline)}",
        )

    def _initialize_runtime_components(self) -> None:
        """Instantiate runtime-bound plugin and sampler objects eagerly."""
        from .sample import BaseSampler

        self._plugin_objects = self._get_plugins()
        self._configure_score_orchestration_plugins()
        self._sampler_obj = BaseSampler.resolve(self)

    @property
    def X(self) -> MatrixLike | None:
        """Convenience alias for the loaded feature matrix."""
        return self._X

    @property
    def scores(self) -> dict[str, Any]:
        """Canonical score container alias (backed by ``score_dict``)."""
        return self.score_dict

    @scores.setter
    def scores(self, value: dict[str, Any]) -> None:
        self.score_dict = value

    @property
    def y(self) -> ArrayLike | None:
        """Convenience alias for the loaded target vector."""
        return self._y

    @X.setter
    def X(self, value: MatrixLike | None) -> None:
        """Set the loaded feature matrix."""
        self._X = value

    @y.setter
    def y(self, value: ArrayLike | None) -> None:
        """Set the loaded target vector."""
        self._y = value

    def load(self, filepath: Union[str, None] = None):
        """Load a cached DataConfig object from pickle artifact storage.

        This method does not materialize datasets. Use :meth:`load_dataset` for
        runtime dataset loading from defaults/files.
        """
        if filepath is None:
            raise ValueError("filepath is required for DataConfig.load()")
        loaded = super().load(filepath)
        if isinstance(loaded, DataConfig) and loaded is not self:
            self.__dict__.update(loaded.__dict__)
            return self
        return loaded

    def save(self, payload: Any = None, filepath: str | None = None) -> None:
        """Save this DataConfig object as a pickle cache artifact."""
        target_path = filepath
        if target_path is None and isinstance(payload, (str, Path)):
            target_path = str(payload)
            payload = self
        if payload is None:
            payload = self
        if target_path is None:
            raise ValueError("filepath is required for DataConfig.save()")
        super().save(payload=payload, filepath=target_path)

    def load_raw_data(self) -> tuple[MatrixLike, ArrayLike]:
        """Compatibility alias for runtime dataset loading."""
        self.load_dataset()
        return self._X, self._y

    def load_default_dataset(self, dataset_name: str, **loader_params: Any):
        """Public default dataset load entry-point delegated to declarations."""
        from .declarations import load_default_dataset

        return load_default_dataset(self, dataset_name=dataset_name, **loader_params)



    def fit(self, run_hooks: bool = True) -> "DataConfig":
        """Materialize train/test/(optional val) splits for this dataset."""
        self.load_dataset()
        if not hasattr(self, "data_sample_time") or self.data_sample_time is None:
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
                self._set_time("data_pipeline_time", time.process_time() - pipeline_start)
        return self

    def sample(self, run_hooks: bool = True) -> "DataConfig":
        """Public sampling lifecycle method.

        Mirrors ``score/scorer`` naming with ``sample/sampler``.
        """
        self.fit(run_hooks=run_hooks)
        return self

    def score(
        self,
        *args,
        mode: str | None = None,
        stage: str | None = None,
        **kwargs,
    ) -> dict:
        """Thin pass-through scoring call delegated to the configured scorer."""
        if self.scorer is None:
            return {}
        if not callable(self.scorer):
            raise TypeError(
                f"DataConfig.scorer must be callable or None, got {type(self.scorer)}",
            )
        resolved_mode = normalize_data_score_mode(mode or self.score_split)
        mode_token = str(resolved_mode).strip().lower().replace("_", "-")

        y = None
        X = None
        if mode_token == "train":
            y = getattr(self, "y_train", None)
            X = getattr(self, "X_train", None)
        elif mode_token == "test":
            y = getattr(self, "y_test", None)
            X = getattr(self, "X_test", None)
        elif mode_token == "val":
            y = getattr(self, "y_val", None)
            X = getattr(self, "X_val", None)
        elif mode_token == "all":
            y_train = getattr(self, "y_train", None)
            y_test = getattr(self, "y_test", None)
            X_train = getattr(self, "X_train", None)
            X_test = getattr(self, "X_test", None)
            if y_train is not None and y_test is not None and X_train is not None and X_test is not None:
                if isinstance(X_train, (pd.DataFrame, pd.Series)):
                    X = pd.concat([X_train, X_test], ignore_index=True)
                else:
                    X = np.concatenate([np.asarray(X_train), np.asarray(X_test)])
                if isinstance(y_train, (pd.DataFrame, pd.Series)):
                    y = pd.concat([y_train, y_test], ignore_index=True)
                else:
                    y = np.concatenate([np.asarray(y_train), np.asarray(y_test)])

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
        return self.scorer(*args, **scorer_kwargs)

    
    
    
    

    def _get_stratify_col(self, stratify: Union[None, str, bool] = None):
        """Return the stratification array (or ``None``) based on a stratify value.

        Returns
        -------
        pd.Series or None
            The column to stratify on, or ``None`` if stratification is disabled.

        Raises
        ------
        ValueError
            If ``stratify`` is a string that is not a column name in ``self._X``,
            or if ``stratify`` is an unrecognized type.
        """
        if stratify is None:
            stratify = getattr(self, "stratify", None)
        if stratify is None or stratify is False:
            return None
        if stratify is True:
            if self.classifier is False:
                return None
            return self._y
        if isinstance(stratify, str):
            if self._X is not None and stratify in self._X.columns:
                return self._X[stratify]
            raise ValueError(
                f"Stratify column '{stratify}' not found in data columns",
            )
        raise ValueError("stratify must be None, True, False, or a column name")

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

        Raises
        ------
        ValueError
            If data is not loaded, or if the specified stratify column is not found, or if ``stratify`` is invalid.

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
        self.train_indices = train_idx
        self.test_indices = test_idx
        self.val_indices = val_idx
        if len(val_idx) > 0:
            self.X_val = self._X.iloc[self.val_indices].reset_index(drop=True)
            self.y_val = self._y.iloc[self.val_indices].reset_index(drop=True)
            self.val_n = len(self.X_val)

        end_time = time.process_time()
        self._set_time("data_sample_time", end_time - start_time)
        logger.info(f"Data sampled in {self.data_sample_time:.2f} seconds")

        self.X_train = self._X.iloc[self.train_indices].reset_index(drop=True)
        self.y_train = self._y.iloc[self.train_indices].reset_index(drop=True)
        self.X_test = self._X.iloc[self.test_indices].reset_index(drop=True)
        self.y_test = self._y.iloc[self.test_indices].reset_index(drop=True)
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

    def load_dataset(self):
        """
        Loads dataset based on the provided dataset name or file type.

        Supported datasets (without optional dependencies)
        ------------------
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
        - ".csv" (must contain a 'target' column)

        For built-in datasets, calls the corresponding loader method.
        For CSV files, reads the file and splits features and target.
        Raises NotImplementedError for unsupported datasets or file types.
        Updates ``self._X``, ``self._y``, and ``self.data_load_time`` with loaded data and timing information.

        Raises
        ------
        NotImplementedError
            If the dataset or file type is not supported.
        ValueError
            If a CSV file does not contain a 'target' column.
        """
        if hasattr(self, "data_load_time") and self.data_load_time is not None:
            return
        self._run_plugin_hook("before_load_data")
        from .declarations import build_loader_registry

        supported_datasets = build_loader_registry(self)
        filetype = Path(self.dataset_name).suffix
        supported_filetypes = data_supported_filetypes
        if (
            filetype not in supported_filetypes
            and self.dataset_name not in supported_datasets
        ):
            raise NotImplementedError(
                f"Currently only {supported_filetypes} filetypes are supported for loading data. Cannot load {self.dataset_name}",
            )
        if self.dataset_name in supported_datasets:
            start_time = time.process_time()
            self.load_default_dataset(self.dataset_name, **self.data_params)
        elif filetype == ".openml":
            start_time = time.process_time()
            dataset_base_name = Path(self.dataset_name).stem
            from .declarations import load_generic_openml

            load_generic_openml(
                self,
                dataset_name=dataset_base_name,
                **self.data_params,
            )
        elif filetype in supported_filetypes:
            start_time = time.process_time()
            self._load_from_csv(**self.data_params)
            end_time = time.process_time()
            self._set_time("data_load_time", end_time - start_time)
        else:
            raise NotImplementedError(
                f"Dataset {self.dataset_name} not implemented",
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
            f"Data loaded from {self.dataset_name} in {self.data_load_time:.2f} seconds",
        )

    def _load_from_csv(self):
        data = self.load_data(self.dataset_name)
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



    def apply_pipeline(self, pipeline) -> "DataConfig":
        """Attach a pipeline-like plugin object to this data config."""
        if pipeline is None:
            return self
        pipeline_plugins = (
            [pipeline] if not isinstance(pipeline, list) else list(pipeline)
        )
        existing_plugins = list(self.plugins or [])
        self.plugins = [*pipeline_plugins, *existing_plugins]
        return self


    def build_data_time_dict(self) -> dict:
        """Build timing/count metadata dictionary for data runtime outputs."""
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
            self.load(str(data_path))
            return False
        data_path.parent.mkdir(parents=True, exist_ok=True)
        return True
    
    def __call__(
        self,
        *args,
        files: DataFiles | None = None,
        **kwargs,
    ) -> dict:
        """
        Loads and samples the dataset, splits it into training and testing sets, and returns timing and scoring information.
        Strictly validates that all output values are flat and serializable.
        """

        if "data_file" in kwargs or "score_file" in kwargs:
            raise TypeError(
                "DataConfig.__call__ uses files-only persistence. "
                "Pass file aliases via files={data_file: ..., score_file: ...}.",
            )
        self.files = merge_data_files(self.files, files)
        save_flag = self._prepare_files(files=self.files)
        score_file = self.files.get("score_file")
        data_file = self.files.get("data_file")
        scores = dict(getattr(self, "score_dict", {}) or {})
        self._score_orchestration_active = True
        try:
            self.load_dataset()
            logger.info(f"Data loaded in {self.data_load_time:.2f} seconds")
            self.fit()
            self._run_plugin_hook("after_pipeline", score_kwargs=kwargs)
        finally:
            self._score_orchestration_active = False
        time_dict = self.build_data_time_dict()
        if self.X_val is not None:
            logger.info(
                f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}, "
                f"Val set size: {len(self.X_val)}",
            )
        else:
            logger.info(
                f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}",
            )
        data_scores = dict(getattr(self, "score_dict", {}) or {})
        if len(data_scores) == 0:
            data_scores = self.score(*args, **kwargs)
        all_scores = {**scores, **data_scores, **time_dict}
        self.score_dict = all_scores
        self.times.update({k: all_scores.get(k) for k in CANONICAL_DATA_TIMES})
        assert hasattr(self, "score_dict"), "score_dict must be set"
        self.merge_runtime_files(
            self.files,
            {
                "data_file": data_file,
                "score_file": score_file,
            },
        )
        all_scores = dict(self.score_dict)
        all_scores = self.merge_and_persist_scores(all_scores, score_file)
        self.score_dict = all_scores
        if save_flag:
            self.save(data_file)
        return self.score_dict


@dataclass
class DataPipelineStep:
    """
    Represents a step in a data pipeline with optional metadata.

    This dataclass normalizes and documents the optional parameters that
    configure how a step integrates with the pipeline runtime.

    Attributes
    ----------
    name : str
        Fully-qualified class name or file path (e.g., "sklearn.preprocessing.StandardScaler"
        or "path/to/custom.py:CustomTransformer").
    fit_y : bool, default=False
        If True, this step is applied to the target (y) rather than features (X).
        Only one of fit_y or fit_xy can be True.
    dtype : str, optional
        Column selector hint for ColumnTransformer. Supported values: "numeric",
        "num", "float", "int", "object", "string", "category". When specified,
        only columns of matching dtype are passed to this step.
    plugin_hook : str | list[str], optional
        Hook name(s) that trigger when this step runs. Supported hooks:
        - "before_sample": Run this step's pre_sample_fit before KFold sampling.
    """

    name: str
    fit_y: bool = False
    fix_Xy: bool = False
    fit_X: bool = True
    fit_pre_sample: bool = False
    dtype: Optional[str] = None
    plugin_hook: Union[str, list, None] = None

    @classmethod
    def from_config(cls, step_name: str, step_config: dict) -> "DataPipelineStep":
        """
        Extract DataPipelineStep from a pipeline step config dict.

        Parameters
        ----------
        step_name : str
            Name of the step (key in pipeline.steps dict).
        step_config : dict
            Configuration dict for the step.

        Returns
        -------
        DataPipelineStep
            Parsed step metadata.

        Raises
        ------
        ValueError
            If "name" key is missing or if both fit_y and fit_xy are True.
        """
        step_class = step_config.get("name")
        if step_class is None:
            raise ValueError(f"Step {step_name} missing required 'name' key")

        fit_y = step_config.get("fit_y", False)
        fit_xy = step_config.get("fit_xy", False)

        if fit_xy is True:
            raise ValueError("fit_xy pipeline steps are no longer supported.")

        return cls(
            name=step_class,
            fit_y=fit_y,
            dtype=step_config.get("dtype", None),
            plugin_hook=step_config.get("plugin_hook", None),
        )

    def stripped_config(self, step_config: dict) -> dict:
        """
        Remove DataPipelineStep metadata from a step config dict.

        Parameters
        ----------
        step_config : dict
            Original step configuration.

        Returns
        -------
        dict
            New dict with metadata keys removed, ready for transformer instantiation.
        """
        config = dict(step_config)
        step_kwargs = {"name", "fit_y", "fit_Xy", "fit_X", "fit_pre_sample", "dtype", "plugin_hook"}
        for key in step_kwargs:
            config.pop(key, None)
        return config


@dataclass(eq=False, kw_only=True)
class DataPipelineConfig(DataConfig):
    """Legacy alias for DataConfig.

    Historical pipeline-specific behavior now lives in DataConfig via the
    optional ``pipeline`` runtime attribute.
    """

    pass


