import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Union, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from deckard.plugins import HookPlugin

from ...data._mixins import RuntimePayload, _SensitiveColumnsMixin
from ...data.base import DataPipelineConfig
from ...utils import (
    is_default_config_value,
    load_class,
    resolve_class,
)
from ...utils import (
    normalize_optional_list_value as _normalize_optional_list_value,
)
from ...utils import (
    normalize_optional_mapping_or_steps as _normalize_optional_mapping_or_steps,
)

RuntimeScalar = str | int | float | bool | None
RuntimeValue = RuntimeScalar | list["RuntimeValue"] | dict[str, "RuntimeValue"]


@dataclass(eq=False, kw_only=True)
class _PrivacyBehaviorMixin:
    """Reusable privacy behavior mixed into data pipeline configs."""

    anjana_defense: Union[None, bool, Dict[str, Any], list] = None
    identifiers: Optional[Union[str, list]] = None
    quasi_identifiers: Optional[Union[str, list]] = None
    sensitive_attribute: Optional[str] = None
    hierarchies: Optional[Dict[str, Dict[int, Any]]] = None
    hierarchy_interval_sizes: Optional[Dict[str, Union[int, list]]] = None
    hierarchy_fill_value: str = "*"

    @staticmethod
    def _normalize_optional_list(
        value: Optional[Union[str, list]],
    ) -> Optional[list]:
        return _normalize_optional_list_value(value, field_name="optional_list")

    @staticmethod
    def _normalize_optional_mapping_or_steps(
        value: Union[None, bool, Dict[str, Any], list],
        *,
        field_name: str,
    ) -> Union[None, bool, Dict[str, Any]]:
        return cast(
            Union[None, bool, Dict[str, Any]],
            _normalize_optional_mapping_or_steps(
                value,
                field_name=field_name,
            ),
        )

    def _before_post_init(self) -> None:
        self.anjana_defense = self._normalize_optional_mapping_or_steps(
            self.anjana_defense,
            field_name="anjana_defense",
        )
        self.fairness_defense = self._normalize_optional_mapping_or_steps(
            self.fairness_defense,
            field_name="fairness_defense",
        )
        self.identifiers = self._normalize_optional_list(self.identifiers)
        self.quasi_identifiers = self._normalize_optional_list(self.quasi_identifiers)
        self.sensitive_columns = self._normalize_optional_list(self.sensitive_columns)
        if isinstance(self.hierarchy_interval_sizes, DictConfig):
            self.hierarchy_interval_sizes = dict(self.hierarchy_interval_sizes)

    @staticmethod
    def _format_interval_label(lower, upper) -> str:
        if float(lower).is_integer() and float(upper).is_integer():
            return f"[{int(lower)}, {int(upper)})"
        return f"[{lower}, {upper})"

    def _build_interval_hierarchy_level(
        self,
        series: pd.Series,
        interval_size: Union[int, float],
    ):
        values = pd.to_numeric(series, errors="coerce")
        if values.isna().all():
            return np.array(
                [self.hierarchy_fill_value] * len(series),
                dtype=object,
            )
        size = float(interval_size)
        min_val = float(values.min())
        start = np.floor(min_val / size) * size
        labels = []
        for value in values:
            if pd.isna(value):
                labels.append(self.hierarchy_fill_value)
                continue
            lower = np.floor((float(value) - start) / size) * size + start
            upper = lower + size
            labels.append(self._format_interval_label(lower, upper))
        return np.asarray(labels, dtype=object)

    def generate_anjana_hierarchy_dict(
        self,
        frame: Optional[pd.DataFrame] = None,
        quasi_identifiers: Optional[Union[str, list]] = None,
        interval_sizes: Optional[Dict[str, Union[int, list]]] = None,
        fill_value: Optional[str] = None,
    ) -> Dict[str, Dict[int, RuntimeValue]]:
        """Generate interval or categorical hierarchies for ANJANA quasi-identifiers.

        Args:
            frame: Optional source DataFrame. Defaults to runtime ``_X``.
            quasi_identifiers: Optional list of quasi-identifier column names.
            interval_sizes: Optional per-column numeric interval specification.
            fill_value: Optional replacement label for missing values.

        Returns:
            Per-column hierarchy dictionaries keyed by hierarchy level.
        """
        source = frame if frame is not None else getattr(self, "_X", None)
        if not isinstance(source, pd.DataFrame):
            raise TypeError(
                "Hierarchy generation requires a pandas.DataFrame source",
            )

        qids = (
            quasi_identifiers
            if quasi_identifiers is not None
            else self.quasi_identifiers
        )
        if isinstance(qids, str):
            qids = [qids]
        if not isinstance(qids, list) or len(qids) == 0:
            raise ValueError(
                "quasi_identifiers must be provided to generate ANJANA hierarchies",
            )

        interval_cfg = (
            interval_sizes
            if interval_sizes is not None
            else (self.hierarchy_interval_sizes or {})
        )
        replacement = self.hierarchy_fill_value if fill_value is None else fill_value
        hierarchies: Dict[str, Dict[int, Any]] = {}
        for col in qids:
            if col not in source.columns:
                raise KeyError(
                    f"Quasi-identifier '{col}' not found in frame columns",
                )
            series = source[col]
            levels: Dict[int, Any] = {0: series.to_numpy(copy=True)}
            configured = (
                interval_cfg.get(col) if isinstance(interval_cfg, dict) else None
            )
            if configured is not None:
                configured_levels = (
                    configured if isinstance(configured, list) else [configured]
                )
                for level_index, size in enumerate(configured_levels, start=1):
                    levels[level_index] = self._build_interval_hierarchy_level(
                        series,
                        size,
                    )
            else:
                levels[1] = np.asarray(
                    [replacement] * len(series),
                    dtype=object,
                )
            hierarchies[col] = levels
        return hierarchies

    def _resolve_anjana_target_column(self) -> str:
        if isinstance(self.target, str) and self.target.strip() != "":
            return self.target
        return "__deckard_target__"

    def _apply_anjana_defense(self) -> None:
        if self.anjana_defense in [None, False]:
            return
        if self.anjana_defense is True:
            raise ValueError(
                "anjana_defense=True is ambiguous. Provide a config dict with at least a 'name' key.",
            )
        if not isinstance(self.anjana_defense, (dict, DictConfig)):
            raise TypeError(
                "anjana_defense must be a dict/DictConfig, False, or None. "
                f"Got {type(self.anjana_defense)}",
            )

        defense_cfg = dict(self.anjana_defense)
        defense_name = defense_cfg.pop(
            "name",
            defense_cfg.pop("_target_", None),
        )
        if not isinstance(defense_name, str):
            raise ValueError(
                "anjana_defense config must include a 'name' or '_target_' key",
            )

        defense_fn = resolve_class(defense_name)
        if not callable(defense_fn):
            raise TypeError(
                f"Configured ANJANA defense '{defense_name}' is not callable",
            )

        frame = self._build_privacy_frame()
        call_kwargs = dict(defense_cfg)
        call_kwargs.setdefault("data", frame)
        call_kwargs.setdefault("ident", self.identifiers or [])
        if self.quasi_identifiers is not None:
            call_kwargs.setdefault("quasi_ident", self.quasi_identifiers)
        if self.sensitive_attribute is not None:
            call_kwargs.setdefault("sens_att", self.sensitive_attribute)
        call_kwargs.setdefault("supp_level", 100)
        if self.hierarchies is not None:
            call_kwargs.setdefault("hierarchies", self.hierarchies)
        elif self.quasi_identifiers is not None:
            call_kwargs.setdefault(
                "hierarchies",
                self.generate_anjana_hierarchy_dict(frame=frame),
            )

        signature = inspect.signature(defense_fn)
        supports_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        if not supports_var_kwargs:
            accepted = {
                name
                for name, p in signature.parameters.items()
                if p.kind
                in {
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                }
            }
            call_kwargs = {
                key: value for key, value in call_kwargs.items() if key in accepted
            }

        transformed = defense_fn(**call_kwargs)
        if not isinstance(transformed, pd.DataFrame):
            raise TypeError(
                f"ANJANA defense '{defense_name}' must return pandas.DataFrame, got {type(transformed)}",
            )

        target_col = self._resolve_anjana_target_column()
        if target_col not in transformed.columns:
            retained_index = transformed.index.intersection(frame.index)
            transformed = transformed.loc[retained_index].copy()
            self._y = pd.Series(self._y, index=frame.index).loc[retained_index]
        else:
            self._y = transformed[target_col]
            transformed = transformed.drop(columns=[target_col])

        self._X = transformed

    def _inject_privacy_defense_step(self) -> None:
        if self.fairness_defense in [None, False]:
            return
        if self.fairness_defense is True:
            raise ValueError(
                "fairness_defense=True is ambiguous. Provide a config dict with at least a 'name' key.",
            )
        if not isinstance(self.fairness_defense, (dict, DictConfig)):
            raise TypeError(
                "fairness_defense must be a dict/DictConfig, False, or None. "
                f"Got {type(self.fairness_defense)}",
            )
        if self.sensitive_columns is None:
            raise ValueError("sensitive_columns must be configured")
        if (
            not hasattr(self, "_X")
            or self._X is None
            or not isinstance(self._X, pd.DataFrame)
        ):
            return

        sensitive_columns = [
            col for col in self.sensitive_columns if col in self._X.columns
        ]
        if not sensitive_columns:
            raise RuntimeError(
                f"Sensitive features not found for {self.sensitive_columns}.",
            )

        step_config: Dict[str, Any] = {
            "sensitive_feature_ids": list(sensitive_columns),
        }
        step_name = "fairness_correlation_remover"
        custom = dict(self.fairness_defense)
        step_name = custom.pop("step_name", step_name)
        step_config.update(custom)
        if "name" not in step_config:
            raise ValueError(
                "fairness_defense config must include a 'name' key",
            )

        if step_name in self.pipeline:
            return

        self.pipeline = {step_name: step_config, **self.pipeline}

    def _inject_fairness_defense_step(self) -> None:
        """Backward-compatible bridge for fairness-defense pipeline injection."""
        self._inject_privacy_defense_step()

    def _build_privacy_frame(self) -> pd.DataFrame:
        frame = getattr(self, "_X", None)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                "ANJANA data defense requires tabular pandas DataFrame inputs",
            )
        frame = frame.copy()
        target_col = self._resolve_anjana_target_column()
        frame[target_col] = pd.Series(self._y).values
        return frame


@dataclass(eq=False, kw_only=True)
class AnjanaDataConfig(
    _PrivacyBehaviorMixin,
    _SensitiveColumnsMixin,
    DataPipelineConfig,
):
    """Data pipeline config with ANJANA anonymization support.

    This config extends ``DataPipelineConfig`` with optional privacy
    anonymization and fairness-preprocessing hooks. The default plugin setup
    executes ``_apply_anjana_defense`` after data load when an ANJANA defense
    configuration is provided.
    
    Privacy metrics are measured on POST-PIPELINE (anonymized) data by default,
    with results nested under the 'post-pipeline' key.
    """

    plugins: list = field(
        default_factory=lambda: [
            HookPlugin(
                hook_name="after_load_data",
                method_name="_apply_anjana_defense",
                init_params={
                    "library": "anjana",
                    "type": "data",
                    "class": "anonymization",
                },
            ),
        ],
    )
    
    score_mode: str = "post-pipeline"

    def __post_init__(self):
        # Support test patterns that call __post_init__ directly on bare instances.
        self._before_post_init()
        if is_default_config_value(self.scorer, include_best=False):
            ScorerClass = load_class(
                "deckard.plugins.anjana.score.DefaultAnjanaScorerConfig",
            )
            self.scorer = ScorerClass
        super().__post_init__()
        # Ensure scorer is instantiated (not just a class) after parent coercion
        if isinstance(self.scorer, type):
            self.scorer = self.scorer()
        self._validate_init()

    def __call__(
        self,
        *args: RuntimePayload,
        **kwargs: RuntimePayload,
    ) -> dict[str, RuntimeValue]:
        """Execute ANJANA data runtime with scorer auto-resolution.

        Args:
            *args: Positional runtime arguments forwarded to the pipeline runtime.
            **kwargs: Keyword runtime arguments forwarded to the pipeline runtime.

        Returns:
            Runtime score dictionary from the underlying pipeline config.
        """
        if (
            is_default_config_value(self.scorer, include_best=False)
            or self.scorer is None
        ):
            self.scorer = load_class(
                "deckard.plugins.anjana.score.DefaultAnjanaScorerConfig",
            )
        return DataPipelineConfig.__call__(self, *args, **kwargs)

    def _load_data(self):
        result = super()._load_data()
        if not getattr(self, "plugins", None):
            self._apply_anjana_defense()
        return result

    def _init_pipeline(self):
        self._inject_fairness_defense_step()
        return super()._init_pipeline()

    def _sample(self, run_hooks: bool = True):
        _ = run_hooks
        super()._sample()

        if self.fairness_defense not in [None, False]:
            for attr_name in (
                "_sensitive_train",
                "_sensitive_test",
                "_sensitive_all",
                "_sensitive_val",
            ):
                if hasattr(self, attr_name):
                    delattr(self, attr_name)
            return

        train_indices = getattr(self, "train_indices", None)
        test_indices = getattr(self, "test_indices", None)
        if train_indices is None or test_indices is None:
            self._sensitive_train = None
            self._sensitive_test = None
            self._sensitive_all = None
            self._sensitive_val = getattr(self, "_sensitive_val", None)
            return

        self._sensitive_train = self._sensitive_labels_from_frame(
            self._X.iloc[train_indices].reset_index(drop=True),
        )
        self._sensitive_test = self._sensitive_labels_from_frame(
            self._X.iloc[test_indices].reset_index(drop=True),
        )
        self._sensitive_all = self._sensitive_labels_from_frame(self._X)
        self._sensitive_train = self._validate_sensitive_runtime(
            self._sensitive_train,
            "train sampling",
        )
        self._sensitive_test = self._validate_sensitive_runtime(
            self._sensitive_test,
            "test sampling",
        )
        self._sensitive_all = self._validate_sensitive_runtime(
            self._sensitive_all,
            "full-data sampling",
        )
        val_indices = getattr(self, "val_indices", None)
        if val_indices is not None and len(val_indices) > 0:
            self._sensitive_val = self._sensitive_labels_from_frame(
                self._X.iloc[val_indices].reset_index(drop=True),
            )
            self._sensitive_val = self._validate_sensitive_runtime(
                self._sensitive_val,
                "val sampling",
            )
        else:
            self._sensitive_val = None

    def _score(
        self,
        *args,
        mode: Optional[Literal["train", "test", "val", "pre-sample", "post-pipeline"]] = None,
        **kwargs,
    ) -> dict:
        """
        Score the data using ANJANA privacy metrics measured on post-pipeline (anonymized) data.
        
        By default, measures privacy metrics on the full post-pipeline dataset
        (combined train/test splits after anonymization transformations).
        Results are nested under the 'post-pipeline' key.
        """
        # Handle "auto" scorer before checking if callable
        if is_default_config_value(self.scorer, include_best=False):
            ScorerClass = load_class(
                "deckard.plugins.anjana.score.DefaultAnjanaScorerConfig",
            )
            self.scorer = ScorerClass() if isinstance(ScorerClass, type) else ScorerClass
        
        if self.scorer is None:
            return {}
        if not callable(self.scorer):
            raise TypeError(
                f"AnjanaDataConfig.scorer must be callable or None, got {type(self.scorer)}",
            )
        
        # Resolve mode: use parameter or configured score_mode
        resolved_mode = mode
        if resolved_mode is None:
            resolved_mode = getattr(self, "score_mode", None) or "post-pipeline"
        
        # Delegate to parent for all modes, including post-pipeline/post-sample.
        try:
            return super()._score(*args, mode=resolved_mode, **kwargs)
        except TypeError as exc:
            if "data-profile scorer" not in str(exc):
                raise
            if resolved_mode == "pre-sample":
                y_true = getattr(self, "_y", None)
                y_pred = getattr(self, "_X", None)
            elif resolved_mode == "train":
                y_true = getattr(self, "y_train", None)
                y_pred = getattr(self, "X_train", None)
            elif resolved_mode == "val":
                y_true = getattr(self, "y_val", None)
                y_pred = getattr(self, "X_val", None)
            else:
                y_true = getattr(self, "y_test", None)
                y_pred = getattr(self, "X_test", None)
            return self.scorer(
                *args,
                y_true=y_true,
                y_pred=y_pred,
                mode=resolved_mode,
                data=self,
                **kwargs,
            )


# Configs are now loaded from YAML files in examples/*/config/data/
# These dictionaries are kept for reference/legacy code but not registered via safe_store
