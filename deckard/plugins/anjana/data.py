from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from deckard.plugins import HookPlugin
from deckard.plugins.base import compose_hook_plugins
from deckard.model.defense.base import DefenseStep

from ...data._mixins import RuntimePayload, SensitiveColumnsMixin
from ...data.base import DataConfig
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
from .pipeline import ANJANA_PIPELINE_HOOKS, AnjanaPipelineHooksMixin
from .score import ANJANA_SCORING_HOOKS, AnjanaDataScoreHooksMixin

RuntimeScalar = str | int | float | bool | None
SerializableValue = (
    RuntimeScalar | list["SerializableValue"] | dict[str, "SerializableValue"]
)


def default_anjana_data_plugins() -> list[HookPlugin]:
    """Compose ANJANA runtime hooks from separate pipeline/scoring bundles."""
    return compose_hook_plugins(
        ANJANA_PIPELINE_HOOKS,
        ANJANA_SCORING_HOOKS,
    )


@dataclass(eq=False, kw_only=True)
class PrivacyBehaviorMixin(SensitiveColumnsMixin):
    """Reusable privacy behavior mixed into data pipeline configs.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

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
    ) -> Dict[str, Dict[int, SerializableValue]]:
        """Generate interval or categorical hierarchies for ANJANA quasi-identifiers.

        Args:
            frame: Optional source DataFrame. Defaults to runtime ``_X``.
            quasi_identifiers: Optional list of quasi-identifier column names.
            interval_sizes: Optional per-column numeric interval specification.
            fill_value: Optional replacement label for missing values.

        Returns:
            Per-column hierarchy dictionaries keyed by hierarchy level.

        Raises:
            TypeError: If the source data is not a pandas DataFrame.
            ValueError: If quasi-identifiers are not provided.
            KeyError: If any quasi-identifier column is missing from the source frame.
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
        target = getattr(self, "target", None)
        if isinstance(target, str) and target.strip() != "":
            return target
        return "__deckard_target__"

    @staticmethod
    def _normalize_named_defense_mapping(
        defense_mapping: Any,
        *,
        field_name: str,
    ) -> Dict[str, Any]:
        """Normalize defense mapping keys to a canonical runtime shape.

        Requires canonical ``name`` and optional ``defense_params``.
        """
        if isinstance(defense_mapping, DefenseStep):
            defense_name = defense_mapping.name
            if not isinstance(defense_name, str):
                raise ValueError(
                    f"{field_name} config must include 'name'",
                )
            normalized = {
                "name": defense_name,
                "defense_params": dict(defense_mapping.defense_params or {}),
            }
        elif isinstance(defense_mapping, DictConfig):
            normalized = dict(defense_mapping)
        elif isinstance(defense_mapping, dict):
            normalized = dict(defense_mapping)
        else:
            raise TypeError(
                f"{field_name} must be a dict/DictConfig/DefenseStep. Got {type(defense_mapping)}",
            )

        defense_params = normalized.pop("defense_params", {})
        if isinstance(defense_params, DictConfig):
            defense_params = dict(defense_params)
        elif not isinstance(defense_params, dict):
            raise TypeError(
                f"{field_name}.defense_params must be a dict/DictConfig when provided. Got {type(defense_params)}",
            )

        defense_name = normalized.pop("name", None)
        if not isinstance(defense_name, str):
            raise ValueError(
                f"{field_name} config must include 'name'",
            )
        for key in ("apply_fit", "apply_predict", "alias", "_target_", "plugins"):
            normalized.pop(key, None)
        result: Dict[str, Any] = {"name": defense_name}
        for key, value in defense_params.items():
            result[str(key)] = value
        for key, value in normalized.items():
            result[str(key)] = value
        return result

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
        frame = getattr(self, "_X", None)
        if frame is None or not isinstance(frame, pd.DataFrame):
            return

        sensitive_columns = [
            col for col in self.sensitive_columns if col in frame.columns
        ]
        if not sensitive_columns:
            raise RuntimeError(
                f"Sensitive features not found for {self.sensitive_columns}.",
            )

        step_config: Dict[str, Any] = {
            "sensitive_feature_ids": list(sensitive_columns),
        }
        step_name = "fairness_correlation_remover"
        custom = self._normalize_named_defense_mapping(
            dict(self.fairness_defense),
            field_name="fairness_defense",
        )
        step_name = custom.pop("step_name", step_name)
        step_config.update(custom)
        if "name" not in step_config:
            raise ValueError(
                "fairness_defense config must include 'name'",
            )

        if step_name in self.pipeline:
            return

        self.pipeline = {step_name: step_config, **self.pipeline}

    def _build_privacy_frame(self) -> pd.DataFrame:
        frame = getattr(self, "_X", None)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                "ANJANA data defense requires tabular pandas DataFrame inputs",
            )
        frame = frame.copy()
        target_col = self._resolve_anjana_target_column()
        frame[target_col] = pd.Series(getattr(self, "_y", None)).values
        return frame


@dataclass(eq=False, kw_only=True)
class AnjanaDataConfig(
    PrivacyBehaviorMixin,
    AnjanaPipelineHooksMixin,
    AnjanaDataScoreHooksMixin,
    DataConfig,
):
    """Data pipeline config with ANJANA anonymization support.

    This config extends ``DataConfig`` with optional privacy
    anonymization and fairness-preprocessing hooks. The default plugin setup
    executes ``apply_defense`` after data load when an ANJANA defense
    configuration is provided.

    Privacy metrics default to test-split score scope while retaining
    post-pipeline stage hook execution.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    plugins: list = field(
        default_factory=default_anjana_data_plugins,
        metadata={"help": "Configuration field: plugins."},
        repr=True,
    )

    score_mode: str = "test"

    def fit(self, run_hooks: bool = True) -> "AnjanaDataConfig":
        """Fit data splits and refresh split-aligned sensitive feature payloads.

        Args:
            run_hooks: Whether to execute configured runtime hooks during fit.

        Returns:
            The current configuration instance.
        """
        super().fit(run_hooks=run_hooks)

        if self.fairness_defense not in [None, False]:
            for attr_name in (
                "_sensitive_train",
                "_sensitive_test",
                "_sensitive_all",
                "_sensitive_val",
            ):
                if hasattr(self, attr_name):
                    delattr(self, attr_name)
            return self

        train_indices = getattr(self, "train_indices", None)
        test_indices = getattr(self, "test_indices", None)
        if train_indices is None or test_indices is None:
            self._sensitive_train = None
            self._sensitive_test = None
            self._sensitive_all = None
            self._sensitive_val = getattr(self, "_sensitive_val", None)
            return self

        frame = getattr(self, "_X", None)
        if not isinstance(frame, pd.DataFrame):
            self._sensitive_train = None
            self._sensitive_test = None
            self._sensitive_all = None
            self._sensitive_val = None
            return self

        self._sensitive_train = self._sensitive_labels_from_frame(
            frame.iloc[train_indices].reset_index(drop=True),
        )
        self._sensitive_test = self._sensitive_labels_from_frame(
            frame.iloc[test_indices].reset_index(drop=True),
        )
        self._sensitive_all = self._sensitive_labels_from_frame(frame)
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
                frame.iloc[val_indices].reset_index(drop=True),
            )
            self._sensitive_val = self._validate_sensitive_runtime(
                self._sensitive_val,
                "val sampling",
            )
        else:
            self._sensitive_val = None
        return self

    def __post_init__(self):
        # Support test patterns that call __post_init__ directly on bare instances.
        self._before_post_init()
        if is_default_config_value(self.scorer, include_best=False):
            ScorerClass = load_class(
                "deckard.plugins.anjana.score.DefaultAnjanaScorerDictConfig",
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
    ) -> dict[str, SerializableValue]:
        """Execute ANJANA data runtime with canonical file handling.

        Args:
            *args: Positional runtime payloads forwarded to base data execution.
            **kwargs: Keyword runtime payloads and optional file mappings.

        Returns:
            Runtime score and artifact payload mapping.
        """
        files = cast(Any, kwargs.pop("files", None))
        return self.execute_data_runtime(*args, files=files, **kwargs)


__all__ = [
    "ANJANA_PIPELINE_HOOKS",
    "ANJANA_SCORING_HOOKS",
    "PrivacyBehaviorMixin",
    "default_anjana_data_plugins",
    "AnjanaDataConfig",
    "resolve_class",
]
