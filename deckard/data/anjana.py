import inspect
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Union, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from .base import DataPipelineConfig
from .fairness import _SensitiveBehaviorMixin
from ..utils import (
    is_default_config_value,
    load_class,
    safe_store,
    resolve_class,
    normalize_plugin_specs,
    normalize_optional_list_value as _normalize_optional_list_value,
    normalize_optional_mapping_or_steps as _normalize_optional_mapping_or_steps,
)


@dataclass
class AnjanaDefensePlugin:
    """Hookable ANJANA anonymization stub used by data configs."""

    anjana_defense: Union[None, bool, Dict[str, Any], list] = None
    identifiers: Optional[Union[str, list]] = None
    quasi_identifiers: Optional[Union[str, list]] = None
    sensitive_attribute: Optional[str] = None
    hierarchies: Optional[Dict[str, Dict[int, Any]]] = None
    hierarchy_interval_sizes: Optional[Dict[str, Union[int, list]]] = None
    hierarchy_fill_value: str = "*"

    def after_load_data(self, data, **kwargs):
        _ = kwargs
        data._apply_anjana_defense()


class _PrivacyBehaviorMixin:
    """Reusable privacy behavior mixed into data pipeline configs."""

    # Declared for static analyzers; concrete dataclass provides these fields.
    anjana_defense: Union[None, bool, Dict[str, Any], list]
    identifiers: Optional[Union[str, list]]
    quasi_identifiers: Optional[Union[str, list]]
    sensitive_attribute: Optional[str]
    sensitive_columns: Optional[Union[str, list]]
    hierarchies: Optional[Dict[str, Dict[int, Any]]]
    hierarchy_interval_sizes: Optional[Dict[str, Union[int, list]]]
    hierarchy_fill_value: str
    target: Optional[str]
    plugins: list

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
        self.identifiers = self._normalize_optional_list(self.identifiers)
        self.quasi_identifiers = self._normalize_optional_list(self.quasi_identifiers)
        self.sensitive_columns = self._normalize_optional_list(self.sensitive_columns)
        if isinstance(self.hierarchy_interval_sizes, DictConfig):
            self.hierarchy_interval_sizes = dict(self.hierarchy_interval_sizes)
        self.plugins = normalize_plugin_specs(self.plugins)
        if any(
            value is not None
            for value in (
                self.anjana_defense,
                self.sensitive_columns,
            )
        ):
            self.plugins = cast(
                list,
                [
                    AnjanaDefensePlugin(
                        anjana_defense=self.anjana_defense,
                        identifiers=self.identifiers,
                        quasi_identifiers=self.quasi_identifiers,
                        sensitive_attribute=self.sensitive_attribute,
                        hierarchies=self.hierarchies,
                        hierarchy_interval_sizes=self.hierarchy_interval_sizes,
                        hierarchy_fill_value=self.hierarchy_fill_value,
                    ),
                    *self.plugins,
                ],
            )

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
    ) -> Dict[str, Dict[int, Any]]:
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

        frame = self._build_anjana_frame()
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

    def _build_anjana_frame(self) -> pd.DataFrame:
        frame = getattr(self, "_X", None)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                "ANJANA data defense requires tabular pandas DataFrame inputs",
            )
        frame = frame.copy()
        target_col = self._resolve_anjana_target_column()
        frame[target_col] = pd.Series(self._y).values
        return frame


@dataclass(eq=False)
class AnjanaDataConfig(_PrivacyBehaviorMixin, _SensitiveBehaviorMixin, DataPipelineConfig):
    """Data pipeline config with ANJANA anonymization support."""

    identifiers: Optional[Union[str, list]] = None
    quasi_identifiers: Optional[Union[str, list]] = None
    sensitive_attribute: Optional[str] = None
    anjana_defense: Union[None, bool, Dict[str, Any], list] = None
    sensitive_columns: Optional[Union[str, list]] = None
    fairness_defense: Union[None, bool, Dict[str, Any], list] = None
    hierarchies: Optional[Dict[str, Dict[int, Any]]] = None
    hierarchy_interval_sizes: Optional[Dict[str, Union[int, list]]] = None
    hierarchy_fill_value: str = "*"

    def __post_init__(self):
        # Support test patterns that call __post_init__ directly on bare instances.
        self._before_post_init()
        if is_default_config_value(self.scorer, include_best=False):
            self.scorer = load_class(
                "deckard.score.anjana.DefaultAnjanaDataScoreConfig",
            )
        super().__post_init__()
        self._validate_init()

    def __call__(self, *args, **kwargs):
        if (
            is_default_config_value(self.scorer, include_best=False)
            or self.scorer is None
        ):
            self.scorer = load_class(
                "deckard.score.anjana.DefaultAnjanaDataScoreConfig",
            )
        return DataPipelineConfig.__call__(self, *args, **kwargs)

    def _load_data(self):
        return super()._load_data()

    def _init_pipeline(self):
        return super()._init_pipeline()

    def _score(
        self,
        mode: Optional[Literal["train", "test", "val", "pre-sample"]] = None,
        **kwargs,
    ) -> dict:
        if is_default_config_value(self.scorer, include_best=False):
            self.scorer = load_class(
                "deckard.score.anjana.DefaultAnjanaDataScoreConfig",
            )
        if mode is None:
            return super()._score(**kwargs)
        return super()._score(mode=mode, **kwargs)


ANJANA_DATA = {
    "dataset_name": "make_classification",
    "data_params": {
        "n_samples": 1000,
        "n_features": 20,
        "n_informative": 15,
        "n_redundant": 5,
        "n_classes": 2,
        "random_state": 42,
    },
    "test_size": 0.2,
    "random_state": 42,
    "classifier": True,
    "alias": "anjana",
    "sample": "split",
    "quasi_identifiers": ["feature_0", "feature_1"],
    "sensitive_attribute": "target",
    "sensitive_columns": ["feature_0"],
    "hierarchy_interval_sizes": {"feature_0": [1, 2], "feature_1": [1, 2]},
    "_target_": "deckard.data.anjana.AnjanaDataConfig",
}

ANJANA_DIABETES = {
    "dataset_name": "load_diabetes",
    "data_params": {
        "as_frame": True,
    },
    "test_size": 0.2,
    "random_state": 42,
    "classifier": False,
    "alias": "anjana-diabetes",
    "sample": "split",
    "quasi_identifiers": ["age", "sex"],
    "sensitive_attribute": "sex",
    "sensitive_columns": ["sex"],
    "hierarchy_interval_sizes": {"age": [5, 10]},
    "_target_": "deckard.data.anjana.AnjanaDataConfig",
}

ANJANA_ADULT = {
    "dataset_name": "fetch_openml",
    "data_params": {
        "name": "adult",
        "version": 2,
        "as_frame": True,
        "parser": "auto",
    },
    "test_size": 0.2,
    "random_state": 42,
    "classifier": True,
    "alias": "anjana-adult",
    "sample": "split",
    "quasi_identifiers": ["age", "education-num"],
    "sensitive_attribute": "race",
    "sensitive_columns": ["race", "sex"],
    "hierarchy_interval_sizes": {"age": [5, 10], "education-num": [2, 4]},
    "_target_": "deckard.data.anjana.AnjanaDataConfig",
}

safe_store(group="data", name="anjana", node=ANJANA_DATA)
safe_store(group="search/data", name="anjana", node=ANJANA_DATA)
safe_store(group="data", name="anjana-diabetes", node=ANJANA_DIABETES)
safe_store(group="search/data", name="anjana-diabetes", node=ANJANA_DIABETES)
safe_store(group="data", name="anjana-adult", node=ANJANA_ADULT)
safe_store(group="search/data", name="anjana-adult", node=ANJANA_ADULT)
