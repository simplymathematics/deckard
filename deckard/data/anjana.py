from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig

from .base import DataPipelineConfig
from ..utils import (
    coerce_to_list,
    load_class,
    merge_list_of_dicts,
    resolve_class,
)


@dataclass(eq=False)
class AnjanaDataConfig(DataPipelineConfig):
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
        super().__post_init__()
        self._validate_init()

        if isinstance(self.anjana_defense, (list, ListConfig)):
            self.anjana_defense = merge_list_of_dicts(
                coerce_to_list(self.anjana_defense),
            )
        if isinstance(self.fairness_defense, (list, ListConfig)):
            self.fairness_defense = merge_list_of_dicts(
                coerce_to_list(self.fairness_defense),
            )

        if isinstance(self.identifiers, ListConfig):
            self.identifiers = list(self.identifiers)
        elif isinstance(self.identifiers, str):
            self.identifiers = [self.identifiers]

        if isinstance(self.quasi_identifiers, ListConfig):
            self.quasi_identifiers = list(self.quasi_identifiers)
        elif isinstance(self.quasi_identifiers, str):
            self.quasi_identifiers = [self.quasi_identifiers]

        if isinstance(self.sensitive_columns, ListConfig):
            self.sensitive_columns = list(self.sensitive_columns)
        elif isinstance(self.sensitive_columns, str):
            self.sensitive_columns = [self.sensitive_columns]

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

    def _sensitive_labels_from_frame(self, frame: pd.DataFrame) -> pd.Series:
        cols = self.sensitive_columns
        if isinstance(cols, str):
            cols = [cols]
        if cols is None:
            raise ValueError("sensitive_columns must be configured")
        if len(cols) == 1:
            return frame[cols[0]].astype(str)
        labels_df = frame[cols].astype(str)
        return labels_df.apply(lambda row: tuple(row.values.tolist()), axis=1)

    def _validate_sensitive_runtime(
        self,
        sensitive: pd.Series,
        context: str,
    ) -> pd.Series:
        sensitive_series = pd.Series(sensitive)
        if len(sensitive_series) == 0:
            raise ValueError(f"Sensitive features are empty during {context}")
        if sensitive_series.dropna().empty:
            raise ValueError(
                f"Sensitive features are all null during {context}",
            )
        if sensitive_series.astype(str).str.strip().eq("").all():
            raise ValueError(f"Sensitive features are blank during {context}")
        return sensitive_series

    def _inject_fairness_defense_step(self) -> None:
        """Inject fairlearn preprocessing into the data pipeline when configured."""
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
            raise ValueError(
                "sensitive_columns must be configured when fairness_defense is set",
            )
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

    def _build_anjana_frame(self) -> pd.DataFrame:
        if not isinstance(getattr(self, "_X", None), pd.DataFrame):
            raise TypeError(
                "ANJANA data defense requires tabular pandas DataFrame inputs",
            )
        frame = self._X.copy()
        target_col = self._resolve_anjana_target_column()
        frame[target_col] = pd.Series(self._y).values
        return frame

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
        if self.identifiers is not None:
            call_kwargs.setdefault("ident", self.identifiers)
        if self.quasi_identifiers is not None:
            call_kwargs.setdefault("quasi_ident", self.quasi_identifiers)
        if self.sensitive_attribute is not None:
            call_kwargs.setdefault("sens_att", self.sensitive_attribute)
        if self.hierarchies is not None:
            call_kwargs.setdefault("hierarchies", self.hierarchies)
        elif self.quasi_identifiers is not None:
            call_kwargs.setdefault(
                "hierarchies",
                self.generate_anjana_hierarchy_dict(frame=frame),
            )

        transformed = defense_fn(**call_kwargs)
        if not isinstance(transformed, pd.DataFrame):
            raise TypeError(
                f"ANJANA defense '{defense_name}' must return pandas.DataFrame, got {type(transformed)}",
            )

        target_col = self._resolve_anjana_target_column()
        if target_col not in transformed.columns:
            # Fallback: retain labels by index overlap when target column is removed.
            retained_index = transformed.index.intersection(frame.index)
            transformed = transformed.loc[retained_index].copy()
            self._y = pd.Series(self._y, index=frame.index).loc[retained_index]
        else:
            self._y = transformed[target_col]
            transformed = transformed.drop(columns=[target_col])

        self._X = transformed

    def _load_data(self):
        super()._load_data()
        self._apply_anjana_defense()
        return self

    def _init_pipeline(self):
        self._inject_fairness_defense_step()
        return super()._init_pipeline()

    def _sample(self):
        super()._sample()
        if self.sensitive_columns is None:
            return
        self._sensitive_train = self._sensitive_labels_from_frame(self.X_train)
        self._sensitive_test = self._sensitive_labels_from_frame(self.X_test)
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

    def _score(self) -> dict:
        if isinstance(self.scorer, str) and self.scorer.lower() in {
            "auto",
            "default",
        }:
            self.scorer = load_class(
                "deckard.score.anjana.DefaultAnjanaDataScoreConfig",
            )
        if self.scorer is None:
            return {}
        if not callable(self.scorer):
            raise TypeError(
                f"AnjanaDataConfig.scorer must be callable or None, got {type(self.scorer)}",
            )
        y_true = (
            self.y_train if getattr(self, "y_train", None) is not None else self._y
        )
        y_pred = (
            self.X_train if getattr(self, "X_train", None) is not None else self._X
        )
        anjana_scores = self.scorer(
            y_true=y_true,
            y_pred=y_pred,
            mode=None,
            data=self,
        )
        return {"anjana_scores": anjana_scores}
