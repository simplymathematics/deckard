from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import pandas as pd
from omegaconf import DictConfig, ListConfig

from .base import DataHookPlugin, DataPipelineConfig
from ..utils import (
    coerce_to_list,
    is_default_config_value,
    merge_list_of_dicts,
)
from ..score.fairness import (
    DefaultFairlearnClassificationConfig,
    DefaultFairlearnRegressionConfig,
)

import logging

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class _SensitiveBehaviorMixin:
    """Shared sensitive-feature behavior for data configs."""

    sensitive_columns: Optional[Union[str, list]] = None
    fairness_defense: Union[None, bool, Dict[str, Any], list] = None

    def _sensitive_labels_from_targets(
        self,
        frame: pd.DataFrame,
    ) -> Optional[pd.Series]:
        """Fallback sensitive labels from y-splits when sensitive feature is a target label."""
        if getattr(self, "X_train", None) is frame:
            y_values = getattr(self, "y_train", None)
        elif getattr(self, "X_test", None) is frame:
            y_values = getattr(self, "y_test", None)
        elif getattr(self, "X_val", None) is frame:
            y_values = getattr(self, "y_val", None)
        elif getattr(self, "_X", None) is frame:
            y_values = getattr(self, "_y", None)
        else:
            y_values = None
        if y_values is None:
            return None
        return pd.Series(y_values).astype(str)

    def _sensitive_labels_from_frame(
        self,
        frame: Optional[Union[pd.DataFrame, pd.Series]],
    ) -> pd.Series:
        if frame is None:
            raise ValueError(
                "_sensitive_labels_from_frame: frame must not be None",
            )
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        cols = self.sensitive_columns
        if isinstance(cols, str):
            cols = [cols]
        if cols is None:
            fallback = self._sensitive_labels_from_targets(frame)
            if fallback is not None:
                return fallback
            raise ValueError("sensitive_columns must be configured")
        missing_cols = [col for col in cols if col not in frame.columns]
        if missing_cols:
            if len(cols) == 1:
                fallback = self._sensitive_labels_from_targets(frame)
                if fallback is not None:
                    return fallback
            raise KeyError(
                f"Sensitive feature columns not found: {missing_cols}. Available columns: {list(frame.columns)}",
            )
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

    def _sample(self, run_hooks: bool = True):
        super()._sample(run_hooks=run_hooks)

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
        if getattr(self, "X_val", None) is not None:
            self._sensitive_val = self._sensitive_labels_from_frame(self.X_val)
            self._sensitive_val = self._validate_sensitive_runtime(
                self._sensitive_val,
                "val sampling",
            )
        else:
            self._sensitive_val = None


@dataclass(eq=False)
class FairlearnDataConfig(_SensitiveBehaviorMixin, DataPipelineConfig):
    """Data pipeline config with fairlearn-sensitive feature support.

    Initialization params
    ---------------------
    sensitive_columns : str | list[str] | None
        Sensitive-feature column name(s) used for fairness metrics and
        mitigation transforms. This value is required.
    fairness_defense : dict[str, Any] | list[dict[str, Any]] | bool | None
        Fairness-defense step specification consumed by
        ``_inject_fairness_defense_step``.
    plugins : list[DataHookPlugin]
        Declarative runtime plugin specs. Default contains one
        ``DataHookPlugin`` configured with:
        ``hook_name: str = 'before_sample'``,
        ``method_name: str = '_inject_fairness_defense_step'``, and
        ``init_params: dict[str, Any]`` metadata.

    Runtime params
    --------------
    __call__(self, *args: Any, **kwargs: Any) -> Any
        Resolves default fairness scorer when needed and delegates to
        ``DataPipelineConfig.__call__``.
    _score(self, mode: str | None = None) -> dict
        Computes fairness scores with ``y_true`` and ``y_pred`` sourced from
        runtime train buffers when available.
    """

    def __call__(self, *args, **kwargs):
        # Auto-select fairness-compatible scorer if not set
        if (
            is_default_config_value(self.scorer, include_best=False)
            or self.scorer is None
        ):
            from deckard.score import (
                DefaultFairlearnClassificationConfig,
                DefaultFairlearnRegressionConfig,
            )

            self.scorer = (
                DefaultFairlearnClassificationConfig()
                if self.classifier
                else DefaultFairlearnRegressionConfig()
            )
        # Call parent to load and sample data
        result = super().__call__(*args, **kwargs)
        # Output validation removed: allow dict outputs for label_distribution and sensitive_distribution
        assert hasattr(self, "X_train"), ".X_train not found"
        return result

    plugins: list = field(
        default_factory=lambda: [
            DataHookPlugin(
                hook_name="before_sample",
                method_name="_inject_fairness_defense_step",
                init_params={
                    "library": "fairlearn",
                    "type": "data",
                    "class": "CorrelationRemover",
                },
            )
        ]
    )

    def __post_init__(self):
        super().__post_init__()
        self._validate_init()

        if isinstance(self.fairness_defense, (list, ListConfig)):
            self.fairness_defense = merge_list_of_dicts(
                coerce_to_list(self.fairness_defense),
            )

        if self.sensitive_columns is None:
            raise ValueError(
                "sensitive_columns must be specified for FairlearnDataConfig",
            )
        if isinstance(self.sensitive_columns, ListConfig):
            self.sensitive_columns = list(self.sensitive_columns)
        elif isinstance(self.sensitive_columns, str):
            self.sensitive_columns = [self.sensitive_columns]


    def _load_data(self) -> Any:
        super()._load_data()
        assert hasattr(self, "_X"), RuntimeError(
            "self._X not found while loading FairlearnDataConfig",
        )
        assert hasattr(self, "_y"), RuntimeError(
            "self._y not found while loading FairlearnDataConfig",
        )
        assert isinstance(self._X, pd.DataFrame), ValueError(
            "Expected a dataframe for self._X",
        )
        if self.sensitive_columns is None:
            raise ValueError("sensitive_columns must be configured")
        for col in self.sensitive_columns:
            assert col in self._X.columns
        return self

    def _score(self, *args, mode=None, **kwargs) -> dict:
        """Delegate fairness dataset scoring to DefaultFairlearnClassificationConfig or RegressionConfig, and flatten output."""
        if is_default_config_value(self.scorer, include_best=False):
            self.scorer = (
                DefaultFairlearnClassificationConfig()
                if self.classifier
                else DefaultFairlearnRegressionConfig()
            )
        if self.scorer is None:
            return {}
        if not callable(self.scorer):
            raise TypeError(
                f"FairlearnDataConfig.scorer must be callable or None, got {type(self.scorer)}",
            )
        # Use train mode by default to preserve DataConfig key prefixes
        # such as training_class_count/training_mutual_info.
        scorer_mode = mode if mode is not None else "train"
        y_true = (
            self.y_train if getattr(self, "y_train", None) is not None else self._y
        )
        y_pred = (
            self.X_train if getattr(self, "X_train", None) is not None else self._X
        )
        fairness_scores = self.scorer(
            *args,
            y_true=y_true,
            y_pred=y_pred,
            mode=scorer_mode,
            data=self,
            **kwargs,
        )
        # Flatten fairness_scores if it's a dict
        if isinstance(fairness_scores, dict):
            flat = {}
            for k, v in fairness_scores.items():
                if isinstance(v, dict):
                    for subk, subv in v.items():
                        flat[f"{k}_{subk}"] = subv
                else:
                    flat[k] = v
            return flat
        else:
            return {"fairness_score": fairness_scores}
