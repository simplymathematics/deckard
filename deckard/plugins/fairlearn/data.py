from dataclasses import dataclass, field
import logging
from typing import Any, Dict

import pandas as pd
from omegaconf import DictConfig, ListConfig

from deckard.plugins import HookPlugin

from ...data._mixins import RuntimePayload, _SensitiveColumnsMixin
from ...data.base import DataPipelineConfig
from ...utils import (
    coerce_to_list,
    is_default_config_value,
    merge_list_of_dicts,
)
from ...plugins.fairlearn.score import (
    DefaultFairlearnClassificationConfig,
    DefaultFairlearnRegressionConfig,
)

RuntimeScalar = str | int | float | bool | None
RuntimeValue = RuntimeScalar | list["RuntimeValue"] | dict[str, "RuntimeValue"]

logger = logging.getLogger(__name__)


@dataclass(eq=False, kw_only=True)
class _FairnessBehaviorMixin(_SensitiveColumnsMixin):
    """Fairlearn-specific sensitive-feature behavior for data configs.

    Extends the framework-independent :class:`_SensitiveColumnsMixin` with
    fairlearn defense injection and sampling hooks.  Import
    ``_SensitiveColumnsMixin`` directly when you need only the shared
    fields/helpers without pulling in fairlearn-specific logic.
    """

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

    def fit(self, run_hooks: bool = True):
        super().fit(run_hooks=run_hooks)

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
        return self


@dataclass(eq=False, kw_only=True)
class FairlearnDataConfig(_FairnessBehaviorMixin, DataPipelineConfig):
    """Fairlearn-aware data pipeline configuration.

    This extends ``DataPipelineConfig`` with sensitive feature handling,
    fairness-defense pipeline injection, and fairness-oriented scorer defaults.

    Key fields:
    - ``sensitive_columns``: required sensitive feature column name(s).
    - ``fairness_defense``: optional transform/mitigation config.
    - ``plugins``: runtime hook plugins for fairness pipeline behavior.
    """

    def __call__(
        self,
        *args: RuntimePayload,
        **kwargs: RuntimePayload,
    ) -> dict[str, RuntimeValue]:
        """Execute fairness-aware data runtime with scorer auto-selection.

        Args:
                *args: Positional runtime arguments forwarded to the parent pipeline runtime.
                **kwargs: Keyword runtime arguments forwarded to the parent pipeline runtime.

        Returns:
                Runtime score dictionary produced by the parent pipeline config.
        """
        # Auto-select fairness-compatible scorer if not set
        if (
            is_default_config_value(self.scorer, include_best=False)
            or self.scorer is None
        ):
            from deckard.plugins.fairlearn.score import (
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
            HookPlugin(
                hook_name="before_sample",
                method_name="_inject_fairness_defense_step",
                init_params={
                    "library": "fairlearn",
                    "type": "data",
                    "class": "CorrelationRemover",
                },
            ),
        ],
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

    def load_dataset(self) -> Any:
        super().load_dataset()
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

    def score(self, *args, mode=None, **kwargs) -> dict:
        """Delegate fairness dataset scoring and flatten nested dict output."""
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
        scorer_mode = mode if mode is not None else "train"
        if mode == "pre-sample":
            y_true = getattr(self, "_y", None)
            y_pred = getattr(self, "_X", None)
        else:
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


__all__ = ["FairlearnDataConfig"]
