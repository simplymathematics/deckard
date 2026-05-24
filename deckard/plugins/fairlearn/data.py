from dataclasses import dataclass, field
import logging
from typing import Any

import pandas as pd
from omegaconf import DictConfig, ListConfig

from deckard.plugins import HookPlugin
from deckard.plugins.base import compose_hook_plugins

from ...data._mixins import RuntimePayload, SensitiveColumnsMixin
from ...data.base import DataConfig
from ...data.canon import resolve_runtime_files
from ...utils import (
    coerce_to_list,
    is_default_config_value,
    merge_list_of_dicts,
)
from .pipeline import FAIRLEARN_PIPELINE_HOOKS, FairlearnPipelineHooksMixin
from .score import (
    FAIRLEARN_SCORING_HOOKS,
    DefaultFairlearnClassificationScorerDictConfig,
    DefaultFairlearnRegressionScorerDictConfig,
    FairlearnDataScoreHooksMixin,
)

RuntimeScalar = str | int | float | bool | None
SerializableValue = (
    RuntimeScalar
    | list["SerializableValue"]
    | dict[str, "SerializableValue"]
)

logger = logging.getLogger(__name__)


def default_fairlearn_data_plugins() -> list[HookPlugin]:
    """Compose fairlearn data runtime hooks from separate pipeline/scoring bundles."""
    return compose_hook_plugins(
        FAIRLEARN_PIPELINE_HOOKS,
        FAIRLEARN_SCORING_HOOKS,
    )


@dataclass(eq=False, kw_only=True)
class FairnessBehaviorMixin(SensitiveColumnsMixin):
    """Fairlearn-specific sensitive-feature behavior for data configs.

    Extends the framework-independent :class:`_SensitiveColumnsMixin` with
    fairlearn defense injection and sampling hooks.  Import
    ``_SensitiveColumnsMixin`` directly when you need only the shared
    fields/helpers without pulling in fairlearn-specific logic.
    """

    def fit(self, run_hooks: bool = True) -> "FairnessBehaviorMixin":
        """Populate split-aligned sensitive feature payloads after data sampling.

        Args:
            run_hooks: Whether the parent data runtime should execute hook callbacks.

        Returns:
            The current fairness behavior instance.
        """
        super().fit(run_hooks=run_hooks)

        train_indices = getattr(self, "train_indices", None)
        test_indices = getattr(self, "test_indices", None)
        if train_indices is None or test_indices is None:
            self._sensitive_train = None
            self._sensitive_test = None
            self._sensitive_all = None
            self._sensitive_val = getattr(self, "_sensitive_val", None)
            return self

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
class FairlearnDataConfig(
    FairnessBehaviorMixin,
    FairlearnPipelineHooksMixin,
    FairlearnDataScoreHooksMixin,
    DataConfig,
):
    """Fairlearn-aware data pipeline configuration.

    This extends ``DataConfig`` with sensitive feature handling,
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
    ) -> dict[str, SerializableValue]:
        """Execute fairness-aware data runtime.

        Args:
            *args: Positional runtime payloads forwarded to parent execution.
            **kwargs: Keyword runtime payloads and optional file mappings.

        Returns:
            Runtime score and artifact payload mapping.
        """
        files = resolve_runtime_files(
            kwargs,
            kwargs.pop("files", None),
        )
        self._coerce_pipeline_runtime()
        result = super().__call__(*args, files=files, **kwargs)
        assert hasattr(self, "X_train"), ".X_train not found"
        return result

    plugins: list = field(default_factory=default_fairlearn_data_plugins)

    def __post_init__(self):
        super().__post_init__()
        self._validate_init()

        if (
            is_default_config_value(self.scorer, include_best=False)
            or self.scorer is None
        ):
            self.scorer = (
                DefaultFairlearnClassificationScorerDictConfig()
                if self.classifier
                else DefaultFairlearnRegressionScorerDictConfig()
            )

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

    def load_dataset(self) -> "FairlearnDataConfig":
        """Load dataset and validate configured sensitive columns are present.

        Returns:
            The current configuration instance after dataset validation.

        Raises:
            ValueError: If sensitive columns are not configured.
            AssertionError: If runtime datasets are missing expected attributes/columns.
        """
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


__all__ = [
    "FAIRLEARN_PIPELINE_HOOKS",
    "FAIRLEARN_SCORING_HOOKS",
    "FairnessBehaviorMixin",
    "default_fairlearn_data_plugins",
    "FairlearnDataConfig",
]
