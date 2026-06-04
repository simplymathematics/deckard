from dataclasses import dataclass, field
import logging
from typing import Any, cast

import pandas as pd
from omegaconf import ListConfig

from deckard.plugins import HookPlugin
from deckard.plugins.base import compose_hook_plugins

from ...data._mixins import RuntimePayload, SensitiveColumnsMixin
from ...data.base import DataConfig
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
    RuntimeScalar | list["SerializableValue"] | dict[str, "SerializableValue"]
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

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    def _refresh_sensitive_splits(self) -> "DataConfig":
        data_runtime = cast(DataConfig, self)
        train_indices = getattr(data_runtime, "train_indices", None)
        test_indices = getattr(data_runtime, "test_indices", None)
        if train_indices is None or test_indices is None:
            data_runtime._sensitive_train = None
            data_runtime._sensitive_test = None
            data_runtime._sensitive_all = None
            data_runtime._sensitive_val = getattr(data_runtime, "_sensitive_val", None)
            return data_runtime

        def _extract_sensitive(runtime_frame, fallback_frame, context: str):
            try:
                sensitive_payload = data_runtime._sensitive_features_from_frame(
                    runtime_frame,
                )
            except Exception:
                sensitive_payload = data_runtime._sensitive_labels_from_frame(
                    fallback_frame,
                )
            return data_runtime._validate_sensitive_runtime(sensitive_payload, context)

        data_runtime._sensitive_train = _extract_sensitive(
            data_runtime.X_train,
            data_runtime._X.iloc[train_indices].reset_index(drop=True),
            "train sampling",
        )
        data_runtime._sensitive_test = _extract_sensitive(
            data_runtime.X_test,
            data_runtime._X.iloc[test_indices].reset_index(drop=True),
            "test sampling",
        )
        data_runtime._sensitive_all = _extract_sensitive(
            data_runtime.X,
            data_runtime._X,
            "full-data sampling",
        )
        val_indices = getattr(data_runtime, "val_indices", None)
        if val_indices is not None and len(val_indices) > 0:
            data_runtime._sensitive_val = _extract_sensitive(
                data_runtime.X_val,
                data_runtime._X.iloc[val_indices].reset_index(drop=True),
                "val sampling",
            )
        else:
            data_runtime._sensitive_val = None
        return data_runtime

    def sample(self, run_hooks: bool = True) -> "DataConfig":
        """Populate split-aligned sensitive feature payloads after data sampling.

        Args:
            run_hooks: Whether the parent data runtime should execute hook callbacks.

        Returns:
            The current fairness behavior instance.
        """
        DataConfig.sample(cast(DataConfig, self), run_hooks=run_hooks)

        return self._refresh_sensitive_splits()


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

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
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
        files = cast(Any, kwargs.pop("files", None))
        result = self.execute_data_runtime(*args, files=files, **kwargs)
        assert hasattr(self, "X_train"), ".X_train not found"
        return result

    plugins: list = field(
        default_factory=default_fairlearn_data_plugins,
        metadata={"help": "Configuration field: plugins."},
        repr=True,
    )

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
            matches = self._resolve_sensitive_column_matches(
                [str(c) for c in self._X.columns],
                str(col),
            )
            assert len(matches) > 0
        return self


__all__ = [
    "FAIRLEARN_PIPELINE_HOOKS",
    "FAIRLEARN_SCORING_HOOKS",
    "FairnessBehaviorMixin",
    "default_fairlearn_data_plugins",
    "FairlearnDataConfig",
]
