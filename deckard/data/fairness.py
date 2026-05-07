from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import pandas as pd
from omegaconf import DictConfig, ListConfig

from .base import DataConfig, DataPipelineConfig
from ..utils import coerce_to_list, is_default_config_value, load_class, merge_list_of_dicts

@dataclass(eq=False)
class FairlearnDataConfig(DataPipelineConfig):
    """Data pipeline config with fairlearn-sensitive feature support."""

    sensitive_columns: Optional[Union[str, list]] = None
    fairness_defense: Union[None, bool, Dict[str, Any], list] = None

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

    def _sensitive_labels_from_frame(self, frame: Optional[Union[pd.DataFrame, pd.Series]]) -> pd.Series:
        """Build a single sensitive-feature label series for fairlearn APIs.

        Parameters
        ----------
        frame : pd.DataFrame or pd.Series
            The feature matrix (post-split) from which to extract sensitive columns.

        Returns
        -------
        pd.Series
            A series of sensitive-feature labels aligned with *frame*.

        Raises
        ------
        ValueError
            If *frame* is ``None``, ``sensitive_columns`` is not configured, or the
            column is not found in *frame*.
        """
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
        """Validate a sensitive-feature series at runtime.

        Parameters
        ----------
        sensitive : pd.Series
            The candidate sensitive-feature labels to validate.
        context : str
            A short descriptor (e.g. ``'train sampling'``) used in error messages.

        Returns
        -------
        pd.Series
            The validated series (unchanged when valid).

        Raises
        ------
        ValueError
            If the series is empty, all-null, or all-blank.
        """
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
        """Inject a fairlearn preprocessing defense into the DataPipelineConfig pipeline."""
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
        if (
            not hasattr(self, "_X")
            or self._X is None
            or not isinstance(self._X, pd.DataFrame)
        ):
            return

        if self.sensitive_columns is None:
            raise ValueError("sensitive_columns must be configured")
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

    def _init_pipeline(self):
        self._inject_fairness_defense_step()
        return super()._init_pipeline()

    def _sample(self):
        """Override _sample to cache sensitive labels used by fairlearn metrics."""
        super()._sample()

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
        # Compute sensitive labels for the validation split when present.
        if getattr(self, "X_val", None) is not None:
            self._sensitive_val = self._sensitive_labels_from_frame(self.X_val)
            self._sensitive_val = self._validate_sensitive_runtime(
                self._sensitive_val,
                "val sampling",
            )
        else:
            self._sensitive_val = None

    def _score(self) -> dict:
        """Thin wrapper that delegates fairness dataset scoring to ``self.scorer``."""
        if is_default_config_value(self.scorer, include_best=False):
            scorer_cls = (
                "deckard.score.data.DefaultDataClassificationConfig"
                if self.classifier
                else "deckard.score.data.DefaultDataRegressionConfig"
            )
            self.scorer = load_class(scorer_cls)
        if self.scorer is None:
            return {}
        if not callable(self.scorer):
            raise TypeError(
                f"FairlearnDataConfig.scorer must be callable or None, got {type(self.scorer)}",
            )
        y_true = (
            self.y_train if getattr(self, "y_train", None) is not None else self._y
        )
        y_pred = (
            self.X_train if getattr(self, "X_train", None) is not None else self._X
        )
        if isinstance(y_pred, pd.DataFrame):
            non_numeric = y_pred.select_dtypes(exclude=["number"]).columns
            if len(non_numeric) > 0:
                y_pred = pd.get_dummies(y_pred, drop_first=False)
        fairness_scores = self.scorer(
            y_true=y_true,
            y_pred=y_pred,
            mode=None,
            data=self,
        )
        return {"fairness_scores": fairness_scores}
