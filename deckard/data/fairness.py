from typing import Any, Dict, Optional, Union
import pandas as pd
from dataclasses import dataclass
from omegaconf import DictConfig, ListConfig

from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    demographic_parity_ratio,
    mean_prediction,
    selection_rate,
)
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

from .data import DataPipelineConfig

fairness_scores = {
    "classification": [
        "count",
        "selection_rate",
        "selection_rate_difference",
        "selection_rate_ratio",
        "demographic_parity_difference",
        "demographic_parity_ratio",
        "target_mutual_information",
        "target_normalized_mutual_information",
    ],
    "regression": [
        "count",
        "mean_target",
        "mean_target_difference",
        "mean_target_ratio",
    ],
}


@dataclass
class FairnessDataConfig(DataPipelineConfig):
    """
    Extended DataConfig class that overloads key methods to operate on pandas groupby objects.

    This allows stratified analysis of fairness metrics across different demographic groups.
    """

    groupby_columns: Union[str, list] = None
    sensitive_columns: Optional[Union[str, list]] = None
    fairness_defense: Union[None, bool, Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize with groupby_column support."""
        super().__post_init__()
        self._validate_init()
        if self.groupby_columns is None:
            raise ValueError("groupby_column must be specified for FairnessDataConfig")

        if isinstance(self.groupby_columns, ListConfig):
            self.groupby_columns = list(self.groupby_columns)
        if isinstance(self.groupby_columns, str):
            self.groupby_columns = [self.groupby_columns]

        if self.sensitive_columns is None:
            self.sensitive_columns = list(self.groupby_columns)
        elif isinstance(self.sensitive_columns, ListConfig):
            self.sensitive_columns = list(self.sensitive_columns)
        elif isinstance(self.sensitive_columns, str):
            self.sensitive_columns = [self.sensitive_columns]

    def _group_labels_from_frame(self, frame: pd.DataFrame) -> pd.Series:
        """Build a single sensitive-feature label series for fairlearn APIs."""
        if len(self.groupby_columns) == 1:
            return frame[self.groupby_columns[0]].astype(str)
        return frame[self.groupby_columns]

    def _validate_sensitive_runtime(
        self,
        sensitive: pd.Series,
        context: str,
    ) -> pd.Series:
        sensitive_series = pd.Series(sensitive)
        if len(sensitive_series) == 0:
            raise ValueError(f"Sensitive features are empty during {context}")
        if sensitive_series.dropna().empty:
            raise ValueError(f"Sensitive features are all null during {context}")
        if sensitive_series.astype(str).str.strip().eq("").all():
            raise ValueError(f"Sensitive features are blank during {context}")
        return sensitive_series

    def _inject_fairness_defense_step(self) -> None:
        """Inject a fairlearn preprocessing defense into the DataPipelineConfig pipeline."""
        # Default behavior is no fairness pipeline step; opt-in via explicit config.
        if self.fairness_defense in [None, False]:
            return
        if self.fairness_defense is True:
            raise ValueError(
                "fairness_defense=True is ambiguous. Provide a config dict with at least a 'name' key.",
            )
        if not isinstance(self.fairness_defense, (dict, DictConfig)):
            raise TypeError(
                f"fairness_defense must be a dict/DictConfig, False, or None. Got {type(self.fairness_defense)}",
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

        sensitive_feature_ids = list(sensitive_columns)
        step_config: Dict[str, Any] = {
            "sensitive_feature_ids": sensitive_feature_ids,
        }
        step_name = "fairness_correlation_remover"
        custom = dict(self.fairness_defense)
        step_name = custom.pop("step_name", step_name)
        step_config.update(custom)
        if "name" not in step_config:
            raise ValueError("fairness_defense config must include a 'name' key")

        if step_name in self.pipeline:
            return

        self.pipeline = {step_name: step_config, **self.pipeline}

    def _load_data(self):
        super()._load_data()
        assert hasattr(self, "_X"), RuntimeError(
            "self.X_ not found while loading FairnessDataConfig",
        )
        assert hasattr(self, "_y"), RuntimeError(
            "self.y_ not found whilte loading FairnessDataConfig",
        )
        assert isinstance(self._X, pd.DataFrame), ValueError(
            "Expected a dataframe for self.X_",
        )
        for col in self.groupby_columns:
            assert col in self._X.columns
        return self

    def _init_pipeline(self):
        self._inject_fairness_defense_step()
        return super()._init_pipeline()

    def _sample(self):
        """Override _sample to handle groupby objects for fairness analysis.

        Keeps X_train, y_train the same for all groups, but creates separate
        X_test, y_test groups based on the groupby columns.
        """
        # Call parent _sample to get standard train/test split
        super()._sample()

        # Cache sensitive labels before pipeline transforms can drop sensitive columns.
        self._sensitive_train = self._group_labels_from_frame(self.X_train)
        self._sensitive_test = self._group_labels_from_frame(self.X_test)
        self._sensitive_all = self._group_labels_from_frame(self._X)
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
        """Compute dataset-only fairness metrics using fairlearn."""
        if (
            hasattr(self, "X_test")
            and hasattr(self, "y_test")
            and self.X_test is not None
            and self.y_test is not None
        ):
            X_eval = self.X_test
            y_eval = self.y_test
            sensitive_test = getattr(self, "_sensitive_test", None)
            if sensitive_test is not None and len(sensitive_test) == len(
                y_eval,
            ):
                sensitive = sensitive_test
            else:
                sensitive = self._group_labels_from_frame(X_eval)
        else:
            X_eval = self._X
            y_eval = self._y
            sensitive_all = getattr(self, "_sensitive_all", None)
            if sensitive_all is not None and len(sensitive_all) == len(
                y_eval,
            ):
                sensitive = sensitive_all
            else:
                sensitive = self._group_labels_from_frame(X_eval)
        sensitive = self._validate_sensitive_runtime(sensitive, "fairness scoring")
        if self.classifier:
            metric_frame = MetricFrame(
                metrics={"count": count, "selection_rate": selection_rate},
                y_true=y_eval,
                y_pred=y_eval,
                sensitive_features=sensitive,
            )
            selection_diff = metric_frame.difference(method="between_groups")
            selection_ratio = metric_frame.ratio(method="between_groups")

            # Examine association between sensitive group membership and target labels.
            sensitive_labels = pd.Series(sensitive).astype(str)
            target_labels = pd.Series(y_eval).astype(str)
            target_distribution_by_group = pd.crosstab(
                sensitive_labels,
                target_labels,
                normalize="index",
            ).to_dict(orient="index")
            fairness_payload = {
                "fairness_scores": {
                    "metrics": fairness_scores["classification"],
                    "overall": {
                        "count": int(metric_frame.overall["count"]),
                        "selection_rate": float(metric_frame.overall["selection_rate"]),
                    },
                    "by_group": {
                        str(group): {
                            "count": int(values["count"]),
                            "selection_rate": float(values["selection_rate"]),
                        }
                        for group, values in metric_frame.by_group.to_dict(
                            orient="index",
                        ).items()
                    },
                    "target_distribution_by_group": {
                        str(group): {
                            str(label): float(prob) for label, prob in values.items()
                        }
                        for group, values in target_distribution_by_group.items()
                    },
                    "selection_rate_difference": float(
                        selection_diff["selection_rate"],
                    ),
                    "selection_rate_ratio": float(selection_ratio["selection_rate"]),
                    "demographic_parity_difference": float(
                        demographic_parity_difference(
                            y_true=y_eval,
                            y_pred=y_eval,
                            sensitive_features=sensitive,
                        ),
                    ),
                    "demographic_parity_ratio": float(
                        demographic_parity_ratio(
                            y_true=y_eval,
                            y_pred=y_eval,
                            sensitive_features=sensitive,
                        ),
                    ),
                    "target_mutual_information": float(
                        mutual_info_score(sensitive_labels, target_labels),
                    ),
                    "target_normalized_mutual_information": float(
                        normalized_mutual_info_score(sensitive_labels, target_labels),
                    ),
                },
            }
            return fairness_payload
        metric_frame = MetricFrame(
            metrics={"count": count, "mean_target": mean_prediction},
            y_true=y_eval,
            y_pred=y_eval,
            sensitive_features=sensitive,
        )
        by_group = metric_frame.by_group.to_dict(orient="index")
        means = [float(values["mean_target"]) for values in by_group.values()]
        mean_diff = max(means) - min(means) if means else 0.0
        mean_ratio = (min(means) / max(means)) if means and max(means) != 0 else 0.0
        return {
            "fairness_scores": {
                "metrics": fairness_scores["regression"],
                "overall": {
                    "count": int(metric_frame.overall["count"]),
                    "mean_target": float(metric_frame.overall["mean_target"]),
                },
                "by_group": {
                    str(group): {
                        "count": int(values["count"]),
                        "mean_target": float(values["mean_target"]),
                    }
                    for group, values in by_group.items()
                },
                "mean_target_difference": float(mean_diff),
                "mean_target_ratio": float(mean_ratio),
            },
        }
