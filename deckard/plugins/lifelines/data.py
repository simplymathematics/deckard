"""Survival-specific data configuration and mode management."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from ...data.base import DataConfig
from ...frameworks.types import StringifiedClass


class LifelinesDataMode(str, Enum):
    """Enumeration of survival analysis data modes."""

    NATIVE = "native"
    AUXILIARY_METRIC = "auxiliary_metric"
    AUXILIARY_FAILURE = "auxiliary_failure"
    OPTUNA_DB = "optuna_db"
    # Backwards-compatible aliases.
    AUXILIARY_MODEL = AUXILIARY_METRIC
    AUXILIARY_ATTACK = AUXILIARY_FAILURE


LifelinesScalar = str | int | float | bool | None
LifelinesValue = LifelinesScalar | list["LifelinesValue"] | dict[str, "LifelinesValue"]


class LifelinesValidationMixin:
    """Reusable validation behavior for lifelines-compatible configs.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    mode: "LifelinesDataMode"
    duration_col: str
    event_col: str
    reference_metric: str
    failure_profile: Optional[dict[str, Any]]
    optuna_db: Optional[str]

    def _validate_native_mode(self) -> None:
        if self.duration_col in [None, ""]:
            raise ValueError("duration_col required for NATIVE mode")
        if self.event_col in [None, ""]:
            raise ValueError("event_col required for NATIVE mode")

    def _validate_auxiliary_metric_mode(self) -> None:
        if self.reference_metric in [None, ""]:
            raise ValueError(
                "reference_metric required for AUXILIARY_METRIC mode",
            )

    def _validate_auxiliary_failure_mode(self) -> None:
        if self.failure_profile is None:
            raise ValueError(
                "failure_profile required for AUXILIARY_FAILURE mode",
            )

    def _validate_optuna_db_mode(self) -> None:
        if self.optuna_db is None:
            raise ValueError("optuna_db path required for OPTUNA_DB mode")

    def _validate_mode_requirements(self) -> None:
        if self.mode == LifelinesDataMode.NATIVE:
            self._validate_native_mode()
        elif self.mode == LifelinesDataMode.AUXILIARY_METRIC:
            self._validate_auxiliary_metric_mode()
        elif self.mode == LifelinesDataMode.AUXILIARY_FAILURE:
            self._validate_auxiliary_failure_mode()
        elif self.mode == LifelinesDataMode.OPTUNA_DB:
            self._validate_optuna_db_mode()


@dataclass(eq=False, kw_only=True)
class LifelinesDataConfig(LifelinesValidationMixin, DataConfig):
    """DataConfig specialization for survival-analysis mode management.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    mode: LifelinesDataMode = field(default=LifelinesDataMode.NATIVE)
    duration_col: str = "T"
    event_col: str = "E"
    reference_metric: str = "accuracy"
    failure_profile: Optional[dict[str, Any]] = None
    optuna_db: Optional[str] = None
    optuna_schema: Optional[Union[str, dict[str, Any]]] = None
    optuna_query: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate mode-specific requirements after dataclass initialization."""
        super().__post_init__()
        self._validate_mode_requirements()

    @classmethod
    def from_data_and_model(
        cls,
        data_config: DataConfig,
        duration_col: str = "T",
        event_col: str = "E",
    ) -> "LifelinesDataConfig":
        """Build native survival-mode config from an existing data config.

        Args:
                data_config: Source data configuration to adapt.
                duration_col: Duration column name for survival labels.
                event_col: Event indicator column name for survival labels.

        Returns:
                Lifelines data config in native survival mode.
        """
        return cls(
            mode=LifelinesDataMode.NATIVE,
            dataset_name=data_config.dataset_name,
            target=data_config.target,
            classifier=False,
            duration_col=duration_col,
            event_col=event_col,
        )

    @classmethod
    def from_auxiliary_model(
        cls,
        data_config: DataConfig,
        benign_metric: str = "accuracy",
    ) -> "LifelinesDataConfig":
        """Backwards-compatible wrapper for from_auxiliary_metric."""
        return cls.from_auxiliary_metric(
            data_config,
            reference_metric=benign_metric,
        )

    @classmethod
    def from_auxiliary_metric(
        cls,
        data_config: DataConfig,
        reference_metric: str = "accuracy",
    ) -> "LifelinesDataConfig":
        """Build auxiliary-model mode config from an existing data config.

        Args:
                data_config: Source data configuration to adapt.
                reference_metric: Metric name used for reference failure derivation.

        Returns:
                Lifelines data config in auxiliary-metric mode.
        """
        return cls(
            mode=LifelinesDataMode.AUXILIARY_METRIC,
            dataset_name=data_config.dataset_name,
            target=data_config.target,
            classifier=data_config.classifier,
            reference_metric=reference_metric,
        )

    @classmethod
    def from_auxiliary_attack(
        cls,
        data_config: DataConfig,
        attack_config: dict[str, LifelinesValue],
    ) -> "LifelinesDataConfig":
        """Backwards-compatible wrapper for from_auxiliary_failure."""
        return cls.from_auxiliary_failure(
            data_config,
            failure_profile=attack_config,
        )

    @classmethod
    def from_auxiliary_failure(
        cls,
        data_config: DataConfig,
        failure_profile: dict[str, LifelinesValue],
    ) -> "LifelinesDataConfig":
        """Build auxiliary-attack mode config from an existing data config.

        Args:
                data_config: Source data configuration to adapt.
                failure_profile: Failure-signal configuration payload used for
                    auxiliary failure mode. Can describe attack and non-attack
                    failure sources.

        Returns:
                Lifelines data config in auxiliary-failure mode.
        """
        return cls(
            mode=LifelinesDataMode.AUXILIARY_FAILURE,
            dataset_name=data_config.dataset_name,
            target=data_config.target,
            classifier=data_config.classifier,
            failure_profile=failure_profile,
        )

    @classmethod
    def from_optuna_db(
        cls,
        optuna_db: str,
        dataset_name: StringifiedClass = "optuna",
        optuna_schema: Optional[Union[str, dict[str, LifelinesValue]]] = None,
        optuna_query: Optional[str] = None,
    ) -> "LifelinesDataConfig":
        """Build optuna-db mode config from an Optuna database source.

        Args:
                optuna_db: Path or DSN for the Optuna database.
                dataset_name: Dataset label used for runtime naming.
                optuna_schema: Optional schema mapping for Optuna records.
                optuna_query: Optional query used to filter Optuna rows.

        Returns:
                Lifelines data config in Optuna database mode.
        """
        return cls(
            mode=LifelinesDataMode.OPTUNA_DB,
            dataset_name=dataset_name,
            target="optuna_result",
            classifier=False,
            optuna_db=optuna_db,
            optuna_schema=optuna_schema,
            optuna_query=optuna_query,
        )

    def is_native_survival_data(self) -> bool:
        """Return whether this config is in native survival-data mode.

        Returns:
            ``True`` when mode is native survival data.
        """
        return self.mode == LifelinesDataMode.NATIVE

    def has_auxiliary_model(self) -> bool:
        """Backwards-compatible wrapper for has_auxiliary_metric."""
        return self.has_auxiliary_metric()

    def has_auxiliary_metric(self) -> bool:
        """Return whether this config is in auxiliary-model mode.

        Returns:
            ``True`` when mode is auxiliary metric.
        """
        return self.mode == LifelinesDataMode.AUXILIARY_METRIC

    def has_auxiliary_attack(self) -> bool:
        """Backwards-compatible wrapper for has_auxiliary_failure."""
        return self.has_auxiliary_failure()

    def has_auxiliary_failure(self) -> bool:
        """Return whether this config is in auxiliary-attack mode.

        Returns:
            ``True`` when mode is auxiliary failure.
        """
        return self.mode == LifelinesDataMode.AUXILIARY_FAILURE

    @property
    def benign_metric(self) -> str:
        """Backwards-compatible alias for reference_metric."""
        return self.reference_metric

    @benign_metric.setter
    def benign_metric(self, value: str) -> None:
        self.reference_metric = value

    @property
    def attack_config(self) -> Optional[dict[str, Any]]:
        """Backwards-compatible alias for failure_profile."""
        return self.failure_profile

    @attack_config.setter
    def attack_config(self, value: Optional[dict[str, Any]]) -> None:
        self.failure_profile = value

    def is_optuna_db(self) -> bool:
        """Return whether this config is in Optuna database mode.

        Returns:
            ``True`` when mode is Optuna DB.
        """
        return self.mode == LifelinesDataMode.OPTUNA_DB


__all__ = ["LifelinesDataConfig", "LifelinesDataMode"]
