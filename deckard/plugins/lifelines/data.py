"""Survival-specific data configuration and mode management."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

from ...data.base import DataConfig


class LifelinesDataMode(str, Enum):
	"""Enumeration of survival analysis data modes."""

	NATIVE = "native"
	AUXILIARY_MODEL = "auxiliary_model"
	AUXILIARY_ATTACK = "auxiliary_attack"
	OPTUNA_DB = "optuna_db"


class _LifelinesValidationMixin:
	"""Reusable validation behavior for lifelines-compatible configs."""

	mode: "LifelinesDataMode"
	duration_col: str
	event_col: str
	benign_metric: str
	attack_config: Optional[dict]
	optuna_db: Optional[str]

	def _validate_native_mode(self) -> None:
		if self.duration_col in [None, ""]:
			raise ValueError("duration_col required for NATIVE mode")
		if self.event_col in [None, ""]:
			raise ValueError("event_col required for NATIVE mode")

	def _validate_auxiliary_model_mode(self) -> None:
		if self.benign_metric in [None, ""]:
			raise ValueError(
				"benign_metric required for AUXILIARY_MODEL mode",
			)

	def _validate_auxiliary_attack_mode(self) -> None:
		if self.attack_config is None:
			raise ValueError(
				"attack_config required for AUXILIARY_ATTACK mode",
			)

	def _validate_optuna_db_mode(self) -> None:
		if self.optuna_db is None:
			raise ValueError("optuna_db path required for OPTUNA_DB mode")

	def _validate_mode_requirements(self) -> None:
		if self.mode == LifelinesDataMode.NATIVE:
			self._validate_native_mode()
		elif self.mode == LifelinesDataMode.AUXILIARY_MODEL:
			self._validate_auxiliary_model_mode()
		elif self.mode == LifelinesDataMode.AUXILIARY_ATTACK:
			self._validate_auxiliary_attack_mode()
		elif self.mode == LifelinesDataMode.OPTUNA_DB:
			self._validate_optuna_db_mode()


@dataclass(eq=False)
class LifelinesDataConfig(_LifelinesValidationMixin, DataConfig):
	"""DataConfig specialization for survival-analysis mode management."""

	mode: LifelinesDataMode = field(default=LifelinesDataMode.NATIVE)
	duration_col: str = "T"
	event_col: str = "E"
	benign_metric: str = "accuracy"
	attack_config: Optional[dict] = None
	optuna_db: Optional[str] = None
	optuna_schema: Optional[Union[str, dict]] = None
	optuna_query: Optional[str] = None

	def __post_init__(self):
		super().__post_init__()
		self._validate_mode_requirements()

	@classmethod
	def from_data_and_model(
		cls,
		data_config: DataConfig,
		duration_col: str = "T",
		event_col: str = "E",
	) -> "LifelinesDataConfig":
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
		return cls(
			mode=LifelinesDataMode.AUXILIARY_MODEL,
			dataset_name=data_config.dataset_name,
			target=data_config.target,
			classifier=data_config.classifier,
			benign_metric=benign_metric,
		)

	@classmethod
	def from_auxiliary_attack(
		cls,
		data_config: DataConfig,
		attack_config: dict,
	) -> "LifelinesDataConfig":
		return cls(
			mode=LifelinesDataMode.AUXILIARY_ATTACK,
			dataset_name=data_config.dataset_name,
			target=data_config.target,
			classifier=data_config.classifier,
			attack_config=attack_config,
		)

	@classmethod
	def from_optuna_db(
		cls,
		optuna_db: str,
		dataset_name: str = "optuna",
		optuna_schema: Optional[Union[str, dict]] = None,
		optuna_query: Optional[str] = None,
	) -> "LifelinesDataConfig":
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
		return self.mode == LifelinesDataMode.NATIVE

	def has_auxiliary_model(self) -> bool:
		return self.mode == LifelinesDataMode.AUXILIARY_MODEL

	def has_auxiliary_attack(self) -> bool:
		return self.mode == LifelinesDataMode.AUXILIARY_ATTACK

	def is_optuna_db(self) -> bool:
		return self.mode == LifelinesDataMode.OPTUNA_DB


__all__ = ["LifelinesDataConfig", "LifelinesDataMode"]
