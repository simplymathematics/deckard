import logging
from pathlib import Path
from typing import Optional

import optuna
import pandas as pd
from paretoset import paretoset

from ..utils import create_parser_from_function, save_data


logger = logging.getLogger(__name__)


def _parse_csv_arg(value: Optional[str]) -> list[str]:
	if value is None:
		return []
	value = value.strip()
	if value == "":
		return []
	return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_direction(direction: str) -> str:
	d = direction.strip().lower()
	if "." in d:
		d = d.split(".")[-1]
	if d in {"maximize", "max"}:
		return "maximize"
	if d in {"minimize", "min"}:
		return "minimize"
	if d == "diff":
		return "diff"
	raise ValueError(
		f"Unsupported direction '{direction}'. Use maximize/minimize/diff (or max/min).",
	)


def _resolve_study(optuna_db: str, study_name: Optional[str]) -> optuna.study.Study:
	if study_name:
		logger.info(f"Loading study '{study_name}' from {optuna_db}")
		return optuna.study.load_study(storage=optuna_db, study_name=study_name)

	summaries = optuna.study.get_all_study_summaries(storage=optuna_db)
	if len(summaries) == 0:
		raise ValueError(f"No studies found in {optuna_db}")
	if len(summaries) > 1:
		names = [getattr(s, "study_name", getattr(s, "name", "<unknown>")) for s in summaries]
		raise ValueError(
			"Multiple studies found. Please provide study_name. "
			f"Available studies: {names}",
		)

	inferred_name = getattr(summaries[0], "study_name", getattr(summaries[0], "name", None))
	if inferred_name is None:
		raise ValueError("Could not infer study name from summary")
	logger.info(f"No study_name provided; using only available study '{inferred_name}'")
	return optuna.study.load_study(storage=optuna_db, study_name=inferred_name)


def _infer_default_optimizers(study: optuna.study.Study, trials_df: pd.DataFrame) -> list[str]:
	metric_names = list(getattr(study, "metric_names", []) or [])
	if len(metric_names) > 0:
		return metric_names

	value_cols = sorted(
		[c for c in trials_df.columns if c.startswith("values_")],
		key=lambda c: int(c.split("_")[1]),
	)
	if len(value_cols) > 0:
		return value_cols
	if "value" in trials_df.columns:
		return ["value"]

	raise ValueError(
		"Could not infer objective columns. Provide optimizers explicitly.",
	)


def _infer_directions_for_objectives(
	study: optuna.study.Study,
	objective_names: list[str],
	objective_columns: list[str],
	metric_names: list[str],
) -> list[str]:
	"""Infer directions per selected objective.

	If an objective does not map to an Optuna study objective, default to maximize.
	"""
	raw_directions = [_normalize_direction(str(d)) for d in list(getattr(study, "directions", []) or [])]
	if len(raw_directions) == 0:
		return ["maximize"] * len(objective_names)

	if len(metric_names) == 0 and len(raw_directions) == len(objective_names):
		return raw_directions

	resolved_directions: list[str] = []
	for name, column in zip(objective_names, objective_columns):
		idx = None
		if name in metric_names:
			idx = metric_names.index(name)
		elif column.startswith("values_"):
			suffix = column.removeprefix("values_")
			if suffix.isdigit():
				cand = int(suffix)
				if cand < len(raw_directions):
					idx = cand
			elif suffix in metric_names:
				idx = metric_names.index(suffix)

		if idx is None:
			logger.warning(
				"Could not infer direction for objective '%s' (column '%s') from study objectives; defaulting to maximize.",
				name,
				column,
			)
			resolved_directions.append("maximize")
		else:
			resolved_directions.append(raw_directions[idx])

	return resolved_directions


def _objective_to_column(
	objective: str,
	objective_idx: int,
	trials_df: pd.DataFrame,
	metric_names: list[str],
	allow_index_fallback: bool = True,
) -> str:
	if objective in trials_df.columns:
		return objective

	if objective.startswith("values_") and objective in trials_df.columns:
		return objective

	named_value_column = f"values_{objective}"
	if named_value_column in trials_df.columns:
		return named_value_column

	named_attr_column = f"user_attrs_{objective}"
	if named_attr_column in trials_df.columns:
		return named_attr_column

	named_param_column = f"params_{objective}"
	if named_param_column in trials_df.columns:
		return named_param_column

	if len(metric_names) > 0 and objective in metric_names:
		idx = metric_names.index(objective)
		candidate = f"values_{idx}"
		if candidate in trials_df.columns:
			return candidate

	if allow_index_fallback:
		fallback = f"values_{objective_idx}"
		if fallback in trials_df.columns:
			return fallback

	if objective_idx == 0 and "value" in trials_df.columns:
		return "value"

	raise ValueError(
		f"Could not map optimizer '{objective}' to a trials dataframe column. "
		f"Available columns: {list(trials_df.columns)}",
	)


def _coerce_objective_columns_numeric(trial_df: pd.DataFrame, objective_columns: list[str]) -> pd.DataFrame:
	result = trial_df.copy()
	for col in objective_columns:
		if pd.api.types.is_numeric_dtype(result[col]):
			continue
		coerced = pd.to_numeric(result[col], errors="coerce")
		if coerced.isna().all():
			raise ValueError(
				f"Objective column '{col}' is non-numeric and cannot be optimized. "
				"Select numeric criteria or provide different objective columns.",
			)
		result[col] = coerced
	return result


def _complete_trials_only(trials_df: pd.DataFrame) -> pd.DataFrame:
	if "state" not in trials_df.columns:
		return trials_df

	state_text = trials_df["state"].astype(str)
	complete_mask = state_text.str.upper().str.contains("COMPLETE")
	return trials_df.loc[complete_mask].copy()


def pareto_main(
	output_file: str,
	optuna_db: str,
	study_name: str = None,
	optimizers: str = None,
	directions: str = None,
	top_k: int = 1,
):
	"""Select best Optuna trials by one or more optimization criteria.

	Parameters
	----------
	output_file : str
		Output path for selected trial rows (csv/parquet/pkl/html/json/xlsx).
	optuna_db : str
		Optuna storage URI (for example: sqlite:///optuna.db).
	study_name : str, optional
		Name of the Optuna study. If omitted, this command requires exactly one study
		in storage and uses it automatically.
	optimizers : str, optional
		Comma-separated optimization criteria (Hydra-style objective names).
		Example: "accuracy,evasion_accuracy".
		If omitted, criteria are inferred from study metric names or trial columns.
	directions : str, optional
		Comma-separated objective directions, each in {maximize,minimize,diff,max,min}.
		If omitted, directions are inferred from the Optuna study.
	top_k : int, optional
		Number of trials to return for single-objective optimization.
		For multi-objective optimization, all Pareto-optimal trials are returned.
	"""
	if top_k < 1:
		raise ValueError(f"top_k must be >= 1. Got {top_k}")

	output_path = Path(output_file)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	study = _resolve_study(optuna_db=optuna_db, study_name=study_name)
	study_trial_df = study.trials_dataframe()
	trial_df = _complete_trials_only(study_trial_df)
	if len(trial_df) == 0:
		raise ValueError("No COMPLETE trials found in selected study")

	objective_names = _parse_csv_arg(optimizers)
	user_specified_objectives = len(objective_names) > 0
	if len(objective_names) == 0:
		objective_names = _infer_default_optimizers(study=study, trials_df=trial_df)

	metric_names = list(getattr(study, "metric_names", []) or [])
	objective_columns = [
		_objective_to_column(
			name,
			idx,
			trial_df,
			metric_names,
			allow_index_fallback=not user_specified_objectives,
		)
		for idx, name in enumerate(objective_names)
	]
	trial_df = _coerce_objective_columns_numeric(trial_df, objective_columns)

	direction_list = [_normalize_direction(d) for d in _parse_csv_arg(directions)]
	if len(direction_list) == 0:
		direction_list = _infer_directions_for_objectives(
			study=study,
			objective_names=objective_names,
			objective_columns=objective_columns,
			metric_names=metric_names,
		)

	if len(objective_names) != len(direction_list):
		raise ValueError(
			"optimizers and directions must have matching lengths. "
			f"Got {len(objective_names)} optimizers and {len(direction_list)} directions.",
		)

	if len(objective_columns) == 1:
		if direction_list[0] == "diff":
			raise ValueError(
				"Direction 'diff' requires multi-objective Pareto selection. "
				"Use minimize/maximize for single-objective selection.",
			)
		column = objective_columns[0]
		ascending = direction_list[0] == "minimize"
		selected = (
			trial_df.sort_values(by=column, ascending=ascending)
			.head(top_k)
			.copy()
		)
		selected["_selection_type"] = "single_objective"
	else:
		sense_map = {"maximize": "max", "minimize": "min", "diff": "diff"}
		senses = [sense_map[d] for d in direction_list]
		mask = paretoset(trial_df[objective_columns], sense=senses)
		selected = trial_df.loc[mask].copy()
		selected["_selection_type"] = "pareto_front"

	selected["_study_name"] = study.study_name
	selected["_objective_names"] = ",".join(objective_names)
	selected["_directions"] = ",".join(direction_list)

	logger.info(
		"Selected %s trial(s) from study '%s' using objectives=%s directions=%s",
		len(selected),
		study.study_name,
		objective_names,
		direction_list,
	)
	save_data(data=selected, filepath=str(output_path))


pareto_parser = create_parser_from_function(pareto_main)


if __name__ == "__main__":
	args = pareto_parser.parse_args()
	pareto_main(**vars(args))