"""Utilities for selecting top Optuna trials and Pareto fronts."""

import logging
from pathlib import Path

import optuna
import pandas as pd
from paretoset import paretoset

from ..utils import create_parser_from_function, save_data
from ._trial_utils import (
    complete_trials_only as _complete_trials_only,
)
from ._trial_utils import (
    infer_default_optimizers as _infer_default_optimizers,
)
from ._trial_utils import (
    normalize_direction as _normalize_direction,
)
from ._trial_utils import (
    objective_to_column as _objective_to_column,
)
from ._trial_utils import (
    parse_csv_arg as _parse_csv_arg,
)
from ._trial_utils import (
    resolve_study as _resolve_study,
)

logger = logging.getLogger(__name__)


def _infer_directions_for_objectives(
    study: optuna.study.Study,
    objective_names: list[str],
    objective_columns: list[str],
    metric_names: list[str],
) -> list[str]:
    """Infer directions per selected objective.

    If an objective does not map to an Optuna study objective, default to maximize.
    """
    raw_directions = [
        _normalize_direction(str(d))
        for d in list(getattr(study, "directions", []) or [])
    ]
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
                "Could not infer direction for objective '%s' (column '%s') "
                "from study objectives; defaulting to maximize.",
                name,
                column,
            )
            resolved_directions.append("maximize")
        else:
            resolved_directions.append(raw_directions[idx])

    return resolved_directions


def _coerce_objective_columns_numeric(
    trial_df: pd.DataFrame,
    objective_columns: list[str],
) -> pd.DataFrame:
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


def pareto_main(
    output_file: str,
    optuna_db: str,
    study_name: str = None,
    optimizers: str = None,
    directions: str = None,
    top_k: int = 1,
) -> None:
    """Select best Optuna trials by one or more optimization criteria.

    Args:
        output_file: Output path for selected trial rows.
        optuna_db: Optuna storage URI, for example ``sqlite:///optuna.db``.
        study_name: Name of the Optuna study. When omitted, this command
            requires exactly one study in storage and uses it automatically.
        optimizers: Comma-separated optimization criteria such as
            ``"accuracy,evasion_accuracy"``. When omitted, criteria are
            inferred from study metric names or trial columns.
        directions: Comma-separated objective directions in
            ``{maximize,minimize,diff,max,min}``. When omitted, directions are
            inferred from the Optuna study.
        top_k: Number of trials to return for single-objective optimization.
            Multi-objective optimization returns all Pareto-optimal trials.
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
        objective_names = _infer_default_optimizers(
            study=study,
            trials_df=trial_df,
        )

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
            trial_df.sort_values(by=column, ascending=ascending).head(top_k).copy()
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
