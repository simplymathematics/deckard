"""Shared Optuna trial parsing helpers for layers utilities."""

from __future__ import annotations

from typing import Optional

import optuna
import pandas as pd


def parse_csv_arg(value: Optional[str]) -> list[str]:
    if value is None:
        return []
    text = value.strip()
    if text == "":
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def normalize_direction(direction: str) -> str:
    token = str(direction).strip().lower()
    if "." in token:
        token = token.rsplit(".", maxsplit=1)[-1]
    if token in {"maximize", "max"}:
        return "maximize"
    if token in {"minimize", "min"}:
        return "minimize"
    if token == "diff":
        return "diff"
    raise ValueError(
        f"Invalid direction '{direction}'. Use maximize/minimize/diff.",
    )


def resolve_study(optuna_db: str, study_name: Optional[str]) -> optuna.study.Study:
    if study_name not in {None, ""}:
        return optuna.study.load_study(storage=optuna_db, study_name=str(study_name))

    summaries = optuna.study.get_all_study_summaries(storage=optuna_db)
    if len(summaries) == 0:
        raise ValueError(f"No studies found in {optuna_db}")
    if len(summaries) > 1:
        names = [
            getattr(summary, "study_name", getattr(summary, "name", "<unknown>"))
            for summary in summaries
        ]
        raise ValueError(
            "Multiple studies found. Provide study_name. "
            f"Available studies: {names}",
        )
    inferred = getattr(summaries[0], "study_name", getattr(summaries[0], "name", None))
    if inferred in {None, ""}:
        raise ValueError("Could not infer study name from Optuna summaries")
    return optuna.study.load_study(storage=optuna_db, study_name=str(inferred))


def complete_trials_only(trials_df: pd.DataFrame) -> pd.DataFrame:
    if "state" not in trials_df.columns:
        return trials_df
    state_text = trials_df["state"].astype(str)
    return trials_df.loc[state_text.str.upper().str.contains("COMPLETE")].copy()


def infer_default_optimizers(
    study: optuna.study.Study,
    trials_df: pd.DataFrame,
) -> list[str]:
    metric_names = list(getattr(study, "metric_names", []) or [])
    if len(metric_names) > 0:
        return [str(item) for item in metric_names]

    value_cols = sorted(
        [column for column in trials_df.columns if column.startswith("values_")],
        key=lambda column: int(column.split("_")[1]),
    )
    if len(value_cols) > 0:
        return value_cols
    if "value" in trials_df.columns:
        return ["value"]
    raise ValueError("Could not infer optimizer objectives from study trials")


def objective_to_column(
    objective: str,
    objective_idx: int,
    trials_df: pd.DataFrame,
    metric_names: list[str],
    *,
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
        metric_idx = metric_names.index(objective)
        candidate = f"values_{metric_idx}"
        if candidate in trials_df.columns:
            return candidate

    if allow_index_fallback:
        fallback = f"values_{objective_idx}"
        if fallback in trials_df.columns:
            return fallback

    if objective_idx == 0 and "value" in trials_df.columns:
        return "value"

    raise ValueError(
        f"Could not map objective '{objective}' to a trials dataframe column.",
    )
