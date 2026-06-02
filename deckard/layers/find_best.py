import logging
from pathlib import Path
from typing import Any, Optional

import optuna
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import ListConfig, OmegaConf
from paretoset import paretoset

from ..utils import create_parser_from_function

logger = logging.getLogger(__name__)


def _parse_csv_arg(value: Optional[str]) -> list[str]:
    if value is None:
        return []
    text = value.strip()
    if text == "":
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _normalize_direction(direction: str) -> str:
    token = str(direction).strip().lower()
    if "." in token:
        token = token.split(".")[-1]
    if token in {"maximize", "max"}:
        return "maximize"
    if token in {"minimize", "min"}:
        return "minimize"
    if token == "diff":
        return "diff"
    raise ValueError(
        f"Unsupported direction '{direction}'. Use maximize/minimize/diff.",
    )


def _resolve_study(optuna_db: str, study_name: Optional[str]) -> optuna.study.Study:
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


def _complete_trials_only(trials_df: pd.DataFrame) -> pd.DataFrame:
    if "state" not in trials_df.columns:
        return trials_df
    state_text = trials_df["state"].astype(str)
    return trials_df.loc[state_text.str.upper().str.contains("COMPLETE")].copy()


def _infer_objectives(study: optuna.study.Study, trials_df: pd.DataFrame) -> list[str]:
    metric_names = list(getattr(study, "metric_names", []) or [])
    if len(metric_names) > 0:
        return [str(item) for item in metric_names]

    value_cols = sorted(
        [c for c in trials_df.columns if c.startswith("values_")],
        key=lambda c: int(c.split("_")[1]),
    )
    if len(value_cols) > 0:
        return value_cols
    if "value" in trials_df.columns:
        return ["value"]
    raise ValueError("Could not infer optimizer objectives from study trials")


def _objective_to_column(
    objective: str,
    objective_idx: int,
    trials_df: pd.DataFrame,
    metric_names: list[str],
) -> str:
    if objective in trials_df.columns:
        return objective

    named = f"values_{objective}"
    if named in trials_df.columns:
        return named

    if objective in metric_names:
        idx = metric_names.index(objective)
        candidate = f"values_{idx}"
        if candidate in trials_df.columns:
            return candidate

    fallback = f"values_{objective_idx}"
    if fallback in trials_df.columns:
        return fallback

    if objective_idx == 0 and "value" in trials_df.columns:
        return "value"

    raise ValueError(
        f"Could not map objective '{objective}' to a trials dataframe column.",
    )


def _infer_directions(
    study: optuna.study.Study,
    objective_names: list[str],
    objective_columns: list[str],
    metric_names: list[str],
) -> list[str]:
    raw_directions = [
        _normalize_direction(str(direction))
        for direction in list(getattr(study, "directions", []) or [])
    ]
    if len(raw_directions) == 0:
        return ["maximize"] * len(objective_names)

    if len(metric_names) == 0 and len(raw_directions) == len(objective_names):
        return raw_directions

    resolved: list[str] = []
    for name, column in zip(objective_names, objective_columns):
        idx = None
        if name in metric_names:
            idx = metric_names.index(name)
        elif column.startswith("values_"):
            suffix = column.removeprefix("values_")
            if suffix.isdigit():
                parsed = int(suffix)
                if parsed < len(raw_directions):
                    idx = parsed
        if idx is None:
            resolved.append("maximize")
        else:
            resolved.append(raw_directions[idx])
    return resolved


def _coerce_objective_columns_numeric(
    trials_df: pd.DataFrame,
    objective_columns: list[str],
) -> pd.DataFrame:
    frame = trials_df.copy()
    for col in objective_columns:
        if pd.api.types.is_numeric_dtype(frame[col]):
            continue
        coerced = pd.to_numeric(frame[col], errors="coerce")
        if coerced.isna().all():
            raise ValueError(
                f"Objective column '{col}' is non-numeric and cannot be optimized.",
            )
        frame[col] = coerced
    return frame


def _load_config_optimizers_and_directions(
    config_dir: str,
    config_name: str,
) -> tuple[list[str], list[str]]:
    cfg_name = config_name.removesuffix(".yaml").removesuffix(".yml")
    config_root = Path(config_dir).resolve()
    if not config_root.exists():
        raise FileNotFoundError(f"Config directory not found: {config_root}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(config_root), version_base=None):
        cfg = compose(config_name=cfg_name)

    def _coerce_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, ListConfig)):
            return [str(item) for item in value]
        return [str(value)]

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        return [], []

    optimizers = _coerce_list(cfg_dict.get("optimizers"))
    directions = _coerce_list(cfg_dict.get("directions"))
    if len(directions) == 0:
        directions = _coerce_list(cfg_dict.get("direction"))
    return optimizers, [_normalize_direction(item) for item in directions]


def _split_subset_exprs(subset: Optional[str]) -> dict[str, str]:
    items = _parse_csv_arg(subset)
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid subset expression '{item}'. Use key=value")
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _normalize_trial_frame_columns(trials_df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in trials_df.columns:
        new_col = col
        if col.startswith("params_"):
            new_col = col.removeprefix("params_")
        elif col.startswith("user_attrs_"):
            new_col = col.removeprefix("user_attrs_")
        elif col.startswith("+") or col.startswith("~"):
            new_col = col[1:]
        renamed[col] = new_col
    return trials_df.rename(columns=renamed)


def _apply_subset_filter(
    trials_df: pd.DataFrame,
    subset: Optional[str],
) -> pd.DataFrame:
    subset_map = _split_subset_exprs(subset)
    if len(subset_map) == 0:
        return trials_df

    normalized = _normalize_trial_frame_columns(trials_df)
    filtered = normalized
    for key, value in subset_map.items():
        if key not in filtered.columns:
            raise ValueError(
                f"Subset key '{key}' not found in trial columns: {list(filtered.columns)}",
            )
        filtered = filtered.loc[filtered[key].astype(str) == value]
    return trials_df.loc[filtered.index].copy()


def _normalize_value_for_override(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    return str(value)


def _select_best_trial_number(
    trials_df: pd.DataFrame,
    objective_columns: list[str],
    directions: list[str],
) -> int:
    if len(objective_columns) == 1:
        column = objective_columns[0]
        direction = directions[0]
        ascending = direction == "minimize"
        sorted_df = trials_df.sort_values(by=column, ascending=ascending)
        return int(sorted_df.iloc[0]["number"])

    sense_map = {"maximize": "max", "minimize": "min", "diff": "diff"}
    senses = [sense_map[item] for item in directions]
    frontier_mask = paretoset(trials_df[objective_columns], sense=senses)
    frontier = trials_df.loc[frontier_mask].copy()
    if frontier.empty:
        raise ValueError("No Pareto-optimal trials found for selected objectives")

    rank_sum = pd.Series(0.0, index=frontier.index)
    for col, direction in zip(objective_columns, directions):
        if direction == "maximize":
            rank = frontier[col].rank(ascending=False, method="average")
        elif direction == "minimize":
            rank = frontier[col].rank(ascending=True, method="average")
        else:
            # For diff-style objectives, treat closeness to zero as better.
            rank = frontier[col].abs().rank(ascending=True, method="average")
        rank_sum = rank_sum + rank
    frontier = frontier.assign(_rank_sum=rank_sum)
    winner = frontier.sort_values(by=["_rank_sum", "number"], ascending=[True, True])
    return int(winner.iloc[0]["number"])


def find_best_main(
    output_file: str,
    optuna_db: str = "sqlite:///optuna.db",
    study_name: str = None,
    config_dir: str = "config",
    config_name: str = "default.yaml",
    optimizers: str = None,
    directions: str = None,
    subset: str = None,
    exclude: str = None,
) -> dict[str, Any]:
    """Recreate a composed ExperimentConfig YAML from the best Optuna trial.

    Args:
        output_file: Destination YAML path for composed best config.
        optuna_db: Optuna storage URI.
        study_name: Optuna study name. If omitted, storage must contain one study.
        config_dir: Hydra config directory used for composing final config.
        config_name: Hydra config name used for composing final config.
        optimizers: Optional comma-separated objective names to optimize.
            When omitted, reads ``optimizers`` from composed config, then study defaults.
        directions: Optional comma-separated objective directions
            (maximize|minimize|diff). When omitted, reads ``directions``/``direction``
            from config, then study defaults.
        subset: Optional comma-separated subset filters (``k=v``) applied to trials.
        exclude: Optional comma-separated parameter keys to omit from generated overrides.

    Returns:
        Metadata describing selected trial and destination output path.
    """
    study = _resolve_study(optuna_db=optuna_db, study_name=study_name)
    trials_df = study.trials_dataframe()
    trials_df = _complete_trials_only(trials_df)
    if len(trials_df) == 0:
        raise ValueError("No COMPLETE trials found in selected study")

    trials_df = _apply_subset_filter(trials_df, subset=subset)
    if len(trials_df) == 0:
        raise ValueError("No trials remaining after subset filter")

    cfg_optimizers, cfg_directions = _load_config_optimizers_and_directions(
        config_dir=config_dir,
        config_name=config_name,
    )

    objective_names = _parse_csv_arg(optimizers)
    if len(objective_names) == 0:
        objective_names = (
            cfg_optimizers
            if len(cfg_optimizers) > 0
            else _infer_objectives(study, trials_df)
        )

    metric_names = list(getattr(study, "metric_names", []) or [])
    objective_columns = [
        _objective_to_column(
            objective=name,
            objective_idx=idx,
            trials_df=trials_df,
            metric_names=metric_names,
        )
        for idx, name in enumerate(objective_names)
    ]
    trials_df = _coerce_objective_columns_numeric(trials_df, objective_columns)

    direction_values = [
        _normalize_direction(item) for item in _parse_csv_arg(directions)
    ]
    if len(direction_values) == 0:
        direction_values = cfg_directions
    if len(direction_values) == 0:
        direction_values = _infer_directions(
            study=study,
            objective_names=objective_names,
            objective_columns=objective_columns,
            metric_names=metric_names,
        )
    if len(direction_values) != len(objective_names):
        raise ValueError(
            "Length mismatch: directions and optimizers must match. "
            f"Got {len(direction_values)} directions vs {len(objective_names)} optimizers.",
        )

    best_trial_number = _select_best_trial_number(
        trials_df=trials_df,
        objective_columns=objective_columns,
        directions=direction_values,
    )
    best_trial = next(
        trial
        for trial in study.get_trials(deepcopy=False)
        if int(trial.number) == int(best_trial_number)
    )

    excluded = set(_parse_csv_arg(exclude))
    overrides: list[str] = []
    for key, value in (best_trial.params or {}).items():
        if key in excluded:
            continue
        overrides.append(f"++{key}={_normalize_value_for_override(value)}")

    config_root = Path(config_dir).resolve()
    cfg_name = config_name.removesuffix(".yaml").removesuffix(".yml")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_root), version_base=None):
        merged_cfg = compose(config_name=cfg_name, overrides=overrides)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=merged_cfg, f=output_path.as_posix(), resolve=True)

    logger.info(
        "Saved best-trial config for study '%s' trial #%s to %s",
        study.study_name,
        best_trial_number,
        output_path,
    )
    return {
        "study_name": study.study_name,
        "trial_number": int(best_trial_number),
        "output_file": output_path.as_posix(),
        "optimizers": objective_names,
        "directions": direction_values,
    }


find_best_parser = create_parser_from_function(find_best_main)


if __name__ == "__main__":
    args = find_best_parser.parse_args()
    find_best_main(**vars(args))
