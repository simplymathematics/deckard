# Script to query the database

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Union, cast

import optuna
import pandas as pd
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, ListConfig, OmegaConf
import yaml

storage = "sqlite:///optuna.db"
study_name = "gzip_knn_20-0"
metric_names = ["accuracy"]
directions = ["maximize"]
output_file = "optuna.csv"


def _normalize_optuna_storage(storage: Any = None) -> Any:
    if storage is None:
        storage = "sqlite:///optuna.db"
    if not isinstance(storage, str):
        return storage
    if str(storage).strip() == "":
        storage = "sqlite:///optuna.db"
    storage = str(storage)
    if "://" in storage:
        return storage
    path = Path(storage)
    if path.suffix in {".db", ".sqlite3"}:
        return f"sqlite:///{path.as_posix()}"
    return storage


def parse_study_name(
    study_name: str,
    schema: Union[dict[str, Any], str, None] = None,
) -> pd.DataFrame:
    """Parse study metadata from study name using schema mapping."""
    if schema is None:
        schema = {}
    if isinstance(schema, str):
        schema_path = Path(schema)
        assert schema_path.exists(), (
            "Schema must be a dictionary or an existing file path. " f"Got: {schema}"
        )
        with open(schema_path, "r") as handle:
            conf = yaml.safe_load(handle) or {}
        schema = conf.pop("schema", conf)

    if not isinstance(schema, dict):
        raise TypeError(f"schema must resolve to a dictionary, got {type(schema)}")
    schema_copy: dict[str, Any] = dict(schema)
    sep = schema_copy.pop("sep", "_")
    parts = study_name.split(sep)
    frame = pd.DataFrame()
    for key, value in schema_copy.items():
        if isinstance(value, int):
            frame[key] = [parts[value] if value < len(parts) else None]
            continue
        if isinstance(value, str):
            assert len(value.split(":")) == 2, (
                "Schema value should be an int index or inclusive range first:last. "
                f"Got: {value}"
            )
            start, end = map(int, value.split(":"))
            end = min(end, len(parts) - 1)
            frame[key] = sep.join(parts[start : end + 1])
            continue
        raise ValueError(f"Unknown schema entry type for '{key}': {type(value)}")
    return frame


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Optuna dataframe prefixes for easier downstream usage."""
    cleaned: list[str] = []
    for col in df.columns:
        if col.startswith("values_") or col.startswith("params_"):
            cleaned.append(col[7:])
        elif col.startswith("user_attrs_"):
            cleaned.append(col[11:])
        elif col.startswith("++") or col.startswith("~"):
            cleaned.append(col[2:])
        else:
            cleaned.append(col)
    df.columns = cleaned
    return df


def _resolve_selected_study_names(
    storage_uri: Any,
    study_name: str | None,
    study_names: list[str] | None,
) -> list[str]:
    selected_names: list[str] = []
    if study_name is not None and str(study_name).strip() != "":
        selected_names.append(str(study_name).strip())
    if study_names is not None:
        selected_names.extend(
            str(name).strip() for name in study_names if str(name).strip() != ""
        )

    if len(selected_names) == 0:
        summaries = optuna.study.get_all_study_summaries(storage=storage_uri)
        assert len(summaries) > 0, f"No studies found in {storage_uri}"
        selected_names = [
            str(getattr(summary, "study_name", getattr(summary, "name", ""))).strip()
            for summary in summaries
        ]
        selected_names = [name for name in selected_names if name != ""]

    seen: set[str] = set()
    unique_names: list[str] = []
    for name in selected_names:
        if name not in seen:
            unique_names.append(name)
            seen.add(name)
    return unique_names


def _apply_trial_filters(
    merged: pd.DataFrame,
    trial_numbers: Iterable[int] | None,
    trial_number_range: tuple[int, int] | list[int] | None,
    trial_states: Iterable[str] | None,
) -> pd.DataFrame:
    if trial_numbers is not None and "number" in merged.columns:
        allowed = {int(number) for number in trial_numbers}
        merged = merged[merged["number"].isin(allowed)]

    if trial_number_range is not None and "number" in merged.columns:
        assert len(trial_number_range) == 2, "trial_number_range must be [start, end]"
        start, end = int(trial_number_range[0]), int(trial_number_range[1])
        low, high = (start, end) if start <= end else (end, start)
        merged = merged[(merged["number"] >= low) & (merged["number"] <= high)]

    if trial_states is not None and "state" in merged.columns:
        allowed_states = {str(state).strip().upper() for state in trial_states}
        merged = merged[merged["state"].astype(str).str.upper().isin(allowed_states)]
    return merged


def _apply_sort_slice_and_pagination(
    merged: pd.DataFrame,
    sort_by: str | list[str] | None,
    ascending: bool,
    row_slice: slice | tuple[int | None, int | None] | str | None,
    offset: int,
    limit: int | None,
) -> pd.DataFrame:
    if sort_by is not None:
        sort_cols = [sort_by] if isinstance(sort_by, str) else list(sort_by)
        sort_cols = [col for col in sort_cols if col in merged.columns]
        if len(sort_cols) > 0:
            merged = merged.sort_values(sort_cols, ascending=ascending)

    if row_slice is not None:
        if isinstance(row_slice, str):
            parts = row_slice.split(":")
            assert len(parts) == 2, "row_slice string must be in 'start:end' format"
            start = int(parts[0]) if parts[0] != "" else None
            end = int(parts[1]) if parts[1] != "" else None
            merged = merged.iloc[slice(start, end)]
        elif isinstance(row_slice, tuple):
            assert len(row_slice) == 2, "row_slice tuple must be (start, end)"
            merged = merged.iloc[slice(row_slice[0], row_slice[1])]
        else:
            merged = merged.iloc[row_slice]

    offset = max(int(offset), 0)
    if limit is not None:
        limit = max(int(limit), 0)
        merged = merged.iloc[offset : offset + limit]
    elif offset > 0:
        merged = merged.iloc[offset:]
    return merged


def _apply_column_selection(
    merged: pd.DataFrame,
    columns: list[str] | None,
    include_columns: list[str] | None,
    exclude_columns: list[str] | None,
) -> pd.DataFrame:
    if columns is not None:
        keep_cols = [col for col in columns if col in merged.columns]
        return merged[keep_cols]

    if include_columns is not None:
        include_cols = [col for col in include_columns if col in merged.columns]
        merged = merged[include_cols]
    if exclude_columns is not None:
        drop_cols = [col for col in exclude_columns if col in merged.columns]
        if len(drop_cols) > 0:
            merged = merged.drop(columns=drop_cols)
    return merged


def load_optuna_studies_dataframe(
    storage: Any = None,
    study_name: str | None = None,
    schema: Union[dict[str, Any], str, None] = None,
    study_names: list[str] | None = None,
    columns: list[str] | None = None,
    include_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
    trial_numbers: Iterable[int] | None = None,
    trial_number_range: tuple[int, int] | list[int] | None = None,
    trial_states: Iterable[str] | None = None,
    row_slice: slice | tuple[int | None, int | None] | str | None = None,
    sort_by: str | list[str] | None = None,
    ascending: bool = True,
    offset: int = 0,
    limit: int | None = None,
) -> pd.DataFrame:
    """Query Optuna studies without SQL, across any Optuna-supported RDB storage.

    Supports multi-study selection, trial-level filtering, column projection,
    sort/slice, and pagination through Optuna's Python API only.
    """
    storage_uri = _normalize_optuna_storage(storage)
    unique_names = _resolve_selected_study_names(storage_uri, study_name, study_names)

    merged = pd.DataFrame()
    for resolved_name in unique_names:
        study = optuna.study.load_study(storage=storage_uri, study_name=resolved_name)
        frame = study.trials_dataframe()
        frame["study_name"] = resolved_name
        if schema is not None:
            meta_df = parse_study_name(study_name=resolved_name, schema=schema)
            frame = frame.merge(meta_df, how="cross")
        merged = pd.concat([merged, frame], ignore_index=True)

    merged = clean_column_names(merged)

    merged = _apply_trial_filters(
        merged,
        trial_numbers,
        trial_number_range,
        trial_states,
    )
    merged = _apply_sort_slice_and_pagination(
        merged,
        sort_by,
        ascending,
        row_slice,
        offset,
        limit,
    )
    merged = _apply_column_selection(merged, columns, include_columns, exclude_columns)

    return merged.reset_index(drop=True)


@dataclass
class OptunaStudyDumpCallback(Callback):
    """
    Optuna callback to dump study results to CSV after multirun.

    Args:
        storage (str): Optuna storage URI.
        study_name (str): Name of the Optuna study.
        metric_names (Union[str, ListConfig, list]): Metric names to track.
        directions (Union[str, ListConfig, list]): Optimization directions.
        output_file (str): Path to output CSV file.
    """

    def __init__(
        self,
        storage: str,
        study_name: str,
        metric_names: Union[str, ListConfig, list],
        directions: Union[str, ListConfig, list],
        output_file: str,
    ):
        """
        Initialize the OptunaStudyDumpCallback.

        Args:
            storage (str): Optuna storage URI.
            study_name (str): Name of the Optuna study.
            metric_names (Union[str, ListConfig, list]): Metric names to track.
            directions (Union[str, ListConfig, list]): Optimization directions.
            output_file (str): Path to output CSV file.
        """
        self.storage = storage
        self.study_name = study_name
        # Make sure the folder exists
        db_file = self.storage.split("///")[-1]
        db_folder = Path(db_file).parent
        Path(db_folder).mkdir(parents=True, exist_ok=True)
        # Set metric names
        if isinstance(metric_names, ListConfig):
            resolved_metric_names = OmegaConf.to_container(
                metric_names,
                resolve=True,
            )
        elif isinstance(metric_names, list):
            resolved_metric_names = metric_names
        else:
            resolved_metric_names = [metric_names]
        self.metric_names = [
            str(item) for item in cast(list[Any], resolved_metric_names)
        ]
        # Set direction
        if isinstance(directions, ListConfig):
            resolved_directions = OmegaConf.to_container(directions, resolve=True)
        elif isinstance(directions, list):
            resolved_directions = directions
        else:
            resolved_directions = [directions]
        self.directions = [str(item) for item in cast(list[Any], resolved_directions)]
        self.output_file = output_file
        super().__init__()

    def on_multirun_start(self, config: DictConfig, **kwargs) -> None:
        """
        Called at the start of a multirun. Deletes existing study and creates a new one.

        Args:
            config (DictConfig): Hydra config.
            **kwargs: Additional keyword arguments.
        """
        try:
            _study = optuna.load_study(
                study_name=self.study_name,
                storage=self.storage,
            )
            _ = _study
            optuna.delete_study(study_name=self.study_name, storage=self.storage)
        except Exception:
            pass
        if len(self.directions) == 1:
            direction = self.directions[0]
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                direction=direction,
                load_if_exists=True,
            )
        else:
            directions = self.directions
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                directions=directions,
                load_if_exists=True,
            )

        if hasattr(study, "set_metric_names"):
            study.set_metric_names(self.metric_names)
        else:
            print("Cannot set metric names")

    def on_multirun_end(self, config: DictConfig, **kwargs) -> None:
        """
        Called at the end of a multirun. Saves the study trials to CSV.

        Args:
            config (DictConfig): Hydra config.
            **kwargs: Additional keyword arguments.
        """
        df = load_optuna_studies_dataframe(
            storage=self.storage,
            study_name=self.study_name,
        )
        df.to_csv(self.output_file, index=False)
        print(f"Saved to {self.output_file}")
