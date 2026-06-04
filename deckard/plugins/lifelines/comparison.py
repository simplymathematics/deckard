from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...utils import update_data


def write_aft_comparison_table(table: pd.DataFrame, folder: str) -> Path:
    """Write or update the canonical lifelines comparison table on disk.

    If the CSV already exists, rows whose ``model`` key appears in ``table``
    are replaced and any previously unseen models are preserved.  This allows
    repeated runs to accumulate results across multiple model types without
    losing data from prior experiments.

    Args:
        table: Dataframe with at least a ``model`` column and metric columns.
        folder: Destination folder; the file is always named
            ``aft_comparison.csv`` inside this folder.

    Returns:
        Absolute path to the written CSV file.
    """
    csv_path = Path(folder) / "aft_comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if "model" in table.columns:
        update_data(table, filepath=str(csv_path), key="model")
    else:
        update_data(table, filepath=str(csv_path))
    return csv_path
