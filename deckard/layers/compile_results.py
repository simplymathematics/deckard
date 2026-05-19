import logging
import warnings
from pathlib import Path
from typing import Any, Union

import optuna
import pandas as pd
import yaml

from ..utils import create_parser_from_function, save_data

# suppress future warning
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def parse_study_name(
    study_name: str,
    schema: Union[dict[str, Any], str, None] = None,
) -> pd.DataFrame:
    """Parse Optuna study metadata from a study name.

    Parameters
    ----------
    study_name : str
        Study name to parse.
    schema : dict[str, Any] | str
        Mapping of output column names to indices/ranges, or a YAML file path
        containing a ``schema`` mapping.

    Returns
    -------
    pd.DataFrame
        Single-row dataframe with parsed metadata columns.
    """
    if schema is None:
        schema = {}
    if isinstance(schema, str):
        assert Path(
            schema,
        ).exists(), f"Schema must be a dictionary or a file path. Got type {type(schema)} and the filepath does not exist."
        with open(schema, "r") as f:
            conf = yaml.safe_load(f)
            schema = conf.pop("schema", conf)
    schema_copy = schema.copy()
    sep = schema_copy.pop("sep", "_")
    name_list = study_name.split(sep)
    meta_df = pd.DataFrame()
    other_keys = ["sep"]
    for k, v in schema_copy.items():
        if k in other_keys:
            continue
        elif isinstance(v, int):
            try:
                meta_df[k] = [name_list[v]]
            except IndexError as e:
                logger.debug(e)
                meta_df[k] = None
        elif isinstance(v, str):
            assert (
                len(v.split(":")) == 2
            ), f"Schema value should either be a an integer or a an inclusive range in the form first:last. Got {v}"
            start, end = map(int, v.split(":"))
            end = min(end, len(name_list) - 1)
            meta_df[k] = sep.join(name_list[start : end + 1])  # NOQA E203
        else:
            raise ValueError("Unknown value type for schema entry:", type(v))
    return meta_df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Optuna trial dataframe column prefixes for reporting."""
    cols = df.columns
    clean_cols = []
    for col in cols:
        if col.startswith("values_") or col.startswith("params_"):
            col = col[7:]
            clean_cols.append(col)
        elif col.startswith("user_attrs_"):
            col = col[11:]
            clean_cols.append(col)
        elif col.startswith("++") or col.startswith("~"):
            col = col[2:]
            clean_cols.append(col)
        else:
            clean_cols.append(col)
    df.columns = clean_cols
    return df


def parse_studies(
    optuna_db: str,
    schema: Union[str, dict[str, Any]],
) -> pd.DataFrame:
    """Load and merge all studies from an Optuna storage into one dataframe."""
    studies = optuna.study.get_all_study_summaries(storage=optuna_db)
    assert len(studies) > 0, f"No studies found in {optuna_db}"
    df = pd.DataFrame()
    for summary in studies:
        name = getattr(summary, "study_name", getattr(summary, "name", None))
        assert name is not None, "Study summary did not expose a study name"
        study = optuna.study.load_study(storage=optuna_db, study_name=name)
        tmp_df = study.trials_dataframe()
        meta_df = parse_study_name(study_name=name, schema=schema)
        tmp_df = tmp_df.merge(meta_df, how="cross")
        df = pd.concat([df, tmp_df], ignore_index=True)
    df = clean_column_names(df)
    return df


def compile_results_main(
    output_file: str,
    optuna_db: str,
    schema: Union[str, dict[str, Any], None] = None,
) -> None:
    """Compile Optuna studies into a single tabular results file.

    Parameters
    ----------
    output_file : str
        Destination path for compiled results.
    optuna_db : str
        Optuna storage URI.
    schema : str | dict[str, Any] | None, optional
        Optional schema map (or file path) for parsing study names.
    """
    # Check if schema is string or dict
    if schema is not None:
        schema_yaml = yaml.safe_dump(schema)
        if isinstance(schema_yaml, dict):
            pass
        else:
            schema = str(Path(schema).absolute())
            assert Path(
                schema,
            ).is_file(), f"Schema must be a dictionary or a valid file. Got {schema.absolute()}."
    else:
        schema = {}
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    optuna_db = optuna_db
    output_file = str(output_path)

    df = parse_studies(optuna_db=optuna_db, schema=schema)
    save_data(data=df, filepath=output_file)


compile_results_parser = create_parser_from_function(compile_results_main)

if __name__ == "__main__":
    args = compile_results_parser.parse_args()
    compile_results_main(**vars(args))
