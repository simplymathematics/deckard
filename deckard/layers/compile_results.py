import logging
import warnings
from pathlib import Path
from typing import Any, Union

import optuna
import pandas as pd
import yaml

from ..optuna_callback import (
    clean_column_names as _clean_optuna_columns,
    load_optuna_studies_dataframe,
    parse_study_name,
)
from ..utils import create_parser_from_function, save_data

# suppress future warning
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def parse_study_name(
    study_name: str,
    schema: Union[dict[str, Any], str, None] = None,
) -> pd.DataFrame:
    """Parse Optuna study metadata from a study name.

    Args:
        study_name: Study name to parse.
        schema: Mapping of output column names to indices or ranges, or a YAML
            file path containing a ``schema`` mapping.

    Returns:
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
    return _clean_optuna_columns(df)


def parse_studies(
    optuna_db: str,
    schema: Union[str, dict[str, Any]],
) -> pd.DataFrame:
    """Load and merge all studies from an Optuna storage into one dataframe."""
    return load_optuna_studies_dataframe(
        storage=optuna_db,
        study_name=None,
        schema=schema,
    )


def compile_results_main(
    output_file: str,
    optuna_db: str,
    schema: Union[str, dict[str, Any], None] = None,
) -> None:
    """Compile Optuna studies into a single tabular results file.

    Args:
        output_file: Destination path for compiled results.
        optuna_db: Optuna storage URI.
        schema: Optional schema map, or a file path, for parsing study names.
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
