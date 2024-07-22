import pandas as pd
from pathlib import Path
import logging
import yaml
import argparse
from numpy import nan as np_nan
from deckard.layers.compile import save_results

logger = logging.getLogger(__name__)


__all__ = ["merge_csv", "merge_main", "merge_parser"]


def merge_csv(
    output_file: str,
    smaller_file: list = None,
    fillna: dict = {},
    metadata: list = [],
    subset_metadata: list = [],
    how: str = "outer",
    **kwargs,
):
    """
    The function `merge_csv` merges two CSV files, one from a big directory and one from a little
    directory, and saves the merged file.

    Args:
      big_dir: The `big_dir` parameter is the directory path where the dataset to be merged into is located. This
    dataset is assumed to have a file named "raw.csv" which will be read.
      little_dir: The `little_dir` parameter is the directory path where the smaller dataset is located.
      data_file: The `data_file` parameter is the name of the CSV file that will be used for both the
    `big` and `small` dataframes. If `smaller_file` is not provided, then the `data_file` from
    the `big` directory will be used for both dataframes. Defaults to raw.csv
      smaller_file: The parameter `smaller_file` is an optional argument that specifies
    the name of the data file in the `little_dir` directory. If this argument is provided, the function
    will read the data from the specified file in the `little_dir` directory. If this argument is not
    provided, the

    Returns:
      None.
    """
    if Path(output_file).exists():
        big = pd.read_csv(Path(output_file), index_col=0, dtype=str)
    else:
        big = pd.DataFrame()
    # Cast all columns to strings:
    big = big.replace(np_nan, "").astype(str)
    # Strip whitepsace around entries:
    big = big.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    # strip whitespace around column names
    big.columns = big.columns.astype(str).str.strip()
    if "id" not in big:
        big["id"] = None
    if smaller_file is not None:
        small = pd.read_csv(Path(smaller_file), index_col=0, dtype=str)
        if "id" not in small:
            small["id"] = Path(smaller_file).stem
    # Cast all columns to strings:
    small = small.replace(np_nan, "").astype(str)
    # Strip whitepsace around entries:
    small = small.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    # strip whitespace around column names
    small.columns = small.columns.str.strip()
    logger.info(f"Shape of big: {big.shape}")
    logger.info(f"Shape of small: {small.shape}")
    try:
        merged = pd.merge(big, small, how=how, **kwargs)
    except (
        pd.errors.MergeError
    ) as e:  # will raise if no common columns to merge on, so just concat
        if "No common columns to perform merge on" in str(e):
            merged = pd.concat([big, small], axis=0)
        else:
            raise e
    for k, v in fillna.items():
        if k in merged.columns:
            merged[k] = merged[k].fillna(v)
        else:
            merged[k] = v
    logger.info(f"Shape of merged: {merged.shape}")
    logger.info(f"Saving merged to {output_file}.")
    results_folder = Path(output_file).parent
    results_file = Path(output_file).name
    results_folder.mkdir(parents=True, exist_ok=True)
    # Add metadata if it exists
    if len(metadata) > 0:
        merged = add_metadata(merged, metadata)
    # Add subset metadata if it exists
    if len(subset_metadata) > 0:
        merged = add_subset_metadata(merged, subset_metadata)
    saved_path = save_results(
        merged,
        results_file=results_file,
        results_folder=results_folder,
    )

    assert Path(saved_path).exists(), f"Saved path {saved_path} does not exist."
    return None


def parse_cleaning_config(config_file, metadata_file=None, subset_metadata_file=None):
    dict_ = {}
    if config_file is not None:
        with open(config_file, "r") as stream:
            dict_ = yaml.safe_load(stream)
        fillna = dict_.get("fillna", dict_)
    else:
        fillna = {}
    dict_["fillna"] = fillna
    if metadata_file is not None:
        with open(metadata_file, "r") as stream:
            metadata = yaml.safe_load(stream)
        metadata = metadata.get("metadata", metadata)
    elif "metadata" in dict_:
        metadata = dict_["metadata"]
    else:
        metadata = {}
    dict_["metadata"] = metadata
    if subset_metadata_file is not None:
        with open(subset_metadata_file, "r") as stream:
            subset_metadata = yaml.safe_load(stream)
        subset_metadata = subset_metadata.get("subset_metadata", subset_metadata)
    elif "subset_metadata" in dict_:
        subset_metadata = dict_["subset_metadata"]
    else:
        subset_metadata = {}
    dict_["subset_metadata"] = subset_metadata
    return dict_


def merge_main(args):
    config = parse_cleaning_config(args.config, args.metadata, args.subset_metadata)
    if args.output_folder is None:
        args.output_folder = Path().cwd()
    output_file = Path(args.output_folder) / args.output_file
    if isinstance(args.smaller_file, list):
        for little in args.smaller_file:
            merge_csv(
                smaller_file=little,
                fillna=config.get("fillna", {}),
                metadata=config.get("metadata", {}),
                subset_metadata=config.get("subset_metadata", {}),
                output_file=output_file,
                how=args.how,
            )
    else:
        merge_csv(
            smaller_file=args.smaller_file,
            fillna=config.get("fillna", {}),
            metadata=config.get("metadata", {}),
            subset_metadata=config.get("subset_metadata", {}),
            output_file=output_file,
            how=args.how,
        )
    return None


def add_metadata(df, metadata_dict={}):
    """
    The function `add_metadata` adds metadata to the dataframe.

    Args:
      df: The `df` parameter is the dataframe to which metadata will be added.
      metadata_dict: The `metadata_dict` parameter is a dictionary containing the metadata to be added to the dataframe.

    Returns:
      df: The dataframe with the added metadata.
    """
    for thing in metadata_dict:
        k = thing.split(":")[0]
        v = thing.split(":")[1]
        df[k] = v
    return df


def add_subset_metadata(df, metadata_list=[]):
    """
    The function `add_subset_metadata` adds metadata to the dataframe.

    Args:
      df: The `df` parameter is the dataframe to which metadata will be added.
      metadata_dict: The `metadata_dict` parameter is a dictionary containing the metadata to be added to the dataframe.

    Returns:
      df: The dataframe with the added metadata.
    """
    for thing in metadata_list:
        subset = thing.get("subset", ValueError("subset not found in entry."))
        subset_col = subset.split(" ")[0]
        subset_op = subset.split(" ")[1]
        subset_val = subset.split(" ")[2]
        filter = f"df[{subset_col}] {subset_op} {subset_val}"
        key = thing.get("key", ValueError("key not found in entry."))
        value = thing.get("value", ValueError("value not found in entry."))
        subset_df = df[subset_col].where(filter)
        subset_df[key] = value
        df.update(subset_df)
    return df


merge_parser = argparse.ArgumentParser()
merge_parser.add_argument(
    "--output_file",
    type=str,
    help="Name of the output file",
    default="merged.csv",
)
merge_parser.add_argument(
    "--output_folder",
    type=str,
    help="Name of the output folder",
    required=False,
)
merge_parser.add_argument(
    "--smaller_file",
    type=str,
    help="Name(s) of the files to merge into the big file.",
    required=False,
    nargs="*",
)
merge_parser.add_argument(
    "--config",
    type=str,
    help="Name of file containing a 'fillna' config dictionary.",
    required=False,
)
merge_parser.add_argument(
    "--metadata",
    type=str,
    help="Name of file containing a 'metadata' dictionary.",
    required=False,
    # set default to --config
    default=None,
)
merge_parser.add_argument(
    "--subset_metadata",
    type=str,
    help="Name of file containing a 'subset_metadata' dictionary.",
    required=False,
    default=None,
)
merge_parser.add_argument(
    "--how",
    type=str,
    help="Type of merge to perform. Default is 'outer'.",
    default="outer",
)

if __name__ == "__main__":
    args = merge_parser.parse_args()
    merge_main(args)
