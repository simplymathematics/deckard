import pandas as pd
from pathlib import Path
import logging
import yaml
import argparse
from deckard.layers.compile import save_results

logger = logging.getLogger(__name__)


__all__ = ["merge_csv", "merge_main", "merge_parser"]


def merge_csv(
    big_dir,
    little_dir,
    output_file="merged",
    data_file="raw.csv",
    little_dir_data_file=None,
    fillna={},
    how="outer",
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
    `big` and `small` dataframes. If `little_dir_data_file` is not provided, then the `data_file` from
    the `big` directory will be used for both dataframes. Defaults to raw.csv
      little_dir_data_file: The parameter `little_dir_data_file` is an optional argument that specifies
    the name of the data file in the `little_dir` directory. If this argument is provided, the function
    will read the data from the specified file in the `little_dir` directory. If this argument is not
    provided, the

    Returns:
      None.
    """
    if Path(Path(big_dir) / data_file).exists() is False:
        big = pd.DataFrame()
    else:
        big = pd.read_csv(Path(big_dir) / data_file, index_col=0)
        assert "name" in big
    if little_dir is None:
        little_dir = big_dir
    if little_dir_data_file is not None:
        small = pd.read_csv(Path(little_dir) / little_dir_data_file, index_col=0)
        assert "name" in small
    else:
        small = pd.read_csv(Path(little_dir) / data_file)
    logger.info(f"Shape of big: {big.shape}")
    logger.info(f"Shape of small: {small.shape}")
    try:
        merged = pd.merge(big, small, how=how, **kwargs)
    except pd.errors.MergeError as e:
        logger.error(f"Merge error: {e}")
        logger.error(f"Big columns: {big.columns}")
        logger.error(f"Small columns: {small.columns}")
        raise e
    for k, v in fillna.items():
        if k in merged.columns:
            merged[k] = merged[k].fillna(v)
        else:
            merged[k] = v
    logger.info(f"Shape of merged: {merged.shape}")
    logger.info(f"Saving merged to {data_file}.")
    results_folder = Path(output_file).parent
    results_file = Path(output_file).name
    results_folder.mkdir(parents=True, exist_ok=True)
    merged["id"] = merged["name"]
    saved_path = save_results(
        merged,
        results_file=results_file,
        results_folder=results_folder,
    )
    assert Path(saved_path).exists(), f"Saved path {saved_path} does not exist."
    return None


def merge_main(args):
    if args.config is not None:
        with open(args.config, "r") as stream:
            fillna = yaml.safe_load(stream).get("fillna", {})
    else:
        fillna = {}
    if args.output_folder is None:
        args.output_folder = Path().cwd()
    output_file = Path(args.output_folder) / args.output_file
    if isinstance(args.little_dir_data_file, list):
        for little in args.little_dir_data_file:
            merge_csv(
                args.big_dir,
                args.little_dir,
                data_file=args.data_file,
                little_dir_data_file=little,
                fillna=fillna,
                output_file=output_file,
            )
            args.big_dir = Path(args.output_folder)
            args.data_file = Path(args.output_file).name
            print(f"Big dir: {args.big_dir}")
            print(f"Data file: {args.data_file}")
            print(f"Output file: {args.output_file}")
    else:
        merge_csv(
            args.big_dir,
            args.little_dir,
            data_file=args.data_file,
            little_dir_data_file=args.little_dir_data_file,
            fillna=fillna,
            output_file=output_file,
            how="outer",
        )


merge_parser = argparse.ArgumentParser()
merge_parser.add_argument(
    "--big_dir",
    type=str,
    help="Directory of the big run",
    required=True,
)
merge_parser.add_argument(
    "--little_dir",
    type=str,
    help="Directory of the small run",
    required=False,
)
merge_parser.add_argument(
    "--data_file",
    type=str,
    help="Name of the data file",
    required=True,
)
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
    "--little_dir_data_file",
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

if __name__ == "__main__":
    args = merge_parser.parse_args()
    merge_main(args)
