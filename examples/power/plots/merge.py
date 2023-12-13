import pandas as pd
from pathlib import Path
import logging
import yaml
from deckard.layers.compile import save_results

logger = logging.getLogger(__name__)


__all__ = ["merge_run"]


def merge_run(
    big_dir,
    little_dir,
    output_file="merged",
    data_file="raw.csv",
    little_dir_data_file=None,
    fillna={
        "model.art.preprocessor.name": "art.defences.preprocessor.FeatureSqueezing",
        "model.art.preprocessor.params": "bit_depth",
    },
):
    """
    The function `merge_run` merges two CSV files, one from a big directory and one from a little
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
    big = pd.read_csv(Path(big_dir) / data_file)
    if little_dir_data_file is not None:
        small = pd.read_csv(Path(little_dir) / little_dir_data_file)
    else:
        small = pd.read_csv(Path(little_dir) / data_file)
    logger.info(f"Shape of big: {big.shape}")
    logger.info(f"Shape of small: {small.shape}")
    merged = pd.merge(big, small, how="outer")
    for k, v in fillna.items():
        merged[k] = merged[k].fillna(v)
    logger.info(f"Shape of merged: {merged.shape}")
    logger.info(f"Saving merged to {data_file}.")
    results_folder = Path(output_file).parent
    results_file = Path(output_file).name
    results_folder.mkdir(parents=True, exist_ok=True)
    saved_path = save_results(
        merged, results_file=results_file, results_folder=results_folder
    )
    assert Path(saved_path).exists(), f"Saved path {saved_path} does not exist."
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--big_dir", type=str, help="Directory of the big run", required=True
    )
    parser.add_argument(
        "--little_dir", type=str, help="Directory of the small run", required=False
    )
    parser.add_argument(
        "--data_file", type=str, help="Name of the data file", required=True
    )
    parser.add_argument(
        "--output_file", type=str, help="Name of the output file", default="merged.csv"
    )
    parser.add_argument(
        "--output_folder", type=str, help="Name of the output folder", required=False
    )
    parser.add_argument(
        "--little_dir_data_file",
        type=str,
        help="Name of the output folder",
        required=False,
    )
    parser.add_argument(
        "--config", type=str, help="Name of the output folder", required=False
    )
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, "r") as stream:
            fillna = yaml.safe_load(stream).get("fillna", {})
    else:
        fillna = {}
    if args.output_folder is None:
        args.output_folder = Path().cwd()
    output_file = Path(args.output_folder) / args.output_file
    merge_run(
        args.big_dir,
        args.little_dir,
        data_file=args.data_file,
        little_dir_data_file=args.little_dir_data_file,
        fillna=fillna,
        output_file=output_file,
    )
