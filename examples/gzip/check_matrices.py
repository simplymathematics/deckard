from pathlib import Path
import numpy as np
import logging
import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

logger = logging.getLogger(__name__)
sns.set_theme(context="paper", style="whitegrid", font="Times New Roman", font_scale=2)


def check_symmetry_identity(matrix: np.ndarray) -> bool:
    logger.debug("Checking symmetry and identity.")
    new_matrix = matrix - matrix.T
    counts = np.count_nonzero(new_matrix)
    logger.debug(f"Number of non-symmetric elements: {counts}")
    return counts


def check_non_negative_matrix(matrix: np.ndarray) -> bool:
    logger.debug("Checking non-negative matrix.")
    new_matrix = matrix - np.abs(matrix)
    counts = np.count_nonzero(new_matrix)
    logger.debug(f"Number of negative elements: {counts}")
    return counts


def check_that_some_elements_are_not_zero(matrix: np.ndarray, file) -> bool:
    logger.debug("Checking that all elements are not zero.")
    counts = np.count_nonzero(matrix)
    size = matrix.shape[0] * matrix.shape[1]
    logger.debug(f"Number of zero elements: {size - counts}")
    if counts == 0:
        logger.error(f"Matrix {file} has all zero elements.")


def check_triangle_inequality(matrix: np.ndarray, max_samples=-1) -> bool:
    logger.debug("Checking triangle inequality.")
    n = matrix.shape[0]
    if max_samples == -1:
        max_samples = n * n
    # Assert that the matrix is square
    assert matrix.shape[0] == matrix.shape[1], "Matrix is not square."
    assert (
        max_samples <= n * n
    ), "Number of samples is greater than the number of elements in the matrix."
    # Sample the indices
    indices = np.random.choice(n, size=(max_samples, 3))
    counts = 0
    for i, j, k in tqdm(
        indices,
        desc="Checking triangle inequality",
        position=1,
        leave=False,
    ):
        if i != j and i != k and j != k:
            if matrix[i, j] > matrix[i, k] + matrix[k, j]:
                counts += 1
    logger.debug(f"Number of triangle inequality violations: {counts}")
    return counts


def check_diagonal_matrix(matrix: np.ndarray) -> bool:
    logger.debug("Checking diagonal matrix.")
    # Count the number of non-zero elements in the diagonal
    counts = np.count_nonzero(np.diag(matrix))
    logger.debug(f"Number of non-zero elements in the diagonal: {counts}")
    return counts


def check_zero_identity(matrix: np.ndarray) -> bool:
    logger.debug("Checking non-zero except diagonal.")
    diagonal_failures = check_diagonal_matrix(matrix)
    non_diagonal_successes = np.count_nonzero(matrix - np.diag(np.diag(matrix)))
    total_size = matrix.shape[0] * matrix.shape[1]
    total_size_wo_diagonal = total_size - matrix.shape[0]
    non_diagonal_failures = total_size_wo_diagonal - non_diagonal_successes
    logger.debug(
        f"Number of non-zero elements except diagonal: {non_diagonal_failures}",
    )
    return diagonal_failures + non_diagonal_failures


def count_failures(matrix: np.ndarray) -> dict:
    failures = {
        "symmetry": 0,
        "non_negative": 0,
        "triangle_inequality": 0,
        "zero_identity": 0,
    }

    failures["zero_identity"] = check_zero_identity(matrix)
    failures["symmetry"] = check_symmetry_identity(matrix)
    failures["non_negative"] = check_non_negative_matrix(matrix)
    failures["triangle_inequality"] = check_triangle_inequality(matrix)
    total = matrix.shape[0] * matrix.shape[1]
    failures["symmetry"] = (
        failures["symmetry"] / total / 2
    )  # Divide by 2 because we assume symmetry
    failures["non_negative"] = (
        failures["non_negative"] / total
    )  # Divide by the total number of elements
    failures["triangle_inequality"] = failures["triangle_inequality"] / total
    failures["zero_identity"] = failures["zero_identity"] / total
    return failures


def extract_metadata_from_filename(file: Path) -> dict:
    parents = [p.name for p in file.parents]
    wd = parents[-2]
    dataset = parents[-3]
    metric = parents[-5]
    algorithm = parents[-6]
    filename = file.name.split(".")[0]
    logger.info(f"Filename: {filename}")
    logger.info(f"working directory: {wd}")
    logger.info(f"dataset: {dataset}")
    logger.info(f"metric: {metric}")
    logger.info(f"Algorithm: {algorithm}")
    test_or_train, training_samples, test_samples, random_state = filename.split("-")
    return {
        "working_directory": wd,
        "dataset": dataset,
        "metric": metric,
        "training_samples": training_samples,
        "test_samples": test_samples,
        "random_state": random_state,
        "algorithm": algorithm,
        "mode": test_or_train,
    }


def check_failures_for_all_files(files: list, results_file=None) -> dict:
    result_list = []
    df = pd.DataFrame()
    for file in tqdm(files, desc="Checking matrices", position=0, leave=True):
        result_dict = get_results_for_file(file)
        result_list.append(result_dict)
    df = pd.DataFrame(result_list)
    if results_file:
        if Path(results_file).exists():
            Path(results_file).unlink()
        df.to_csv(results_file)
    return df


def get_results_for_file(file):
    result_dict = {}
    metadata = extract_metadata_from_filename(file)
    logger.debug(f"Metadata: {metadata}")
    matrix = np.load(file)["X"]
    assert matrix.shape[0] == matrix.shape[1], "Matrix is not square."
    check_that_some_elements_are_not_zero(matrix, file)
    logger.debug(f"Matrix shape: {matrix.shape}")
    failures = count_failures(matrix)
    result_dict.update(failures)
    result_dict.update(metadata)
    return result_dict


def plot_results(df: pd.DataFrame, results_plot: Path):
    value_vars = ["symmetry", "non_negative", "triangle_inequality", "zero_identity"]

    id_vars = [c for c in df.columns if c not in value_vars]
    df = pd.melt(df, id_vars=id_vars, value_vars=value_vars)
    metric_dict = {
        "brotli": "Brotli",
        "gzip": "GZIP",
        "bz2": "BZ2",
        "ratio": "Ratio",
        "levenshtein": "Levenshtein",
        "hamming": "Hamming",
    }
    df["metric"] = df["metric"].map(metric_dict)
    results_folder = results_plot.parent
    plot_csv = results_folder / "melted.csv"
    df.to_csv(plot_csv)
    variable_dict = {
        "symmetry": "Symmetry",
        "non_negative": "Non-negative",
        "triangle_inequality": "Triangle inequality",
        "zero_identity": "Zero identity",
    }
    df["variable"] = df["variable"].map(variable_dict)
    # Create two plots on the same figure
    fig, ax = plt.subplots(1, 2, figsize=(20, 12))
    g = sns.catplot(
        x="variable",
        y="value",
        data=df,
        hue="metric",
        col="dataset",
        kind="bar",
        row="algorithm",
        sharex=False,
        sharey=True,
    )
    # ylabel
    g.set_ylabels("Probability of Metric Property Violation")
    # title
    g.set_titles("{col_name} - {row_name}")
    # Set legend name to metric
    g._legend.set_title("Metric")
    # rotate x labels
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    g.set(yscale="log")
    # Tight layout
    g.tight_layout()
    g.savefig(results_plot)
    # save the plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check matrices for metric space properties.",
    )
    parser.add_argument(
        "--working_dir",
        type=str,
        help="Directory containing the matrices to check.",
        required=True,
    )
    parser.add_argument(
        "--file_regex",
        type=str,
        help="Regex to match the files to check.",
        required=True,
    )
    parser.add_argument(
        "--results_file",
        type=str,
        help="Name of the output file.",
        default="failures.csv",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="Name of the log file.",
        default="check_matrices.log",
    )
    parser.add_argument(
        "--results_plot",
        type=str,
        help="Name of the plot file.",
        default="failures.pdf",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        help="Name of the results folder.",
        default="check_matrices",
    )
    args = parser.parse_args()
    # Make the directory if it doesn't exist
    Path(args.results_folder).mkdir(parents=True, exist_ok=True)
    # Set up logging
    log_file = Path(args.results_folder) / Path(args.log_file)
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    # Get all the files
    files = list(Path(args.working_dir).glob(args.file_regex))
    logger.debug(f"Found {len(files)} files to check.")
    # Check the files
    # Remove the old results file
    df = check_failures_for_all_files(
        files,
        results_file=Path(args.results_folder) / Path(args.results_file),
    )

    # Read the results file
    df = pd.read_csv(Path(args.results_folder) / Path(args.results_file))
    # Plot the results
    plot_results(df, results_plot=Path(args.results_folder) / Path(args.results_plot))
