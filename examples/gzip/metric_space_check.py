import random
import string
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Literal
from itertools import product, combinations
import logging
import argparse

import seaborn as sns
import pandas as pd

# Parallelize the for loop using joblib
from joblib import Parallel, delayed

from gzip_classifier import compressors

logger = logging.getLogger(__name__)

sns.set_theme(context="paper", style="whitegrid", font="Times New Roman", font_scale=2)


def old_ncd(
    x1,
    x2,
    cx1=None,
    cx2=None,
    method: Literal["gzip", "lzma", "bz2", "zstd", "pkl", "brotli", None] = "gzip",
) -> float:
    """
    Calculate the normalized compression distance between two objects treated as strings.
    Args:
        x1 (str): The first object
        x2 (str): The second object
    Returns:
        float: The normalized compression distance between x1 and x2
    """

    metric_len = (
        compressors[method] if method in compressors.keys() else compressors["gzip"]
    )
    Cx1 = metric_len(x1) if cx1 is None else cx1
    Cx2 = metric_len(x2) if cx2 is None else cx2
    x1x2 = "".join([x1, x2])
    Cx1x2 = metric_len(x1x2)
    min_ = min(Cx1, Cx2)
    max_ = max(Cx1, Cx2)
    ncd = (Cx1x2 - min_) / max_
    return ncd


def modified_ncd(x, y, method="gzip"):
    x = str(x) if not isinstance(x, str) else x
    y = str(y) if not isinstance(y, str) else y
    if x == y:
        return 0
    x, y = sort_xy(x, y)
    return old_ncd(x, y, method=method)


def sorted_ncd(x, y, method="gzip"):
    x = str(x) if not isinstance(x, str) else x
    y = str(y) if not isinstance(y, str) else y
    if x == y:
        return 0
    x, y = sort_xy(x, y)
    return old_ncd(x, y, method=method)


def averaged_ncd(x, y, method="gzip"):
    x = str(x) if not isinstance(x, str) else x
    y = str(y) if not isinstance(y, str) else y
    ncd1 = old_ncd(x, y, method=method)
    ncd2 = old_ncd(y, x, method=method)
    return (ncd1 + ncd2) / 2


def sort_xy(x, y):
    x = str(x) if not isinstance(x, str) else x
    y = str(y) if not isinstance(y, str) else y
    lx = len(x)
    ly = len(y)
    if lx < ly:
        res = y, x
    elif lx == ly:
        if x > y:
            res = x, y
        else:
            res = y, x
    else:
        res = x, y
    return res


def unmodified_ncd(x, y, method="gzip"):
    return old_ncd(x, y, method=method)


def string_generator(
    size=6,
    alphabet_size=52,
):
    chars = (
        string.ascii_uppercase
        + string.ascii_lowercase
        + string.digits
        + string.punctuation
        + " "
    )
    chars = chars[:alphabet_size]
    chars = list(set(chars))
    chars.sort()
    return "".join(random.choice(chars) for _ in range(size))


def byte_generator(
    size=6,
    alphabet_size=256,
):
    return bytes([random.randint(0, alphabet_size) for _ in range(size)])


def check_triangle_inequality(x, y, z, xz, xy, yz):
    if xz > xy + yz:
        raise ValueError(
            f"Triangle Inequality failed for {x}, {y}, {z}. <x,z> = {xz} > <x,y> + <y,z> = {xy + yz}",
        )
    if yz > xy + xz:
        raise ValueError(
            f"Triangle Inequality failed for {x}, {y}, {z}. <y,z> = {yz} > <x,y> + <x,z> = {xy + xz}",
        )
    if xy > xz + yz:
        raise ValueError(
            f"Triangle Inequality failed for {x}, {y}, {z}. <x,y> = {xy} > <x,z> + <y,z> = {xz + yz}",
        )
    return 0


def check_symmetry(xy, xz, yz, yx, zx, zy):
    if xz != zx:
        raise ValueError(f"XZ: {xz} != {zx}")
    elif yz != zy:
        raise ValueError(f"YZ: {yz} != {zy}")
    elif xy != yx:
        raise ValueError(f"XY: {xy} != {yx}")
    else:
        return 0


def check_zero(x, y, z, xx, yy, zz, xy, xz, yz, yx, zx, zy):
    if xx != 0:
        raise ValueError(f"<x,x> = {xx}")
    elif yy != 0:
        raise ValueError(f"<y,y> = {yy}")
    elif zz != 0:
        raise ValueError(f"<z,z> = {zz}")

    # Checks that other inner products are 0 if and only if the elements are equal
    if yx == 0:
        if y != x:
            raise ValueError(f"<{y},{x}> = 0, but {y} != {x}")
    if xy == 0:
        if y != x:
            raise ValueError(f"<{x},{y}> = 0, but {x} != {y}")
    if zx == 0:
        if z != x:
            raise ValueError(f"<{z},{x}> = 0, but {z} != {x}")
    if xz == 0:
        if z != x:
            raise ValueError(f"<{x},{z}> = 0, but {x} != {z}")
    if yz == 0:
        if z != y:
            raise ValueError(f"<{y},{z}> = 0, but {y} != {z}")
    if zy == 0:
        if z != y:
            raise ValueError(f"<{z},{y}> = 0, but {z} != {y}")
    return 0


def check_positivity(xy, xz, yz, yx, zx, zy):
    if xz < 0:
        raise ValueError(f"<x,z> = {xz} < 0")
    if zx < 0:
        raise ValueError(f"<z,x> = {zx} < 0")
    if yz < 0:
        raise ValueError(f"<y,z> = {yz} < 0")
    if zy < 0:
        raise ValueError(f"<z,y> = {zy} < 0")
    if xy < 0:
        raise ValueError(f"<x,y> = {xy} < 0")
    if yx < 0:
        raise ValueError(f"<y,x> = {yx} < 0")
    return 0


def check_loop(
    samples=1000,
    sig_figs=10,
    max_string_size=1000,
    data="random",
    distance="unmodified_ncd",
    alphabet_size=52,
    metric="gzip",
):

    arg_list = []

    # Generate a list of arguments for the parallelized for loop
    for i in range(samples):
        x, y, z = get_data_triplet(max_string_size, data, alphabet_size, samples, i)
        arg_list += [(sig_figs, distance, metric, x, y, z)]

    # Parallelize the for loop using joblib and tqdm
    df = np.array(
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(count_metric_assumption_failures)(*args)
            for args in tqdm(
                arg_list,
                desc=f"Checking metric space assumptions for {distance} algorithm using the {metric} metric.",
                position=2,
                leave=False,
            )
        ),
    )  # 4 columns, 1 for each assumption,
    # Convert failures to percent
    df = df.sum(axis=0) / samples
    logger.info(f"Percent of examples where triangle inequality was violated: {df[0]}")
    logger.info(f"Percent of examples where symmetry was violated: {df[1]}")
    logger.info(f"Percent of examples where zero identity was violated: {df[2]}")
    logger.info(f"Percent of examples where positivity was violated: {df[3]}")
    logger.info(f"Shape of df is {df.shape}")
    return df


def get_data_triplet(max_string_size, data, alphabet_size, samples, i):

    # Choose the dataset
    if data == "combinations":
        combinations = find_all_combinations(
            max_alphabet_size=alphabet_size,
            max_string_size=max_string_size,
            reverse=True,
        )
        combinations = random.sample(combinations, samples)
    elif data in ["random", "alphabet", "bytes"]:
        pass
    else:
        raise NotImplementedError(
            "Only random strings and alphabet combinations are supported at the data.",
        )

    if data in ["random", "alphabet"]:
        x = string_generator(
            size=random.randint(1, max_string_size),
            alphabet_size=alphabet_size,
        )
        y = string_generator(
            size=random.randint(1, max_string_size),
            alphabet_size=alphabet_size,
        )
        z = string_generator(
            size=random.randint(1, max_string_size),
            alphabet_size=alphabet_size,
        )
    elif data in ["bytes"]:
        assert max_string_size <= 256, ValueError("Max string size is too large")
        x = byte_generator(
            size=random.randint(1, max_string_size),
            alphabet_size=alphabet_size,
        )
        y = byte_generator(
            size=random.randint(1, max_string_size),
            alphabet_size=alphabet_size,
        )
        z = byte_generator(
            size=random.randint(1, max_string_size),
            alphabet_size=alphabet_size,
        )
    elif data == "combinations":
        x, y, z = combinations[i]
    elif isinstance(data, str) and Path(data).exists():
        raise NotImplementedError
    return x, y, z


def count_metric_assumption_failures(sig_figs, distance, metric, x, y, z):
    # Choose the distance function
    dist = get_distance_function(distance)
    xx = dist(x, x, method=metric)
    yy = dist(y, y, method=metric)
    zz = dist(z, z, method=metric)
    xy = dist(x, y, method=metric)
    xz = dist(x, z, method=metric)
    yz = dist(y, z, method=metric)
    yx = dist(y, x, method=metric)
    zx = dist(z, x, method=metric)
    zy = dist(z, y, method=metric)
    xx = round(xx, sig_figs)
    yy = round(yy, sig_figs)
    zz = round(zz, sig_figs)
    xy = round(xy, sig_figs)
    xz = round(xz, sig_figs)
    yz = round(yz, sig_figs)
    yx = round(yx, sig_figs)
    zx = round(zx, sig_figs)
    zy = round(zy, sig_figs)
    try:
        symmetric_failures = check_symmetry(xy=xy, xz=xz, yz=yz, yx=yx, zx=zx, zy=zy)
    except ValueError as e:
        symmetric_failures = 1
        logger.error(
            f"Symmetry failed for {x}, {y}, {z}. {e} and distance is {distance} with metric {metric}",
        )
    try:
        triangle_failures = check_triangle_inequality(
            x=x,
            y=y,
            z=z,
            xz=xz,
            xy=xy,
            yz=yz,
        )
    except ValueError as e:
        triangle_failures = 1
        logger.error(
            f"Triangle Inequality failed for {x}, {y}, {z}. {e} and distance is {distance} with metric {metric}",
        )
    try:
        zero_failures = check_zero(
            x=x,
            y=y,
            z=z,
            xx=xx,
            yy=yy,
            zz=zz,
            xy=xy,
            xz=xz,
            yz=yz,
            yx=yx,
            zx=zx,
            zy=zy,
        )
    except ValueError as e:  # noqa E722
        zero_failures = 1
        logger.error(
            f"Zero Identity failed for {x}, {y}, {z}. {e} and distance is {distance} with metric {metric}",
        )
    try:
        positivity_failures = check_positivity(xy=xy, xz=xz, yz=yz, yx=yx, zx=zx, zy=zy)
    except ValueError as e:
        positivity_failures = 1
        logger.error(
            f"Positivity Identity failed for {x}, {y}, {z}. {e} and distance is {distance} with metric {metric}",
        )
    return triangle_failures, symmetric_failures, zero_failures, positivity_failures


def get_distance_function(distance):
    if distance == "Vanilla":
        dist = unmodified_ncd
    elif distance == "Assumed":
        dist = modified_ncd
    elif distance == "Enforced":
        dist = sorted_ncd
    elif distance == "Averaged":
        dist = averaged_ncd
    else:
        raise NotImplementedError(
            f"Only unmodified_ncd, modified_ncd, and length_sorted_ncd are supported as distance functions. You chose {distance}",
        )
    return dist


def check_all_metric_space_assumptions(
    max_sig_figs=10,
    samples=1000,
    max_string_size=1000,
    max_alphabet_size=95,
    iterate="sig_figs",
    data="random",
    distance="unmodified_ncd",
    metric="gzip",
):
    symmetries = []
    zeroes = []
    triangles = []
    positivities = []
    iterators = []
    if iterate == "sig_figs":
        kwargs = {
            "samples": samples,
            "max_string_size": max_string_size,
            "alphabet_size": max_alphabet_size,
            "data": data,
        }
        # Run all sig figs
        iterator = range(0, max_sig_figs)
        title = "Significant Figures"
    elif iterate == "max_string_size":
        kwargs = {
            "sig_figs": max_sig_figs,
            "samples": samples,
            "alphabet_size": max_alphabet_size,
            "data": data,
        }
        # Run all powers of 10 from 1e-5 to max_string_size using np.logspace
        iterator = range(1, max_string_size)
        title = "Max String Size"
    elif iterate == "alphabet_size":
        kwargs = {
            "sig_figs": max_sig_figs,
            "samples": samples,
            "max_string_size": max_string_size,
            "data": data,
        }
        iterator = range(1, max_alphabet_size)
        title = "Alphabet Size"
    else:
        raise ValueError("Invalid iterate")
    kwargs["metric"] = metric
    if len(iterator) > 10:
        # divide the iterator into 10 parts
        max_ = max(iterator)
        min_ = min(iterator)
        # Create 10 parts of the iterator
        iterator = np.linspace(min_, max_, 10, endpoint=True)
    iterator = [int(i) for i in iterator]
    for i in tqdm(
        iterator,
        desc=f"Running {iterate} for {metric} compression and distance algorithm {distance}.",
        total=len(iterator),
        position=1,
        leave=False,
    ):
        logger.info(f"{title.capitalize()}")
        logger.info(f"Running {iterate} = {i}")
        if iterate == "samples" and i == 0:
            continue
        if iterate == "alphabet_size" and i == 0:
            continue
        if iterate == "max_string_size" and i == 0:
            continue
        kwargs[iterate] = i
        t, s, z, p = check_loop(**kwargs, distance=distance)
        logger.info(f"Percent of examples where zero identity was violated: {z}")
        logger.info(f"Percent of examples where positivity  was violated: {p}")
        logger.info(f"Percent of examples where symmetry was violated: {s}")
        logger.info(f"Percent of examples where triangle inequality was violated: {t}")
        symmetries.append(s)
        zeroes.append(z)
        triangles.append(t)
        positivities.append(p)
        iterators.append(i)

    # # Turn results into a dataframe
    results = {
        "Symmetry": symmetries,
        "Zero Identity": zeroes,
        "Triangle Inequality": triangles,
        "Positivity Identity": positivities,
        "Iterate": iterate,
        "Values": iterators,
        "i": iterators,
    }
    df = pd.DataFrame(results)
    # Melt the dataframe so that it can be plotted using seaborn
    df = pd.melt(
        df,
        id_vars=["Iterate", "i"],
        value_vars=[
            "Symmetry",
            "Zero Identity",
            "Triangle Inequality",
            "Positivity Identity",
        ],
    )
    df["Identity"] = df["variable"]
    del df["variable"]
    df["Percent Violations"] = df["value"]
    df["Metric"] = metric
    df["Algorithm"] = distance
    del df["value"]
    return df


def find_all_combinations(max_alphabet_size=52, max_string_size=10, reverse=False):
    chars = []
    chars = (
        string.ascii_lowercase + string.ascii_uppercase
    )  # Default dictionary of 52 characters
    # Add  string.digits + string.punctuation + " " to the end
    chars += (
        string.digits
        + string.punctuation
        + " "
        + string.ascii_uppercase
        + string.ascii_lowercase
    )
    # Join into a string
    assert max_alphabet_size <= len(chars), ValueError("Alphabet size is too large")
    chars = chars[:max_alphabet_size]
    # Find all combinations of length max_string_size
    perms = []
    for i in tqdm(range(1, max_string_size + 1), desc="Finding all combinations"):
        marginal = combinations(chars, i)
        # Join each entry into a string
        marginal = ["".join(m) for m in marginal]
        perms.extend(marginal)
    # Find all combinations of length 3
    logger.info("Finding triplets")
    combs = list(product(perms, repeat=3))
    # Filter out any combinations that are not unique
    return combs


def check_all_distances(args):
    distances = [
        "Vanilla",
        "Assumed",
        "Enforced",
        "Averaged",
    ]  # , "subset_ncd" "distance_safe_ncd", "subset_ncd"
    # metric_list = list(compressors.keys())
    metric_list = ["gzip", "bz2", "brotli", "levenshtein", "ratio", "hamming"]
    arg_list = []
    for metric in metric_list:
        for i in range(len(distances)):
            distance = distances[i]
            # Make sure the log file exists
            big_df = pd.DataFrame()
            # Parallelize the for loop
            for iterate in args.iterate:
                arg_dict = {
                    "max_sig_figs": args.sig_figs,
                    "samples": args.samples,
                    "max_string_size": args.max_string_size,
                    "max_alphabet_size": args.max_alphabet_size,
                    "distance": distance,
                    "iterate": iterate,
                    "data": args.data,
                    "metric": metric,
                }
                arg_list.append(arg_dict)
    dfs = []
    for arg_dict in tqdm(
        arg_list,
        desc="Checking all  , iterates, and NCD algorithms",
        position=0,
        leave=True,
    ):
        df = check_all_metric_space_assumptions(**arg_dict)
        dfs.append(df)
        big_df = checkpoint_results(args, *dfs)
        plot_identity_violations(args, big_df)
    return big_df


def checkpoint_results(args, *dfs):
    big_df = pd.concat([*dfs], axis=0)
    csv_file = Path(args.folder) / Path(args.results_file)
    # Save the results to a csv file
    if not Path(csv_file).exists():
        big_df.to_csv(csv_file, index=False, header=True)
    else:
        big_df.to_csv(
            csv_file,
            index=False,
            mode="a",
            header=False,
        )

    return big_df


def plot_identity_violations(args, big_df):
    big_df["i"] = big_df["i"].astype(int)
    big_df["Percent Violations"] = big_df["Percent Violations"] * 100
    big_df["Percent Violations"] = big_df["Percent Violations"].round(2).astype(float)
    # Replace the underscore in "Iterate", "Algorithm" with a space
    big_df["Iterate"] = big_df["Iterate"].str.replace("_", " ")
    big_df["Algorithm"] = big_df["Algorithm"].str.replace("_", " ")
    # Title-ize the iterate and distance columns
    big_df["Iterate"] = big_df["Iterate"].str.title()
    big_df["Iterate"].str.replace("Sig Figs", "Significant Figures")
    metric_dict = {
        "gzip": "GZIP",
        "bz2": "BZ2",
        "brotli": "Brotli",
        "levenshtein": "Levenshtein",
        "ratio": "Ratio",
        "hamming": "Hamming",
    }
    big_df["Identity"] = big_df["Identity"].str.replace("_", " ").str.title()
    big_df["Metric"] = big_df["Metric"].map(metric_dict)
    # Drop " Ncd" from the distance column
    cols = ["Sig Figs", "Max String Size", "Alphabet Size"]
    g = sns.relplot(
        data=big_df,
        x="i",
        y="Percent Violations",
        col="Iterate",
        row="Algorithm",
        kind="line",
        height=4,
        aspect=1,
        hue="Identity",
        style="Metric",
        row_order=["Vanilla", "Assumed", "Enforced", "Averaged"],
        col_order=cols,
        facet_kws={"sharex": True, "sharey": True},
    )
    g.set_titles("{col_name} - {row_name}")
    g.set_axis_labels("", "Percent Violations")
    g._legend.set_title("Identity")
    # Rotate the x-axis labels
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    # Tight layout
    g.tight_layout()
    g.savefig(f"{args.folder}/{args.plot_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sig_figs", type=int, default=5)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--max_string_size", type=int, default=5)
    parser.add_argument("--max_alphabet_size", type=int, default=95)  # acutal max is 95
    parser.add_argument(
        "--iterate",
        type=str,
        nargs="+",
        default=["sig_figs", "max_string_size", "alphabet_size"],
    )
    parser.add_argument("--data", type=str, default="alphabet")
    parser.add_argument("--folder", type=str, default="metric_space_check")
    parser.add_argument("--log_file", type=str, default="metric_space_check.log")
    parser.add_argument("--results_file", type=str, default="results.csv")
    parser.add_argument("--plot_file", type=str, default="results.pdf")
    args = parser.parse_args()
    # Log file
    log_file = Path(args.folder) / Path(args.log_file)
    # Make the directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    # Touch the file
    log_file.touch()
    log_file = log_file.as_posix()
    logging.basicConfig(filename=log_file, level=logging.ERROR)

    big_df = check_all_distances(args)

    # Generate a sns.catplot with cols=iterate, x=Values, y=Percent Violations, hue=Identity
    csv_file = Path(args.folder) / Path(args.results_file)
    big_df = pd.read_csv(csv_file)
    plot_identity_violations(args, big_df)
