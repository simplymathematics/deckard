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
# import wrapper from python decorator


# letter_frequency_upper = {
#     "E": 12.0,
#     "T": 9.10,
#     "A": 8.12,
#     "O": 7.68,
#     "I": 7.31,
#     "N": 6.95,
#     "S": 6.28,
#     "R": 6.02,
#     "H": 5.92,
#     "D": 4.32,
#     "L": 3.98,
#     "U": 2.88,
#     "C": 2.71,
#     "M": 2.61,
#     "F": 2.30,
#     "Y": 2.11,
#     "W": 2.09,
#     "G": 2.03,
#     "P": 1.82,
#     "B": 1.49,
#     "V": 1.11,
#     "K": 0.69,
#     "X": 0.17,
#     "Q": 0.11,
#     "J": 0.10,
#     "Z": 0.07,
# }

# letter_frequency_lower = {k.lower(): v for k, v in letter_frequency_upper.items()}

# letter_frequencies = {**letter_frequency_upper, **letter_frequency_lower}


def find_longest_common_substring(x, y):
    m = len(x)
    n = len(y)
    common = ""
    for i in range(m):
        for j in range(n):
            length = 0
            while i + length < m and j + length < n and x[i + length] == y[j + length]:
                length += 1
            if length > len(common):
                common = x[i : i + length]  # noqa E203
    return common


def old_ncd(
    x1,
    x2,
    cx1=None,
    cx2=None,
    method: Literal["gzip", "lzma", "bz2", "zstd", "pkl", None] = "gzip",
) -> float:
    """
    Calculate the normalized compression distance between two objects treated as strings.
    Args:
        x1 (str): The first object
        x2 (str): The second object
    Returns:
        float: The normalized compression distance between x1 and x2
    """

    compressor_len = (
        compressors[method] if method in compressors.keys() else compressors["gzip"]
    )
    Cx1 = compressor_len(x1) if cx1 is None else cx1
    Cx2 = compressor_len(x2) if cx2 is None else cx2
    x1x2 = "".join([x1, x2])
    Cx1x2 = compressor_len(x1x2)
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
    x, y = sort_xy(x, y)
    return old_ncd(x, y, method=method)


def sort_xy(x, y):
    x = str(x) if not isinstance(x, str) else x
    y = str(y) if not isinstance(y, str) else y
    if x < y:
        return y, x
    else:
        return x, y


def distance_safe_ncd(x, y, method="gzip"):
    # Return 0 if x == y
    if x == y:
        return 0
    # Sort x and y to ensure that the distance is symmetric
    x, y = sort_xy(x, y)
    # Calculate the distance
    ncd = actual_min_ncd(x, y, method=method)
    return ncd


def subset_ncd(x, y, method="gzip", replacement=-1, max_iters=-1, shortest_common=3):
    x = str(x) if not isinstance(x, str) else x
    y = str(y) if not isinstance(y, str) else y
    # Use ascii punctuation + digits as the "new" alphabet, stored as a list
    new_alphabet = string.punctuation + string.digits
    new_alphabet = list(set(new_alphabet))
    if isinstance(replacement, int):
        assert len(new_alphabet) >= replacement, ValueError("Replacement is too large")
        replacement = new_alphabet[replacement]
    else:
        assert isinstance(replacement, str), ValueError(
            "Replacement must be an integer or a string",
        )
    x, y = sort_xy(x, y)
    if x == y:
        return 0
    if x in y:
        # remove x from y
        y = y.replace(x, replacement)
    elif y in x:
        # remove y from x
        x = x.replace(y, replacement)
    else:
        if max_iters is None:
            longest_common = find_longest_common_substring(x, y)
            x = x.replace(longest_common, replacement)
            y = y.replace(longest_common, replacement)
        else:
            if max_iters == -1:
                max_iters = max(len(x), len(y), len(replacement))
            for i in range(max_iters):
                longest_common = find_longest_common_substring(x, y)
                if i > len(replacement) - 1:  # only works up 42^2 = 1764 iterations
                    replacement_j = new_alphabet[i % len(new_alphabet)]
                    replacement_i = new_alphabet[i // len(new_alphabet)]
                    replacement_i = replacement_i + replacement_j
                else:
                    replacement_i = replacement[i]
                x = x.replace(longest_common, replacement_i)
                y = y.replace(longest_common, replacement_i)
                if longest_common == "" or shortest_common > len(longest_common):
                    break
    return sorted_ncd(x, y, method=method)


def replace_largest_common_substring(x, y, replacement=""):
    common = find_longest_common_substring(x, y)
    x = x.replace(common, replacement)
    y = y.replace(common, replacement)
    return x, y


def actual_min_ncd(x, y, method="gzip"):
    x = str(x) if not isinstance(x, str) else x
    y = str(y) if not isinstance(y, str) else y
    compressed_length_xy = compressors[method](x + y)
    compressed_length_x = compressors[method](x)
    compressed_length_y = compressors[method](y)
    compressed_length_yx = compressors[method](y + x)
    actual_min = min(
        compressed_length_xy,
        compressed_length_x,
        compressed_length_y,
        compressed_length_yx,
        # length_x,
        # length_y,
        # length_xy,
    )
    if actual_min == compressed_length_xy and (
        actual_min != compressed_length_x and actual_min != compressed_length_y
    ):
        print(f"Compressed length of x: {compressed_length_x}")
        print(f"Compressed length of y: {compressed_length_y}")
        print(f"Compressed length of xy: {compressed_length_xy}")
        input(
            "Actual min is compressed length of xy, but not compressed length of x or y",
        )

    ncd = (compressed_length_xy - actual_min) / max(
        compressed_length_x,
        compressed_length_y,
    )
    return ncd


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


def check_triangle_inequality(x, y, z, dist=unmodified_ncd, method="gzip"):
    xz = dist(x, z, method=method)
    yz = dist(y, z, method=method)
    xy = dist(x, y, method=method)
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


def check_symmetry(x, y, z, sig_figs=10, dist=unmodified_ncd, method="gzip"):
    xz = dist(x, z, method=method)
    yz = dist(y, z, method=method)
    xy = dist(x, y, method=method)
    zx = dist(z, x, method=method)
    zy = dist(z, y, method=method)
    yx = dist(y, x, method=method)
    xz = round(xz, sig_figs)
    yz = round(yz, sig_figs)
    xy = round(xy, sig_figs)
    zx = round(zx, sig_figs)
    zy = round(zy, sig_figs)
    yx = round(yx, sig_figs)
    # assert xz == zx, ValueError(f"XZ: {xz} != {zx}")
    # assert yz == zy, ValueError(f"YZ: {yz} != {zy}")
    # assert xy == yx, ValueError(f"XY: {xy} != {yx}")
    # return None
    if xz != zx:
        raise ValueError(f"XZ: {xz} != {zx}")
    elif yz != zy:
        raise ValueError(f"YZ: {yz} != {zy}")
    elif xy != yx:
        raise ValueError(f"XY: {xy} != {yx}")
    else:
        return 0


def check_zero(x, y, z, sig_figs=10, dist=unmodified_ncd, method="gzip"):
    xx = dist(x, x, method=method)
    yy = dist(y, y, method=method)
    zz = dist(z, z, method=method)
    xx = round(xx, sig_figs)
    yy = round(yy, sig_figs)
    zz = round(zz, sig_figs)
    if xx != 0:
        raise ValueError(f"<x,x> = {xx}")
    elif yy != 0:
        raise ValueError(f"<y,y> = {yy}")
    elif zz != 0:
        raise ValueError(f"<z,z> = {zz}")

    yx = dist(y, x)
    xy = dist(x, y)
    zx = dist(z, x)
    xz = dist(x, z)
    yz = dist(y, z)
    zy = dist(z, y)
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


def check_positivity(x, y, z, sig_figs=10, dist=unmodified_ncd, method="gzip"):
    xz = dist(x, z, method=method)
    zx = dist(z, x, method=method)
    yz = dist(y, z, method=method)
    zy = dist(z, y, method=method)
    xy = dist(x, y, method=method)
    yx = dist(y, x, method=method)
    xz = round(xz, sig_figs)
    zx = round(zx, sig_figs)
    yz = round(yz, sig_figs)
    zy = round(zy, sig_figs)
    xy = round(xy, sig_figs)
    yx = round(yx, sig_figs)
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
    compressor="gzip",
):

    # Choose the distance function
    if distance == "unmodified_ncd":
        dist = unmodified_ncd
    elif distance == "modified_ncd":
        dist = modified_ncd
    elif distance == "sorted_ncd":
        dist = sorted_ncd
    elif distance == "subset_ncd":
        dist = subset_ncd
    elif distance == "distance_safe_ncd":
        dist = distance_safe_ncd
    else:
        raise NotImplementedError(
            f"Only unmodified_ncd, modified_ncd, and length_sorted_ncd are supported as distance functions. You chose {distance}",
        )
    arg_list = []

    # Generate a list of arguments for the parallelized for loop
    for i in range(samples):
        x, y, z = get_data_triplet(max_string_size, data, alphabet_size, samples, i)
        arg_list += [(sig_figs, distance, compressor, dist, x, y, z)]

    # Parallelize the for loop using joblib and tqdm
    df = np.array(
        Parallel(n_jobs=-1, verbose=False)(
            delayed(count_metric_assumption_failures)(*args)
            for args in tqdm(
                arg_list,
                desc=f"Checking metric space assumptions for {distance} algorithm using the {compressor} compressor.",
            )
        ),
    )  # 4 columns, 1 for each assumption,
    # Convert failures to percent
    df = df.sum(axis=0) / samples
    print(f"Percent of examples where triangle inequality was violated: {df[0]}")
    print(f"Percent of examples where symmetry was violated: {df[1]}")
    print(f"Percent of examples where zero identity was violated: {df[2]}")
    print(f"Percent of examples where positivity was violated: {df[3]}")
    print(f"Shape of df is {df.shape}")
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


def count_metric_assumption_failures(sig_figs, distance, compressor, dist, x, y, z):
    try:
        symmetric_failures = check_symmetry(
            x,
            y,
            z,
            sig_figs=sig_figs,
            dist=dist,
            method=compressor,
        )
    except ValueError as e:
        symmetric_failures = 1
        logger.error(
            f"Symmetry failed for {x}, {y}, {z}. {e} and distance is {distance} with compressor {compressor}",
        )
    try:
        triangle_failures = check_triangle_inequality(
            x,
            y,
            z,
            dist=dist,
            method=compressor,
        )
    except ValueError as e:
        triangle_failures = 1
        logger.error(
            f"Triangle Inequality failed for {x}, {y}, {z}. {e} and distance is {distance} with compressor {compressor}",
        )
    try:
        zero_failures = check_zero(
            x,
            y,
            z,
            sig_figs=sig_figs,
            dist=dist,
            method=compressor,
        )
    except ValueError as e:  # noqa E722
        zero_failures = 1
        logger.error(
            f"Zero Identity failed for {x}, {y}, {z}. {e} and distance is {distance} with compressor {compressor}",
        )
    try:
        positivity_failures = check_positivity(x, y, z, dist=dist, method=compressor)
    except ValueError as e:
        positivity_failures = 1
        logger.error(
            f"Positivity Identity failed for {x}, {y}, {z}. {e} and distance is {distance} with compressor {compressor}",
        )
    return triangle_failures, symmetric_failures, zero_failures, positivity_failures


def check_all_metric_space_assumptions(
    max_sig_figs=10,
    samples=1000,
    max_string_size=1000,
    max_alphabet_size=95,
    iterate="sig_figs",
    data="random",
    distance="unmodified_ncd",
    compressor="gzip",
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
    kwargs["compressor"] = compressor
    if len(iterator) > 10:
        # divide the iterator into 10 parts
        max_ = max(iterator)
        min_ = min(iterator)
        # Create 10 parts of the iterator
        iterator = np.linspace(min_, max_, 10)
    iterator = [int(i) for i in iterator]
    for i in iterator:
        print(f"{title.capitalize()}")
        print(f"Running {iterate} = {i}")
        if iterate == "samples" and i == 0:
            continue
        if iterate == "alphabet_size" and i == 0:
            continue
        if iterate == "max_string_size" and i == 0:
            continue
        kwargs[iterate] = i
        t, s, z, p = check_loop(**kwargs, distance=distance)
        print(f"Percent of examples where zero identity was violated: {z}")
        print(f"Percent of examples where positivity  was violated: {p}")
        print(f"Percent of examples where symmetry was violated: {s}")
        print(f"Percent of examples where triangle inequality was violated: {t}")
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
    df["Method"] = compressor
    df["distance"] = distance
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
    print("Finding triplets")
    combs = list(product(perms, repeat=3))
    # Filter out any combinations that are not unique
    return combs


def check_all_distances(args):
    distances = [
        "unmodified_ncd",
        "sorted_ncd",
        "modified_ncd",
    ]  # , "subset_ncd" "distance_safe_ncd", "subset_ncd"
    compressor_list = list(compressors.keys())
    # remove 'pkl' from the list of compressors
    compressor_list.remove("pkl")
    log_file = Path(args.folder) / Path(args.log_file)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.ERROR)
    for compressor in compressor_list:
        for i in range(len(distances)):
            distance = distances[i]
            # Make sure the log file exists
            big_df = pd.DataFrame()
            # Parallelize the for loop
            for iterate in args.iterate:
                df = check_all_metric_space_assumptions(
                    max_sig_figs=args.sig_figs,
                    samples=args.samples,
                    max_string_size=args.max_string_size,
                    max_alphabet_size=args.max_alphabet_size,
                    distance=distance,
                    iterate=iterate,
                    data=args.data,
                    compressor=compressor,
                )
                big_df = checkpoint_results(args, big_df, df)
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
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    big_df["i"] = big_df["i"].astype(int)
    big_df["Percent Violations"] = big_df["Percent Violations"] * 100
    big_df["Percent Violations"] = big_df["Percent Violations"].round(2).astype(float)

    # Replace the underscore in "Iterate", "distance" with a space
    big_df["Iterate"] = big_df["Iterate"].str.replace("_", " ")
    big_df["distance"] = big_df["distance"].str.replace("_", " ")
    # Title-ize the iterate and distance columns
    big_df["Iterate"] = big_df["Iterate"].str.title()
    big_df["Iterate"].str.replace("Sig Figs", "Significant Figures")
    big_df["distance"] = big_df["distance"].str.title()
    big_df["Compressor"] = big_df["method"].str.upper()
    # Drop " Ncd" from the distance column
    big_df["distance"] = big_df["distance"].str.replace(" Ncd", "")
    g = sns.relplot(
        data=big_df,
        x="i",
        y="Percent Violations",
        col="Iterate",
        row="distance",
        kind="line",
        height=4,
        aspect=1,
        hue="Identity",
        style="Compressor",
        row_order=["Unmodified", "Sorted", "Modified"],
        col_order=["Sig Figs", "Max String Size", "Alphabet Size"],
        facet_kws={"sharex": False, "sharey": True},
    )
    g.set_titles("{col_name} | {row_name}")
    g.set_xlabels("{col_name}")
    g.set_axis_labels("Values", "Percent Violations")
    g.savefig(f"{args.folder}/{args.results_plot}")


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
    parser.add_argument("--results_plot", type=str, default="results.pdf")
    args = parser.parse_args()
    # Log file

    big_df = check_all_distances(args)

    # Generate a sns.catplot with cols=iterate, x=Values, y=Percent Violations, hue=Identity
    csv_file = Path(args.folder) / Path(args.results_file)
    big_df = pd.read_csv(csv_file)
    plot_identity_violations(args, big_df)
