import random
import string
from pathlib import Path
from tqdm import tqdm
import numpy as np
from itertools import product, combinations
import logging
import argparse
import plotext as plt

from gzip_classifier import ncd as old_ncd
logger = logging.getLogger(__name__)
# import wrapper from python decorator



letter_frequency_upper = {
    "E": 12.0,
    "T": 9.10,
    "A": 8.12,
    "O": 7.68,
    "I": 7.31,
    "N": 6.95,
    "S": 6.28,
    "R": 6.02,
    "H": 5.92,
    "D": 4.32,
    "L": 3.98,
    "U": 2.88,
    "C": 2.71,
    "M": 2.61,
    "F": 2.30,
    "Y": 2.11,
    "W": 2.09,
    "G": 2.03,
    "P": 1.82,
    "B": 1.49,
    "V": 1.11,
    "K": 0.69,
    "X": 0.17,
    "Q": 0.11,
    "J": 0.10,
    "Z": 0.07,
}

letter_frequency_lower = {k.lower(): v for k, v in letter_frequency_upper.items()}

letter_frequencies = {**letter_frequency_upper, **letter_frequency_lower}


def modified_ncd(x, y):
    x = str(x) if not isinstance(x, str) else x
    y = str(y) if not isinstance(y, str) else y
    if x == y:
        return 0
    else:
        if x > y:
            return old_ncd(x, y)
        else:
            return old_ncd(y, x)


def unmodified_ncd(x, y):
    return old_ncd(x, y)


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
    chars = list(set(chars))
    chars = chars[:alphabet_size]
    return "".join(random.choice(chars) for _ in range(size))


def check_triangle_inequality(x, y, z):
    xz = ncd(x, z)
    yz = ncd(y, z)
    xy = ncd(x, y)
    if xz > xy + yz:
        raise ValueError(
            f"Triangle Inequality failed for {x}, {y}, {z}. <x,z> = {xz} > <x,y> + <y,z> = {xy + yz}"
        )
    return None


def check_symmetry(x, y, z, sig_figs=2):
    xz = ncd(x, z)
    yz = ncd(y, z)
    xy = ncd(x, y)
    zx = ncd(z, x)
    zy = ncd(z, y)
    yx = ncd(y, x)
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
        return None


def check_zero(x, y, z, sig_figs=2):
    xx = ncd(x, x)
    yy = ncd(y, y)
    zz = ncd(z, z)
    xx = round(xx, sig_figs)
    yy = round(yy, sig_figs)
    zz = round(zz, sig_figs)
    if xx != 0:
        raise ValueError(f"<x,x> = {xx}")
    elif yy != 0:
        raise ValueError(f"<y,y> = {yy}")
    elif zz != 0:
        raise ValueError(f"<z,z> = {zz}")
    else:
        return None


def check_positivity(x, y, z, sig_figs=2):
    xz = ncd(x, z)
    zx = ncd(z, x)
    yz = ncd(y, z)
    zy = ncd(z, y)
    xy = ncd(x, y)
    yx = ncd(y, x)
    xz = round(xz, sig_figs)
    zx = round(zx, sig_figs)
    yz = round(yz, sig_figs)
    zy = round(zy, sig_figs)
    xy = round(xy, sig_figs)
    yx = round(yx, sig_figs)
    assert xz >= 0, ValueError("<x,z> < 0")
    assert zx >= 0, ValueError("<z,x> < 0")
    assert yz >= 0, ValueError("<y,z> < 0")
    assert zy >= 0, ValueError("<z,y> < 0")
    assert xy >= 0, ValueError("<x,y> < 0")
    assert yx >= 0, ValueError("<y,x> < 0")

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
    return None


def check_loop(
    samples=1000, sig_figs=2, max_string_size=1000, data="random", alphabet_size=52
):
    triangle_failures = 0
    symmetric_failures = 0
    zero_failures = 0
    positivity_failures = 0
    if data == "random":
        pass
        total = samples
    elif data == "combinations":
        combinations = find_all_combinations(
            max_alphabet_size=alphabet_size,
            max_string_size=max_string_size,
            reverse=True,
        )
        total = len(combinations)
    else:
        raise NotImplementedError(
            "Only random strings and alphabet combinations are supported at the data."
        )
    with tqdm(total=total) as pbar:
        for i in range(total):
            if data == "random":
                x = string_generator(
                    size=random.randint(1, max_string_size), alphabet_size=alphabet_size
                )
                y = string_generator(
                    size=random.randint(1, max_string_size), alphabet_size=alphabet_size
                )
                z = string_generator(
                    size=random.randint(1, max_string_size), alphabet_size=alphabet_size
                )
            elif data == "combinations":
                x, y, z = combinations[i]
            elif isinstance(data, str) and Path(data).exists():
                raise NotImplementedError
            try:
                check_symmetry(x, y, z, sig_figs=sig_figs)
            except ValueError as e:
                symmetric_failures += 1
                logger.error(f"Symmetry failed for {x}, {y}, {z}. {e}")
            try:
                check_triangle_inequality(x, y, z)
            except ValueError as e:  # noqa E722
                triangle_failures += 1
                logger.error(f"Triangle Inequality failed for {x}, {y}, {z}. {e}")
            try:
                check_zero(x, y, z, sig_figs=sig_figs)
            except ValueError as e:  # noqa E722
                zero_failures += 1
                logger.error(f"Zero Identity failed for {x}, {y}, {z}. {e}")
            try:
                check_positivity(x, y, z)
            except ValueError as e:
                positivity_failures += 1
                logger.error(f"Positivity Identity failed for {x}, {y}, {z}. {e}")
            pbar.update(1)
            # Convert failures to percent
    tri = triangle_failures / (samples)
    sym = symmetric_failures / (samples)
    zer = zero_failures / (samples)
    pos = positivity_failures / (samples)
    return tri, sym, zer, pos


def check_all_metric_space_assumptions(
    max_sig_figs=10,
    samples=1000,
    max_string_size=1000,
    max_alphabet_size=95,
    iterate="sig_figs",
    data="random",
):
    symmetries = []
    zeroes = []
    triangles = []
    positivities = []
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
        iterator = np.logspace(1, np.log10(max_string_size), num=20, dtype=int)
        iterator = [int(i) for i in iterator]
        title = "Max String Size"
    elif iterate == "alphabet_size":
        kwargs = {
            "sig_figs": max_sig_figs,
            "samples": samples,
            "max_string_size": max_string_size,
            "data": data,
        }
        iterator = range(0, max_alphabet_size)
        iterator = [int(i) for i in iterator]
        title = "Alphabet Size"
    else:
        raise ValueError("Invalid iterate")
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
        t, s, z, p = check_loop(**kwargs)
        print(f"Percent of examples where zero identity was violated: {z}")
        print(f"Percent of examples where positivity  was violated: {p}")
        print(f"Percent of examples where symmetry was violated: {s}")
        print(f"Percent of examples where triangle inequality was violated: {t}")
        symmetries.append(s)
        zeroes.append(z)
        triangles.append(t)
        positivities.append(p)
    plt.clear_figure()
    plt.plot(symmetries, label="Symmetry violations")
    plt.plot(zeroes, label="Zero Identity Violations")
    plt.plot(positivities, label="Positivity Identity violations")
    plt.plot(triangles, label="Triangle Inequality Violations")
    plt.xticks(
        range(len(iterator)), iterator
    )  # Set the x-axis ticks to be the iterator
    plt.title(f"Identity Violations vs. {title}")
    plt.xlabel(title)
    plt.ylabel("Percent Violations")
    plt.show()
    filename = title.replace(" ", "_").lower()
    plt.save_fig(f"./{filename}.html")


def find_all_combinations(max_alphabet_size=52, max_string_size=10, reverse=False):
    chars = string.ascii_lowercase
    # sort by frequency,
    chars = sorted(chars, key=lambda x: letter_frequencies.get(x, 0), reverse=reverse)
    # Add  string.digits + string.punctuation + " " to the end
    chars += string.digits + string.punctuation + " " + string.ascii_uppercase
    # Join into a string
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sig_figs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--max_string_size", type=int, default=3)
    parser.add_argument("--max_alphabet_size", type=int, default=95)
    parser.add_argument(
        "--iterate",
        type=str,
        nargs="+",
        default=["sig_figs", "max_string_size", "alphabet_size"],
    )
    parser.add_argument("--data", type=str, default="random")
    parser.add_argument("--modified_ncd", default=False, action="store_true")
    args = parser.parse_args()
    # Log file
    logging.basicConfig(filename="metric_space_check.log", level=logging.ERROR)

    # Combinations
    combs = find_all_combinations(args.max_alphabet_size, args.max_string_size)
    # Write to file, one combination per line
    str_ = ""
    for c in tqdm(combs[:10], total=len(combs[:10]), desc="Writing combinations"):
        str_ += f"{c[0]} {c[1]} {c[2]}\n"
    with open("combinations.txt", "w") as f:
        f.write(str_)
    if args.modified_ncd:
        ncd = modified_ncd
    else:
        ncd = unmodified_ncd
    for iterate in args.iterate:
        check_all_metric_space_assumptions(
            max_sig_figs=args.sig_figs,
            samples=1000,
            max_string_size=args.max_string_size,
            max_alphabet_size=args.max_alphabet_size,
            iterate=iterate,
            data=args.data,
        )
