import random
import string
from tqdm import tqdm
from gzip_classifier import ncd

# from gzip_classifier import modified_ncd as ncd
import plotext as plt


def id_generator(
    size=6, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits
):
    return "".join(random.choice(chars) for _ in range(size))


def check_triangle_inequality(x, y, z):
    xz = ncd(x, z)
    yz = ncd(y, z)
    xy = ncd(x, y)
    assert xz <= xy + yz, "Triangle Inequality Broken"
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
    assert xz == zx, f"XZ: {xz} != {zx}"
    assert yz == zy, f"YZ: {yz} != {zy}"
    assert xy == yx, f"XY: {xy} != {yx}"
    return None


def check_zero(x, y, z, sig_figs=2):
    xx = ncd(x, x)
    yy = ncd(y, y)
    zz = ncd(z, z)
    xx = round(xx, sig_figs)
    yy = round(yy, sig_figs)
    zz = round(zz, sig_figs)
    assert xx == 0, f"<x,x> = {xx}"
    assert yy == 0, f"<y,y> = {yy}"
    assert zz == 0, f"<z,z> = {zz}"
    return None


def check_positivity(x, y, z):
    assert ncd(x, z) > 0, f"NCD(x,z) <= 0"
    assert ncd(z, x) > 0, f"NCD(z,x) <= 0"
    assert ncd(x, y) > 0, f"NCD(x,y) <= 0"
    assert ncd(y, x) > 0, f"NCD(y,x) <= 0"
    assert ncd(y, z) > 0, f"NCD(y,z) <= 0"
    assert ncd(z, y) > 0, f"NCD(y,z) <= 0"
    return None


def check_loop(number=1000, sig_figs=2, max_size=1000, data="random"):
    triangle_failures = 0
    symmetric_failures = 0
    zero_failures = 0
    positivity_failures = 0
    with tqdm(total=number) as pbar:
        for i in range(number):
            if data == "random":
                x = id_generator(size=random.randint(1, max_size))
                y = id_generator(size=random.randint(1, max_size))
                z = id_generator(size=random.randint(1, max_size))
            elif isinstance(data, str) and Path(data).exists():
                raise NotImplementedError
            try:
                assert (
                    check_symmetry(x, y, z, sig_figs=sig_figs) is None
                ), "Not Symmetric"
            except:
                symmetric_failures += 1
            try:
                assert (
                    check_triangle_inequality(x, y, z) is None
                ), "Triangle Inequality Broken"
            except:
                triangle_failures += 1
            try:
                assert (
                    check_zero(x, y, z, sig_figs=sig_figs) is None
                ), "Zero Identity broken"
            except:
                zero_failures += 1
            try:
                assert check_positivity(x, y, z) is None, "Positivity identity broken"
            except:
                positivity_failures += 1
            pbar.update(1)
            # Convert failures to percent
        tri = triangle_failures / (number) * 100
        sym = symmetric_failures / (number) * 100
        zer = zero_failures / (number) * 100
        pos = positivity_failures / (number) * 100
    return tri, sym, zer, pos


def check_all_metric_space_assumptions(max_sig_figs=10, samples=1000):
    symmetries = []
    zeroes = []
    triangles = []
    positivities = []
    for i in range(max_sig_figs):
        t, s, z, p = check_loop(1000, sig_figs=i)
        print(f"Significant Figures", i)
        print(f"Percent of examples where zero identity was violated: {z}")
        print(f"Percent of examples where positivity  was violated: {p}")
        print(f"Percent of examples where symmetry was violated: {s}")
        print(f"Percent of examples where triangle inequality was violated: {t}")
        symmetries.append(s)
        zeroes.append(z)
        triangles.append(t)
        positivities.append(p)

    plt.plot(symmetries, label="Symmetry violations")
    plt.plot(zeroes, label="Zero Identity Violations")
    plt.plot(positivities, label="Positivity Identity violations")
    plt.plot(triangles, label="Triangle Inequality Violations")
    plt.title("identity violations vs. Sig Figs")
    plt.show()


if __name__ == "__main__":
    # check_all_metric_space_assumptions(10, 1000)

    from gzip_classifier import modified_ncd as ncd

    check_all_metric_space_assumptions(10, 1000)
