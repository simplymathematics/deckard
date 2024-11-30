import codecs
import os

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "adversarial-robustness-toolbox",
    "numpy",
    "scipy",
    "scikit-learn",
    "tqdm",
    "numba",
    "pillow",
    "pandas",
    "PyYAML",
    "pytest",
    "validators",
    "hydra-core",
    "hydra-optuna-sweeper",
    "hydra-joblib-launcher",
    "sqlalchemy<=1.4.46",
    "dvc",
    "paretoset",
    "lifelines",
    "seaborn",
    "jinja2",
    "paretoset",
    "omegaconf",
]
test_requires = [
    "pytest",
    "pytest-xdist",
    "torch",
    "torchvision",
    "tensorflow",
    "pre-commit",
]
test_requires += install_requires
docs_require = [
    "sphinx >= 1.4",
    "sphinx_rtd_theme",
    "sphinx-autodoc-annotation",
    "sphinx-autodoc-typehints",
    "matplotlib",
    "numpy==1.26.4",
    "scipy>=1.4.1",
    "six>=1.13.0",
    "scikit-learn>=0.22.2,<1.1.0",
    "Pillow>=6.0.0",
]
test_requires += docs_require


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r", encoding="utf-8") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name="deckard",
    version=get_version("deckard/__init__.py"),
    description="Toolbox for adversarial machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="@simplymathematics",
    author_email="hello@simplymathematics.xyz",
    url="https://github.com/simplymathematics/deckard",
    license="MIT",
    install_requires=install_requires,
    extras_require={
        "docs": docs_require,
        "catboost": ["catboost"],
        "test": test_requires,
        "gpy": ["GPy"],
        "keras": ["keras", "h5py"],
        "lightgbm": ["lightgbm"],
        "mxnet": ["mxnet"],
        "tensorflow": ["tensorflow", "tensorflow_addons", "h5py"],
        "tensorflow_image": [
            "tensorflow",
            "tensorflow_addons",
            "h5py",
            "Pillow",
            "ffmpeg-python",
            "opencv-python",
        ],
        "tensorflow_audio": [
            "tensorflow",
            "tensorflow_addons",
            "h5py",
            "pydub",
            "resampy",
            "librosa",
        ],
        "pytorch": ["torch", "torchvision"],
        "pytorch_image": [
            "torch",
            "torchvision",
            "kornia",
            "Pillow",
            "ffmpeg-python",
            "opencv-python",
        ],
        "pytorch_audio": [
            "torch",
            "torchvision",
            "torchaudio",
            "pydub",
            "resampy",
            "librosa",
        ],
        "xgboost": ["xgboost"],
        "lingvo_asr": [
            "tensorflow-gpu==2.1.0",
            "lingvo==0.6.4",
            "pydub",
            "resampy",
            "librosa",
        ],
        "all": [
            "mxnet",
            "catboost",
            "lightgbm",
            "tensorflow",
            "tensorflow-addons",
            "h5py",
            "torch",
            "torchvision",
            "xgboost",
            "pandas",
            "kornia",
            "matplotlib",
            "Pillow",
            "statsmodels",
            "pydub",
            "resampy",
            "ffmpeg-python",
            "cma",
            "librosa",
            "opencv-python",
            "tensorflow",
            "dvc-gdrive",
        ],
        "non_framework": [
            "matplotlib",
            "pillow",
            "statsmodels",
            "pydub",
            "resampy",
            "ffmpeg-python",
            "cma",
            "pandas",
            "librosa",
            "opencv-python",
            "pytest",
            "pytest-flake8",
            "pytest-mock",
            "pytest-cov",
            "codecov",
            "requests",
            "sortedcontainers",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License ::  GPL License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    include_package_data=True,
)
