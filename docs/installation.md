# Installation

deckard is a declarative ML evaluation framework for reproducible experiments
across:

- data preparation and sampling
- model training and evaluation
- adversarial attacks and defenses
- fairness and privacy scoring
- experiment optimization and plotting

deckard uses Hydra/OmegaConf for configuration composition and supports Optuna
for multirun optimization.

## Prerequisites

- Python 3.10 or higher
- Git

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:simplymathematics/deckard.git
cd deckard
```

### 2. Setup a virtual environment

You can use either `pyenv` or `venv`. Below, there are instructions for both:

To set up a virtual environment using `pyenv`, follow these steps:

```bash
pyenv install 3.10
pyenv virtualenv 3.10 env
pyenv activate env
```

- `pyenv install 3.10` downloads and installs the newest Python version compatible with 3.10.
- `pyenv virtualenv 3.10 env` creates a new virtual environment named `env` using Python 3.10.
- `pyenv activate env` activates the `env` environment, so all Python commands use this isolated setup.

To set up a virtual environment using Python's built-in `venv`, run:

```bash
python3 -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
```

- `python3 -m venv env` creates a new virtual environment named `env`.
- `source env/bin/activate` (or `env\Scripts\activate` on Windows) activates the environment so all Python commands use this isolated setup.

This command switches your shell to use the `env` environment, ensuring all Python packages are installed locally within it.



### 3. Install Dependencies

```bash
python -m pip install -e .
```

This command installs the project's dependencies in "editable" mode. Editable mode (`-e .`) allows you to modify the source code and have changes reflected immediately without needing to reinstall the package. The `-m` flag tells Python to run the `pip` module as a script, ensuring you use the correct version of `pip` for your environment.

## Repository layout:

```
.
├── build <- exists after installation, contains the build files for the `deckard` executable.
├── coverage.sh <- Script for tracking test coverage and test timing measurments.
├── deckard <- Source directory
├── develop.md <- Documentation for developers
├── Dockerfile <- A docker environment for testing and deployment
├── docs <- The documentation 
├── examples <- Examples for each framework and optional extensions.
├── LICENSE  <- The software license file
├── notebooks <- Examples, but as Jupyter Notebooks
├── papers <- Published papers that use deckard
├── pyproject.toml <- python spec file for this package
├── README.md <- This file
├── setup.sh
└── test
```

## Usage

Run from the repository root:

```bash
python -m deckard --help
python -m deckard optimize --help
python -m deckard plot --help
```

Quick example:

```bash
python -m deckard optimize --config-name experiment \
	data.dataset_name=make_classification \
	model.model_type=sklearn.ensemble.RandomForestClassifier \
	attack.attack_type=art.attacks.evasion.FastGradientMethod \
	attack.attack_params.eps=0.1
```

Multi-attack example (single `attack` field with list syntax):

```bash
python -m deckard optimize --config-name experiment \
	'+attack=[{"attack_type":"art.attacks.evasion.FastGradientMethod","attack_params":{"eps":0.05},"attack_size":20,"alias":"fgm"},{"attack_type":"art.attacks.evasion.HopSkipJump","attack_params":{"max_iter":5},"attack_size":20,"alias":"hsj"}]'
```

In multi-attack runs, aliases are required and colliding metric keys are
suffixed with `_<alias>`.

For full documentation, use the docs navigation:

- Overview: {doc}`build_docs`
- Landing page: {doc}`index`
- Concepts: {doc}`concepts`

API entry points:

- API hub: {doc}`source/index`
- Package overview: {doc}`source/package`
- Module and extension map: {doc}`source/modules`

Notebook entry point:

- Notebook hub: {doc}`notebooks/index`

### Example Configs

Sklearn examples include reusable presets for attacks, scorers, and plots:

- Attacks: [examples/sklearn/config/attack](https://github.com/simplymathematics/deckard/tree/main/examples/sklearn/config/attack)
- Scorers: [examples/sklearn/config/score](https://github.com/simplymathematics/deckard/tree/main/examples/sklearn/config/score)
- Plots: [examples/sklearn/config/plot](https://github.com/simplymathematics/deckard/tree/main/examples/sklearn/config/plot)
