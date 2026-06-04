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

## Setup

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

- `pyenv install 3.10` downloads and installs the newest Python version
compatible with 3.10.
- `pyenv virtualenv 3.10 env` creates a new virtual environment named `env`
  using Python 3.10.
- `pyenv activate env` activates the `env` environment, so all Python commands
  use this isolated setup.

To set up a virtual environment using Python's built-in `venv`, run:

```bash
python3 -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
```

- `python3 -m venv env` creates a new virtual environment named `env`.
- `source env/bin/activate` (or `env\Scripts\activate` on Windows) activates the
  environment so all Python commands use this isolated setup.

This command switches your shell to use the `env` environment, ensuring all
Python packages are installed locally within it.

### 3. Install Dependencies

```bash
pip install  .
```

Optional dependency stacks (install as needed):

```bash

pip install -e '.[torch]'
pip install -e '.[fairlearn]'
pip install -e '.[lifelines]'
pip install -e '.[seaborn]'
pip install -e '.[yellowbrick]'
pip install -e '.[anjana]'
pip install -e '.[transformers]'
pip install -e '.[test]'
pip install -e '.[docs]'
```

For a full extension stack in one environment:

```bash
pip install -e '.[all]'
```

## Repository layout

```text
.
├── build <- contains build files for the `deckard` executable
├── scripts/ <- Script for testing and developing the package
├── deckard <- Source directory
├── docs/developers/index.md <- Documentation for developers
├── Dockerfile <- A docker environment for testing and deployment
├── docs <- The documentation
├── examples <- Examples for each framework and optional extensions.
├── LICENSE  <- The software license file
├── docs/notebooks <- Examples, but as Jupyter Notebooks
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

Optimization-first run flow:

1. Define scorers/objectives (see [Scoring](scoring)).
1. Compose configs with [Hydra](https://hydra.cc).
1. Run optimization with [Optuna](https://optuna.org).
1. Persist outputs through [File API](/api/file/index).
1. Run post-hoc analysis through [Layers API](/api/layers/index).

Quick example:

```bash
python -m deckard optimize --config-name experiment \
  data.name=make_classification \
  model.name=sklearn.ensemble.RandomForestClassifier \
    attack.name=art.attacks.evasion.FastGradientMethod \
    attack.attack_params.eps=0.1
```

Multi-attack example (single `attack` field with list syntax):

```bash
python -m deckard optimize --config-name experiment \
    '+attack=[{"name":"art.attacks.evasion.FastGradientMethod","attack_params":{"eps":0.05},"attack_size":20,"alias":"fgm"},{"name":"art.attacks.evasion.HopSkipJump","attack_params":{"max_iter":5},"attack_size":20,"alias":"hsj"}]'
```

In multi-attack runs, aliases are required and colliding metric keys are
suffixed with `_<alias>`.

For full documentation, use the docs navigation:

- Developer docs: {doc}`../developers/index`
- Landing page: {doc}`/index`
- Package overview: {doc}`/api/modules`

Notebook entry point:

- Notebook hub: {doc}`/notebooks/index`

Post-hoc layer entry points:

- CLI Functions: {doc}`/api/layers/index`
- Plot API: {doc}`/api/plot/index`

### Example Configs

Full end-to-end pipeline examples are available for each of the supported frameworks here:

- sklearn: [examples/sklearn/config/](https://github.com/simplymathematics/deckard/tree/main/examples/sklearn/config/attack)
- Pytorch: [examples/pytorch/config/](https://github.com/simplymathematics/deckard/tree/main/examples/pytorch/config/attack)
- Transformers: [examples/trasnformers/config/](https://github.com/simplymathematics/deckard/tree/main/examples/sklearn/config/attack)
