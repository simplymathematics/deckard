# deckard

deckard is a declarative ML evaluation framework for reproducible experiments
across:

- data preparation and sampling
- model training and evaluation
- adversarial attacks and defenses
- fairness and privacy scoring
- experiment optimization and plotting

deckard uses Hydra/OmegaConf for configuration composition and supports Optuna
for multirun optimization.

## Quickstart

### 1. Clone

```bash
git clone git@github.com:simplymathematics/deckard.git
cd deckard
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install deckard

```bash
python -m pip install -e .
```

For full installation options and platform notes, see:

- [Installation guide](docs/overview/installation.md)
- [Developer setup](docs/overview/development.md)

## Repository layout:

```
.
├── build <- exists after installation, contains the build files for the `deckard` executable.
├── scripts/coverage.sh <- Script for tracking test coverage and test timing measurements.
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

## Documentation

Canonical documentation lives in `docs/` and is built with Sphinx.

- [Documentation overview](docs/overview/build_docs.md)
- [Installation](docs/overview/installation.md)
- [Development](docs/overview/development.md)
- [Concepts chapter](docs/concepts.md)
- [API hub](docs/source/index.md)
- [Package overview](docs/source/modules.md)
- [Module and extension map](docs/source/modules.md)
- [Notebook hub](docs/notebooks/index.md)


## Citation:
If you find this software useful please cite us:


```bibtex
@software{deckard,
  author       = {deckard team},
  title        = {deckard},
  year         = {2026},
  url          = {https://github.com/simplymathematics/deckard},
  version      = {v.98},
  note         = {GitHub repository}
  author+an    = {1=orcid:0000-0002-1277-9811; 2=orcid:0000-0002-0751-9695}
}
```
