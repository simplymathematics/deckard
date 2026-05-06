
# Documentation
This directory contains the Sphinx documentation for Deckard, including API
references and usage guides for core modules such as:

- [deckard.data](../deckard/data/__init__.py)
- [deckard.model](../deckard/model/__init__.py)
- [deckard.attack](../deckard/attack/__init__.py)
- [deckard.detector](../deckard/detector/__init__.py)
- [deckard.layers](../deckard/layers/__init__.py)
- [deckard.score](../deckard/score/__init__.py)
- [deckard.experiment](../deckard/experiment/__init__.py)
- [deckard.plot](../deckard/plot/__init__.py)

## Directory Structure

```text
docs/
|- Makefile
|- make.bat
|- README.md
|- build/
`- source/
	|- package.rst
	|- conf.py
	|- index.rst
	|- modules.rst
	|- attack.rst
	|- data.rst
	|- detector.rst
	|- experiment.rst
	|- file.rst
	|- layers.rst
	|- lifelines.rst
	|- model.rst
	|- pytorch.rst
	|- plot.rst
	|- score.rst
	|- seaborn.rst
	`- utils.rst

Related runnable example configs live under:

- [examples/sklearn/config/attack](../examples/sklearn/config/attack)
- [examples/sklearn/config/score](../examples/sklearn/config/score)
- [examples/sklearn/config/plot](../examples/sklearn/config/plot)
```

## Prerequisites

From the repository root, install docs dependencies with:

```bash
pip install -e '.[docs]'
```

This installs the docs extras defined in [pyproject.toml](../pyproject.toml),
including Sphinx, sphinx-rtd-theme, sphinx-autodoc-typehints, myst-parser,
and nbsphinx.

## Build The Docs

From the [docs](.) directory:

```bash
make html
```

Or directly with Sphinx:

```bash
sphinx-build -b html source build/html -E
```

Generated site output is written to [docs/build/html](build/html).

## Live Preview

Optional: install sphinx-autobuild and run a live docs server.

```bash
pip install sphinx-autobuild
```

From the [docs](.) directory:

```bash
sphinx-autobuild source build/html
```

Default preview URL: http://127.0.0.1:8000
