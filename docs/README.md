
# Documentation
This directory contains the Sphinx documentation for Deckard, including API
references and usage guides for core modules such as:

- [deckard.data](../deckard/data/__init__.py)
- [deckard.model](../deckard/model/__init__.py)
- [deckard.attack](../deckard/attack/__init__.py)
- [deckard.layers](../deckard/layers/__init__.py)
- [deckard.score](../deckard/score.py)

## Directory Structure

```text
docs/
|- Makefile
|- make.bat
|- README.md
|- build/
`- source/
	|- conf.py
	|- index.rst
	|- modules.rst
	|- attack.rst
	|- data.rst
	|- experiment.rst
	|- file.rst
	|- layers.rst
	|- model.rst
	|- plot.rst
	|- score.rst
	`- utils.rst
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
