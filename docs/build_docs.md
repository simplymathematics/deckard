# Developer Documentation
## Notebook Dependencies

All example notebooks in `docs/notebooks/*.ipynb` require the full set of optional dependencies for the relevant extension stacks. To run all notebooks without errors, install:

```bash
pip install -e '.[docs,test]'
```

This will install all core, extension, and documentation dependencies as defined in the `[project.optional-dependencies]` section of `pyproject.toml`, including:

- torch, torchvision, torchaudio
- anjana, pycanon
- fairlearn
- yellowbrick
- seaborn
- lifelines
- all Sphinx and Jupyter dependencies

If you only install `pip install -e '.[docs]'`, some notebooks may fail to run due to missing ML or plotting libraries. For full reproducibility, always use the full stack above.

# Layout
Sphinx documentation entry points:
- Landing page: [index](index)
- Software modules + extensions index: [api/modules](api/modules)
- Notebook guide index: [notebooks](notebooks)



## Directory Structure

```text
docs/
|- Makefile
|- make.bat
|- build_docs.md
|- conf.py
|- index.rst
|- build/
`- 
	|- modules.rst
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
- docs/notebooks/
	├── anjana.ipynb
	├── art_attacks.ipynb
	├── art_defenses.ipynb
	├── build
	├── deckard.ipynb
	├── deckard.log
	├── detector.ipynb
	├── error.log
	├── fairlearn.ipynb
	├── hydra.ipynb
	├── index.md
	├── lifelines.ipynb
	├── pytorch.ipynb
	├── seaborn.ipynb
	├── sklearn.ipynb
	└── yellowbrick.ipynb

Related runnable example configs live under:

- [examples/sklearn/config/attack](../examples/sklearn/config/attack)
- [examples/sklearn/config/score](../examples/sklearn/config/score)
- [examples/sklearn/config/plot](../examples/sklearn/config/plot)
```


## Prerequisites

From the repository root, install documentation dependencies with:

```bash
pip install -e '.[docs]'
```

This installs the full documentation stack as defined in the `[project.optional-dependencies] docs` section of [pyproject.toml](../pyproject.toml):

**Documentation dependencies:**

- sphinx
- myst-parser
- myst-nb
- pydata-sphinx-theme >= 0.14
- sphinx-autodoc-typehints
- sphinx-rtd-theme
- sphinx-copybutton
- sphinx-design
- sphinx-togglebutton
- jupyterlab
- ipykernel
- sphinxcontrib-bibtex
- roman  # Required for Sphinx LaTeX builder

All of these will be installed automatically with the above pip command.


## Theme Version Requirement

The documentation navigation header and sidebar require `pydata-sphinx-theme >= 0.14` for proper grouping and header support. This is pinned in the `[project.optional-dependencies] docs = [...]` section of `pyproject.toml`.
If you see navigation issues, upgrade with:

	pip install -U pydata-sphinx-theme

## Render the Docs

From the [docs](./index.rst) directory:

```bash
make html
```

Or directly with Sphinx:

```bash
sphinx-build -b html . build/html -E
```

Generated site output is written to `docs/build/html/index.html`.

## Live Preview

Optional: install sphinx-autobuild and run a live docs server.

```bash
pip install sphinx-autobuild
```

From the [docs](./index.rst) directory:

```bash
sphinx-autobuild . build/html
```

Default preview URL: http://127.0.0.1:8000
