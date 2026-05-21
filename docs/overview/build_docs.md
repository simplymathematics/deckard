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

## Test CI Docs Workflow Locally

For a generic local GitHub Actions runner, use:

```bash
./scripts/test_workflow.sh --list
./scripts/test_workflow.sh --workflow compile-docs.yml --job docs
```

This runs any existing workflow via `act` with branch-aware payload generation.

To preview the exact command without running it:

```bash
./scripts/test_workflow.sh --workflow compile-docs.yml --job docs --ref refactor-squashed --dry-run
```

`act` runs require Docker and `act` (`brew install act`).

If Docker is installed but you see an error like `connect: no such file or directory` for `/var/run/docker.sock`, start a Docker engine first (Docker Desktop, `colima start`, or OrbStack), then rerun the script.

# Layout

Sphinx documentation entry points:

- Landing page: [index](../index)
- Software modules + extensions index: [api/modules](../api/modules)
- Notebook guide index: [notebooks/index](../notebooks/index)

## Directory Structure

```text
docs/
|- Makefile
|- make.bat
|- conf.py
|- index.md
|- overview/
|  |- index.md
|  |- build_docs.md
|  |- quickstart.md
|  |- summary.md
|  |- installation.md
|  |- development.md
|  |- docker.md
|  |- extensions.md
|  `- changelog.md
|- build/
`-
    |- modules.md
    |- index.md
    |- attack.md
    |- data.md
    |- detector.md
    |- experiment.md
    |- file.md
    |- layers.md
    |- lifelines.md
    |- model.md
    |- pytorch.md
    |- plot.md
    |- score.md
    |- seaborn.md
    `- utils.md
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

- [examples/sklearn/config/attack](../../examples/sklearn/config/attack)
- [examples/sklearn/config/score](../../examples/sklearn/config/score)
- [examples/sklearn/config/plot](../../examples/sklearn/config/plot)
```

## Prerequisites

From the repository root, install documentation dependencies with:

```bash
pip install -e '.[docs]'
```

This installs the full documentation stack as defined in the `[project.optional-dependencies] docs` section of [pyproject.toml](../../pyproject.toml):

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
- roman # Required for Sphinx LaTeX builder

All of these will be installed automatically with the above pip command.

## Theme Version Requirement

The documentation navigation header and sidebar require `pydata-sphinx-theme >= 0.14` for proper grouping and header support. This is pinned in the `[project.optional-dependencies] docs = [...]` section of `pyproject.toml`.
If you see navigation issues, upgrade with:

```
pip install -U pydata-sphinx-theme
```

## Render the Docs

From the [docs](../index.md) directory:

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

From the [docs](../index.md) directory:

```bash
make autobuild
```

Default preview URL: http://127.0.0.1:8000

The `autobuild` target integrates the new DVC-backed notebook workflow:

- `make notebooks` runs first as a `sphinx-autobuild` pre-build step.
- DVC decides which notebook stages actually need to rerun based on `docs/notebooks/dvc.yaml`.
- Sphinx then renders with `nb_execution_mode=off`, so MyST-NB does not execute notebooks a second time.

This gives you dependency tracking, cache reuse, and reproducible notebook outputs while keeping live preview updates.

Useful variants:

```bash
# Rebuild only one notebook stage before each live refresh
make autobuild DVC_STAGE=notebook_pytorch

# Choose a different preview port if 8000 is already in use
make autobuild AUTOBUILD_PORT=8001

# Pass extra DVC flags through to the pre-build step
make autobuild DVC_REPRO_ARGS="--force"
```

If you prefer to run `sphinx-autobuild` directly, use the equivalent command:

```bash
sphinx-autobuild \
    -j auto \
    --port 8000 \
    --pre-build "make notebooks" \
    --watch ../deckard \
    --watch ../examples \
    --watch notebooks/dvc.yaml \
    -D nb_execution_mode=off \
    . build/html
```
