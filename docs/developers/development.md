# Development Workflow

This page is the canonical developer workflow for installing, testing,
maintaining, and documenting Deckard. Use it as the single entry point for
contributor setup and verification.

## Project Overview

Deckard is a Python package for declarative AI experimentation, evaluation,
and verification. The project uses:

- `setuptools` build system
- `pyproject.toml` for dependency and tool configuration
- optional extras for modular installs
- CLI entrypoint via `deckard`

## Installation

Base install:

```bash
pip install .
```

Editable install for local development:

```bash
pip install -e .
```

## Dependency Model

Core dependencies are defined in `[project].dependencies` in
`pyproject.toml` and are always installed.

Optional dependency groups are defined in
`[project.optional-dependencies]`.

Install all optional dependencies:

```bash
pip install ".[all]"
```

Recommended selective installs:

```bash
pip install ".[test]"
pip install ".[docs]"
pip install ".[fairlearn]"
pip install ".[torch]"
```

Other available extras include `lifelines`, `anjana`, `yellowbrick`,
`seaborn`, `datasets`, and `lint`.

## Contributor Workflow

Typical contributor loop:

1. Identify the pipeline stage affected (`data`, `model`, `attack`, `score`,
   or `experiment`).
2. Update implementation and associated declarations/config wiring.
3. Add or update tests for behavior changes.
4. Update notebooks/docs when the user-facing behavior changes.
5. Re-run focused workflows (tests, docs build, notebook stage) before merge.

## Testing and Validation

Install development test tooling:

```bash
pip install -e .[test]
```

Before opening a PR, ensure:

- tests pass
- formatting is correct
- linting passes
- type checks pass

If CI fails, install broader tooling and run checks locally:

```bash
pip install -e ".[test,lint,docs]"
pre-commit install
bash scripts/coverage.sh
flake8 deckard/
black deckard/
mypy deckard/
```

## Coverage Script

Use `scripts/coverage.sh` for a unified test+coverage workflow.

Generated outputs in `build/`:

- `build/coverage.txt`
- `build/timing.txt`
- `build/error.log`

Run against full suite (default):

```bash
bash scripts/coverage.sh
```

Run against a specific test subtree:

```bash
bash scripts/coverage.sh test/test_layers
```

The script captures errors without failing immediately, then exits non-zero if
any step fails.

## Core Modules

- {doc}`/api/data`
- {doc}`/api/model`
- {doc}`/api/attack`
- {doc}`/api/detector`
- {doc}`/api/experiment`
- {doc}`/api/score`
- {doc}`/api/plot`
- {doc}`/api/layers`
- {doc}`/api/file`
- {doc}`/api/utils`

Extension documentation:

- {doc}`/api/pytorch`
- {doc}`/api/anjana`
- {doc}`/api/lifelines`
- {doc}`/api/seaborn`
- {doc}`/api/yellowbrick`

## Development Guidelines

- Prefer shared utility helpers over duplicated conversion/normalization logic.
- Keep configuration behavior deterministic and explicit rather than relying on
	implicit fallback behavior.
- Keep metric naming stable, especially for multi-attack and extension metrics,
	to preserve downstream report compatibility.
- When modifying notebook-driven workflows, validate with the corresponding DVC
	stage.

## Documentation Responsibilities

Code changes that affect behavior should include doc updates in at least one of
the following:

- API page updates when signatures or semantics change.
- Notebook narrative updates when workflow interpretation changes.
- Overview updates when architecture or user entry points change.

## Documentation Build Workflow

Install docs dependencies:

```bash
pip install -e .[docs]
```

Build docs from the `docs/` directory:

```bash
make html
```

For DVC notebook caching behavior in local/CI docs builds, see
[DVC Cache Setup Summary](gh_actions_cache.md) and
[DVC Cache and Notebook Workflow Guide](gh_actions_cache.md).
