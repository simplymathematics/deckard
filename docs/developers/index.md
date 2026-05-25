# Developer Documentation Index

Welcome to the Deckard developer documentation.
This section contains all design docs, architectural standards, and contributor
guidelines.
All content here is intended for developers extending, maintaining, or
integrating with Deckard.

Use this page as the main contributor entry point for setup, testing,
documentation, and architecture references.

## Development Workflow

Deckard is a Python package for declarative AI experimentation, evaluation,
and verification. The project uses:

- [`setuptools`](https://setuptools.pypa.io) build system
- `pyproject.toml` for dependency and tool configuration
- optional extras for modular installs
- CLI entrypoint via `deckard`

### Dependency Model

Core dependencies are defined in `[project].dependencies` in `pyproject.toml`
and are always installed.

Optional dependency groups are defined in `[project.optional-dependencies]`.

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

Other available extras include
[`lifelines`](../api/lifelines),
[`anjana`](../api/anjana),
[`yellowbrick`](../api/yellowbrick),
[`seaborn`](../api/seaborn),
`datasets`, and `lint`.

### Contributor Workflow

Typical contributor loop:

1. Identify the pipeline stage affected (`data`, `model`, `attack`, `score`,
	 or `experiment`).
2. Update implementation and associated declarations/config wiring.
3. Add or update tests for behavior changes.
4. Update notebooks/docs when the user-facing behavior changes.
5. Re-run focused workflows (tests, docs build, notebook stage) before merge.

### Testing and Validation

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

Tools used:

- [`pre-commit`](https://pre-commit.com) — multi-language pre-commit hook framework
- [`flake8`](https://flake8.pycqa.org) — Python style and lint checker
- [`black`](https://black.readthedocs.io) — opinionated Python code formatter
- [`mypy`](https://mypy.readthedocs.io) — static type checker for Python
- [`Hydra`](https://hydra.cc) — hierarchical configuration composition and overrides
- [`Optuna`](https://optuna.org) — hyperparameter optimization and pruning workflows
- [`Adversarial Robustness Toolbox
	(ART)`](https://adversarial-robustness-toolbox.org/) — adversarial attacks and
	defenses
- [`DVC`](https://dvc.org) — data and artifact versioning for reproducible pipelines

### Coverage Script

Use `scripts/coverage.sh` for a unified test and coverage workflow.

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

### Development Guidelines

- Prefer shared utility helpers over duplicated conversion and normalization logic.
- Keep configuration behavior deterministic and explicit rather than relying on
	implicit fallback behavior.
- Keep metric naming stable, especially for multi-attack and extension metrics,
	to preserve downstream report compatibility.
- When modifying notebook-driven workflows, validate with the corresponding
	[DVC](https://dvc.org) stage.

### Documentation Responsibilities

Code changes that affect behavior should include doc updates in at least one of
the following:

- API page updates when signatures or semantics change.
- Notebook narrative updates when workflow interpretation changes.
- Overview updates when architecture or user entry points change.

### Documentation Build Workflow

Install docs dependencies:

```bash
pip install -e .[docs]
```

Build docs from the `docs/` directory:

```bash
make html
```

For [DVC](https://dvc.org) notebook caching behavior in local and CI docs builds,
see [DVC Cache Setup Summary](actionscache).

## Contents

- [Design Principles](designprinciples)
- [GitHub Actions Workflows](workflows)
- [Refactor Plan](plan)
- [Config Declaration Architecture](declarations)
- [Naming Conventions](naming)
- [Mixin and Plugin Rules](plugins)
- [Data Design and Contract](data)
- [Model Design and Contract](model)
- [Attack Design and Contract](attack)
- [Experiment Design and Contract](experiment)
- [Plugin and Hook Execution Reference](hooks)
- [Persistence and Runtime State Contract](persistence)
- [Score Serialization Contract](score)
- [Optimization Runtime Contract](optimization)
- [Optimize Developer Guide](optimize)
- [Hydra and Optuna Orchestration Contract](hydra)
- [Pruning Runtime Contract](pruning)
- [DVC Pipeline Autogeneration Spec](dvc)
- [Plugin Runtime Migration Guardrails](migration)
- [Docstring Standard](docstrings)
- [GH Actions Cache Setup](actionscache)

```{toctree}
:maxdepth: 2
:hidden:

design
workflows
plan
declarations
naming
plugins
data
model
attack
experiment
hooks
persistence
score
optimization
optimize
hydra
pruning
dvc
migration
docstrings
actionscache
```

For user-facing documentation, see the [Overview](../overview/index) and [Notebooks](../notebooks/index).

______________________________________________________________________

**Quick links:**

- [API Reference](../api/modules)
- [Notebook Index](../notebooks/index)
- [Data Design and Contract](data)
- [Model Design and Contract](model)
- [Attack Design and Contract](attack)
- [Experiment Design and Contract](experiment)
- [Plugin and Hook Execution Reference](hooks)
- [Persistence and Runtime State Contract](persistence)
- [Score Serialization Contract](score)
- [Optimization Runtime Contract](optimization)
- [Optimize Developer Guide](optimize)
- [Hydra and Optuna Orchestration Contract](hydra)
- [Pruning Runtime Contract](pruning)
- [DVC Pipeline Autogeneration Spec](dvc)
- [Plugin Runtime Migration Guardrails](migration)
