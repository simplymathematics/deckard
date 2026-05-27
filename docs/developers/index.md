# Developer Documentation Index

Welcome to the Deckard developer documentation.
This section contains all design docs, architectural standards, and contributor
guidelines.
All content here is intended for developers extending, maintaining, or
integrating with Deckard.

Use this page as the main contributor entry point for setup, testing,
documentation, and architecture references.

## Docs Map

Use this map to choose where to read first:

- User-facing behavior and usage: {doc}`../overview/index` and {doc}`../api/modules`.
- Developer architecture and runtime contracts: {doc}`design`, {doc}`orchestration`, {doc}`canon_runtime`.
- Extension authoring and execution surfaces: {doc}`plugins`, {doc}`hooks`, {doc}`mixins`.

## Core Runtime Docs

- {doc}`Data Design and Contract <data>`
- {doc}`Model Design and Contract <model>`
- {doc}`Attack Design and Contract <attack>`
- {doc}`Detector Design and Contract <detector>`
- {doc}`Experiment Design and Contract <experiment>`
- {doc}`Score Serialization Contract <score>`
- {doc}`Plot Design and Contract <plot>`
- {doc}`Orchestration Guide <orchestration>`
- {doc}`Canon Runtime Execution Guide <canon_runtime>`

## Documentation Standards

- {doc}`Developer Page Template <template>`
- {doc}`Developer to API Parity Map <parity>`

## Framework Integration Docs

- {doc}`../api/pytorch`
- {doc}`Hydra and Optuna Orchestration Contract <hydra>`

## Plugin Integration Docs

- {doc}`Mixin and Plugin Rules <plugins>`
- {doc}`Plugin and Hook Execution Reference <hooks>`
- {doc}`../api/fairlearn`
- {doc}`../api/lifelines`
- {doc}`../api/anjana`
- {doc}`../api/seaborn`
- {doc}`../api/yellowbrick`

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

- {doc}`Design Principles <design>`
- {doc}`GitHub Actions Workflows <workflows>`
- {doc}`Refactor Plan <refactor_plan>`
- {doc}`Config Declaration Architecture <declarations>`
- {doc}`Naming Conventions <naming>`
- {doc}`Mixin and Plugin Rules <plugins>`
- {doc}`Data Design and Contract <data>`
- {doc}`Model Design and Contract <model>`
- {doc}`Attack Design and Contract <attack>`
- {doc}`Experiment Design and Contract <experiment>`
- {doc}`Plugin and Hook Execution Reference <hooks>`
- {doc}`Persistence and Runtime State Contract <persistence>`
- {doc}`Score Serialization Contract <score>`
- {doc}`Optimization Runtime Contract <optimization>`
- {doc}`Optimize Developer Guide <optimize>`
- {doc}`Hydra and Optuna Orchestration Contract <hydra>`
- {doc}`Pruning Runtime Contract <pruning>`
- {doc}`DVC Pipeline Autogeneration Spec <dvc>`
- {doc}`Plugin Runtime Migration Guardrails <migration>`
- {doc}`Docstring Standard <docstrings>`
- {doc}`Config Class Contract <configs>`
- {doc}`Mixin Class Contract <mixins>`
- {doc}`Plugin Rules and Capabilities <plugins>`
- {doc}`Sampler Class Contract <samplers>`
- {doc}`Pipeline Class Contract <pipelines>`
- {doc}`Trainer Class Contract <trainers>`
- {doc}`Defense Class Contract <defenses>`
- {doc}`Scorer Class Contract <scorers>`
- {doc}`Orchestration Guide <orchestration>`
- {doc}`Canon Runtime Execution Guide <canon_runtime>`
- {doc}`Detector Design and Contract <detector>`
- {doc}`Plot Design and Contract <plot>`
- {doc}`Developer Page Template <template>`
- {doc}`Developer to API Parity Map <parity>`
- {doc}`Docs Refactor Checklist <new_docs>`
- {doc}`GH Actions Cache Setup <actionscache>`

```{toctree}
:maxdepth: 2
:hidden:

design
workflows
refactor_plan
declarations
naming
plugins
data
model
attack
detector
experiment
hooks
persistence
score
plot
optimization
optimize
hydra
pruning
dvc
migration
docstrings
configs
mixins
samplers
pipelines
trainers
defenses
scorers
orchestration
canon_runtime
template
parity
new_docs
actionscache
```

For user-facing documentation, see {doc}`../overview/index` and {doc}`../notebooks/index`.

______________________________________________________________________

**Quick links:**

- {doc}`API Reference <../api/modules>`
- {doc}`Notebook Index <../notebooks/index>`
- {doc}`Data Design and Contract <data>`
- {doc}`Model Design and Contract <model>`
- {doc}`Attack Design and Contract <attack>`
- {doc}`Experiment Design and Contract <experiment>`
- {doc}`Plugin and Hook Execution Reference <hooks>`
- {doc}`Persistence and Runtime State Contract <persistence>`
- {doc}`Score Serialization Contract <score>`
- {doc}`Optimization Runtime Contract <optimization>`
- {doc}`Optimize Developer Guide <optimize>`
- {doc}`Hydra and Optuna Orchestration Contract <hydra>`
- {doc}`Pruning Runtime Contract <pruning>`
- {doc}`DVC Pipeline Autogeneration Spec <dvc>`
- {doc}`Plugin Runtime Migration Guardrails <migration>`
- {doc}`Canon Runtime Execution Guide <canon_runtime>`
- {doc}`Detector Design and Contract <detector>`
- {doc}`Plot Design and Contract <plot>`
- {doc}`Developer Page Template <template>`
- {doc}`Developer to API Parity Map <parity>`
- {doc}`Docs Refactor Checklist <new_docs>`
