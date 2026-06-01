# Testing Standards and Escalation Map

This page is the canonical source for contributor testing standards, fail-fast
escalation order, and CI mapping.

## Local Validation Baseline

Install development test tooling:

```bash
pip install -e .[test]
```

Before opening a PR, ensure:

- tests pass
- formatting is correct
- linting passes
- type checks pass

### Pre-commit Hook Requirement

Lint tools do not run automatically on `git commit` unless the local git hook is
installed in your clone.

Install and verify hook wiring:

```bash
pre-commit install
ls -la .git/hooks/pre-commit
```

Expected outcome:

- `.git/hooks/pre-commit` exists and is executable.
- The hook references `.pre-commit-config.yaml`.

If hooks appear to be skipped:

- Check for `git commit --no-verify` usage (this bypasses hooks).
- Re-run `pre-commit install` after cloning/reinitializing the repository.
- Confirm git is not using a custom hooks path:

```bash
git config --get core.hooksPath
```

If this prints a non-empty path, ensure pre-commit is installed into that hooks
directory as well.

If CI fails, install broader tooling and run checks locally:

```bash
pip install -e ".[test,lint,docs]"
pre-commit install
bash scripts/coverage.sh
flake8 deckard/
black deckard/
mypy deckard/
./.venv/bin/pymarkdown scan $(find . -type f -name '*.md' -not -path './.venv/*' -not -path './build/*' -not -path './docs/build/*')
```

Markdown linting uses PyMarkdown with `MD013` (line length) disabled in
`pyproject.toml` to avoid noisy failures on long URLs and table rows.

Tools used:

- [`pre-commit`](https://pre-commit.com) — multi-language pre-commit hook framework
- [`flake8`](https://flake8.pycqa.org) — Python style and lint checker
- [`black`](https://black.readthedocs.io) — opinionated Python code formatter
- [`mypy`](https://mypy.readthedocs.io) — static type checker for Python
- [`PyMarkdown`](https://pymarkdown.readthedocs.io) — Markdown linting and style checks
- [`Hydra`](https://hydra.cc) — hierarchical configuration composition and overrides
- [`Optuna`](https://optuna.org) — hyperparameter optimization and pruning workflows
- [`Adversarial Robustness Toolbox (ART)`](https://adversarial-robustness-toolbox.org/) — adversarial attacks and defenses
- [`DVC`](https://dvc.org) — data and artifact versioning for reproducible pipelines

## Coverage Script

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

## Canonical Refactor Testing Standards

- Test criteria per check mark: unit tests for touched files.
- Final testing: fail fast over each folder, updating tests or code as needed.
- Notebook gating is stage-scoped: run `dvc repro --force <stage>` one stage at
  a time inside `docs/notebooks`, fix the failing notebook, then rerun that same
  stage before moving to the next one.
- Final acceptance criteria: `scripts/coverage.sh` passes -> notebook DVC stages
  pass one at a time -> docs build passes with notebook execution disabled ->
  final workflow validation passes.

## Notebook Gating Canon

Notebook validation is a local gate, not a bulk `dvc repro --force` sweep.

Canonical workflow:

```bash
cd docs/notebooks
dvc repro --force notebook_seaborn
dvc repro --force notebook_yellowbrick
dvc repro --force notebook_detector
# ...continue one stage at a time until the touched / gated notebook stages pass
```

Rules:

- Run one notebook stage at a time with `dvc repro --force <stage>`.
- If a stage fails, stop, fix that notebook or its supporting runtime surface,
  and rerun the same stage before advancing.
- Do not treat `dvc repro --force` without a stage name as the canonical gating
  path for notebook validation.
- Prefer the stage names reported by `dvc stage list` in `docs/notebooks`.

Common notebook stages include:

- `notebook_seaborn`
- `notebook_yellowbrick`
- `notebook_detector`
- `notebook_art_attacks`
- `notebook_art_defenses`
- `notebook_sklearn`
- `notebook_pytorch`
- `notebook_hydra`
- `notebook_deckard`
- `notebook_scoring`
- `notebook_dvc`
- `notebook_optuna`
- `notebook_lifelines`
- `notebook_artifacts`
- `notebook_fairlearn`
- `notebook_anjana`

## Docs Build Canon

After notebook gating passes, build docs with notebook execution disabled.
The repository Makefile already exposes the canonical Sphinx override through
`SPHINX_NB_OFF ?= -D nb_execution_mode=off`.

Run from `docs/`:

```bash
make html SPHINX_NB_OFF='-D nb_execution_mode=off'
```

This is the required local docs-build check before the final workflow
validation stage.

## Canon Fail-Fast Escalation Map

Derived from the `docs/api/modules` toctree.

| Escalation Order | Test Folder | Primary Toctree Entry |
| --- | --- | --- |
| 1 | `test/test_data` | `api/data` (+ `api/sample`, `api/pipeline`) |
| 2 | `test/test_model` | `api/model` (+ `api/train`, `api/defend`) |
| 3 | `test/test_attack` | `api/attack` |
| 4 | `test/test_detector` | `api/detector` |
| 5 | `test/test_experiment` | `api/experiment` |
| 6 | `test/test_score` | `api/score` |
| 7 | `test/test_plot` | `api/plot` |
| 8 | `test/test_file` | `api/file` |
| 9 | `test/test_artifacts` | `api/artifacts` |
| 10 | `test/test_utils` | `api/utils` |
| 11 | `test/test_frameworks` | `api/frameworks/index` |
| 12 | `test/test_plugins` | `api/plugins/index` |
| 13 | `test/test_layers` | `api/layers` |
| 14 | `test/test_package` | `api/layers` (CLI/entrypoint surface) |
| 15 | `test/test_integration` | `api/experiment` (cross-cutting end-to-end gate) |

## CI Mapping Rule

- Use one fail-fast CI job per row in escalation order.
- Run `test/test_integration` as the final cross-cutting gate.
