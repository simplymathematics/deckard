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

## Phase 4 Testing Standards

- Test criteria per check mark: unit tests for touched files.
- Final testing: fail fast over each folder, updating tests or code as needed.
- Final acceptance criteria: `scripts/coverage.sh` passes -> `dvc repro --force`
  in `docs/notebooks` -> docs build passes -> `build_docs` workflow passes.

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
