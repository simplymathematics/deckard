# Deckard Developer & Test Workflow

This document describes how to install, develop, test, and maintain the `deckard` project using the configuration defined in `pyproject.toml`.

---

# 1. Project Overview

`deckard` is a Python package for declarative AI experimentation, evaluation, and verification.

It is structured as a standard Python project using:

- setuptools build system
- `pyproject.toml` configuration
- optional dependency groups for modular installs
- a CLI entrypoint (`deckard`)

---

# 2. Installation

## Base installation

If you want to install this package, use:

```bash
pip install .
```

This installs the core package and all required dependencies.

---

## Development installation

If you intend to edit files locally, use editable mode:

```bash
pip install -e .
```

This ensures changes to source code are immediately reflected without reinstalling.

---

# 3. Dependency Model

## Core dependencies

Defined under:

```
[project]
dependencies = [...]
```

in `pyproject.toml`.

These include:

```
numpy
pandas
scikit-learn
hydra-core
optuna
matplotlib
dvc
```

These are always installed.

---

## Optional dependencies (extras)

Defined under:

```
[project.optional-dependencies]
```

These are modular dependency groups used for different functionality areas.

### Install all optional dependencies

```bash
pip install ".[all]"
```

Warning: this includes all experimental, ML, plotting, fairness, and deep learning dependencies. It significantly increases install size and build complexity.

---

### Recommended selective installs

Install only what you need:

```bash
pip install ".[test]"
pip install ".[docs]"
pip install ".[fairlearn]"
pip install ".[torch]"
```

Each group enables a different subsystem:

- `test`: testing + linting + CI tools
- `docs`: documentation build system
- `fairlearn`: fairness metrics and evaluation
- `torch`: PyTorch-based experiments

---

# 4. Testing and Development Workflow

## Required installation for development

Before contributing, install test dependencies:

```bash
pip install -e .[test]
```

This ensures:
- pytest is available
- linting tools are installed
- pre-commit hooks can run

---

## Pre-PR requirements

Before submitting a pull request, ensure:

- tests pass
- formatting is correct
- linting passes
- type checks pass

---

## If CI fails after submission

Install full dev tooling:

```bash
pip install -e ".[test,lint,docs]"
pre-commit install
```

Then run:

```bash
bash coverage.sh
flake8 deckard/
black deckard/
mypy deckard/
```

Fix any reported issues before pushing updates.

---

# 5. Coverage and Test Execution Script

The project uses a unified script (`coverage.sh`) to run tests, collect coverage, and capture timing.

## Overview

This script:

- Runs pytest test suite
- Collects coverage data
- Generates coverage report
- Captures per-test timing
- Logs failures

All outputs are written to:

```
build/
```

---

## Configuration

Key variables:

- `OUT_DIR="build"`  
  Output directory for all artifacts

- `PKG="deckard"`  
  Package used for coverage analysis

- `TEST_DIR="${1:-test}"`  
  Directory containing tests  
  Defaults to `test` if not provided

---

## Parameter behavior

```
TEST_DIR="${1:-test}"
```

Means:

- Use first command-line argument if provided
- Otherwise default to `test`

---

## Output files

Generated artifacts:

- `build/coverage.txt` → coverage report
- `build/timing.txt` → test execution durations
- `build/error.log` → captured failures

---

## Error handling model

The function `run_and_capture`:

- captures stdout and stderr
- stores outputs in temporary buffers
- writes failures into `error.log`
- does not immediately terminate script on failure

This allows multiple steps to run even if one fails.

---

## Execution flow

### 1. Setup

```bash
mkdir -p "$OUT_DIR"
: > "$ERROR_LOG"
```

Ensures output directory exists and resets error log.

---

### 2. Test execution with coverage

Runs pytest with coverage enabled:

```bash
pytest "$TEST_DIR" --cov="$PKG" --cov-append -q
```

Coverage is accumulated across runs.

---

### 3. Coverage report generation

```bash
coverage report -m > "$OUT_DIR/coverage.txt"
```

Produces line-level coverage summary.

---

### 4. Timing collection

```bash
pytest "$TEST_DIR" --durations=0 -q > "$OUT_DIR/timing.txt"
```

Outputs per-test execution times.

---

### 5. Cleanup

Temporary artifacts are removed:

```bash
rm -f "$OUT_DIR/.tmp.out" "$OUT_DIR/.tmp.err"
```

---

### 6. Final failure check

If errors were recorded:

```
build/error.log
```

the script exits with failure status.

---

# 6. Documentation System

Documentation uses:

- Sphinx
- MyST Markdown
- nbsphinx (Jupyter notebook support)

---

## Install documentation dependencies

```bash
pip install -e .[docs]
```

---

## Build documentation

Standard build process is handled via Sphinx:

```bash
sphinx-build docs docs/_build
```

---

## Documentation source

Further documentation is located in:

```
docs/README.md
```

You can access it directly here:

[Documentation README](docs/README.md)

---
```