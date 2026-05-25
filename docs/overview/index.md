# Overview

## Purpose

The **Overview** section is the fastest way to understand how **deckard** runs
reproducible, multi-objective ML optimization and how to perform post-hoc
analysis from persisted experiment artifacts.

It is intended for:

- **Researchers** who need repeatable experiment workflows
- **Engineers** who need structured security, fairness, and privacy benchmarking
- **Contributors** extending data, model, attack, or scoring components

Core themes covered in this section:

- dependency setup for core and optional extension stacks
- high-level experiment orchestration via `ExperimentConfig`
- optimization and reproducibility via [Hydra](https://hydra.cc) and [Optuna](https://optuna.org)
- extension ecosystem mapping (frameworks and plugins)

Core module details now live in [API Reference](../api/modules.md).
The single core-runtime overview page in this section is:

1. [Experiment Workflow](experiment.md)

## Recommended Reading Order

The overview flow is intentionally short:

1. [Quickstart](quickstart.md)
2. [Summary](summary.md)
3. [Core Modules](core.md)
4. [Experiment Workflow](experiment.md)
5. [Scoring](scoring.md)
6. [Hydra](hydra.md)
7. [Optimization](optimize.md)
8. [DVC](dvc.md)
9. [Extensions](extensions/index.md)
10. [Installation](installation.md)
11. [Docker](docker.md)
12. [Developer Docs](../developers/index.md)
13. [Changelog](changelog.md)

## Navigation Notes

Each page in this section is designed to be independently useful, but together
they provide a complete map of:

- package architecture
- experiment composition
- reproducibility workflows
- extension and contribution patterns

Use the sidebar for direct navigation, or follow the recommended reading order
for a structured introduction.

```{toctree}
:maxdepth: 2
:hidden:

quickstart
summary
core
experiment
scoring
hydra
optimize
dvc
extensions/index
installation
docker
changelog
```
