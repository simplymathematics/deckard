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
- high-level experiment orchestration via {class}`deckard.experiment.ExperimentConfig`
- optimization and reproducibility via [Hydra](https://hydra.cc) and [Optuna](https://optuna.org)
- extension ecosystem mapping (frameworks and plugins)

Core module details now live in [API Reference](../api/modules).
The single core-runtime overview page in this section is:

1. [Experiment Workflow](./experiment)
2. [Workflow Flowcharts](flowcharts)

## Recommended Reading Order

The overview flow is intentionally short:

1. [Quickstart](quickstart)
2. [Summary](summary)
3. [Core Modules](core)
4. [Experiment Workflow](./experiment)
5. [Workflow Flowcharts](flowcharts)
6. [Scoring](scoring)
7. [Hydra](hydra)
8. [Optimization](optimize)
9. [DVC](dvc)
10. [Extensions](extensions/index)
11. [Installation](installation)
12. [Docker](docker)
13. [Developer Docs](../developers/index)
14. [Changelog](changelog)

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
flowcharts
scoring
hydra
optimize
dvc
extensions/index
installation
docker
changelog
```
