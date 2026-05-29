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
- configuration-driven orchestration for full ML evaluation pipelines
- repeatable execution with explicit run metadata
- [PyTorch](../api/pytorch): framework-specific runtime and integration APIs.
- [Anjana](../api/anjana): privacy and anonymization integration APIs.
- [Fairlearn](../api/fairlearn): fairness metrics and evaluation integration APIs.
- [Lifelines](../api/lifelines): survival-analysis integration APIs.
- [Seaborn](../api/seaborn): plotting and visualization integration APIs.
- [Yellowbrick](../api/yellowbrick): model diagnostics and visualization integration APIs.

## Suggested Reading Paths

Choose one path based on your goal:

1. First steps: [Quickstart](quickstart) -> [Core Modules](core) -> [Experiment Workflow](./experiment.md) -> [Scoring](scoring) -> [API Reference](../api/modules)
2. Tuning and optimization workflows: [Quickstart](quickstart) -> [Hydra](hydra) -> [Optimization](optimize) -> [DVC](dvc)
3. Extending the software: [Core Modules](core) -> [Extensions](extensions/index) -> [Developer Docs](../developers/index)

## How This Section Is Organized

This page follows the exact structure of the hidden toctree below. If you read
in that order, you move from conceptual orientation to execution details,
extension surfaces, and release/operations context.

If you want only one API map page, use [API Reference](../api/modules).

## Core Overview

These pages define the core runtime path:

1. [Quickstart](quickstart): practical first steps and command paths.
2. [Core Modules](core): main configuration surfaces and object model.
3. [Experiment Workflow](./experiment.md): end-to-end orchestration lifecycle.
4. [Scoring](scoring): objective outputs and metric handling.
5. [Workflow Flowcharts](flowcharts): visual runtime path and control flow.

## CLI

These pages cover run composition and reproducible execution:

1. [Hydra](hydra): configuration composition and run/multirun behavior.
2. [Optimization](optimize): Optuna-driven objective search workflows.
3. [DVC](dvc): pipeline and artifact tracking conventions.

## Plugins And Frameworks

Use [Extensions](extensions/index) as the entrypoint for framework and plugin
documentation.

Direct API integration pages:

- [PyTorch](../api/pytorch)
- [Anjana](../api/anjana)
- [Fairlearn](../api/fairlearn)
- [Lifelines](../api/lifelines)
- [Seaborn](../api/seaborn)
- [Yellowbrick](../api/yellowbrick)

## Software Notes

These pages cover environment and release-facing references:

1. [Installation](installation)
2. [Docker](docker)
3. [Changelog](changelog)

```{toctree}
:maxdepth: 2
:hidden:
quickstart
core
experiment
scoring
flowcharts
hydra
optimize
dvc
extensions/index
installation
docker
changelog
```
