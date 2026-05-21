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
- concise composition of base runtime configs
	([DataConfig](../api/data), [ModelConfig](../api/model),
	[AttackConfig](../api/attack), [DetectorConfig](../api/detector),
	[ExperimentConfig](../api/experiment), [FileConfig](../api/file))
- scoring and persistence as first-class optimization outputs
- multi-objective optimization via [Optuna](https://optuna.org) and
	[Hydra](https://hydra.cc)
- post-hoc evaluation pipelines via [Layers](../api/layers)

## Recommended Reading Order

The following pages are ordered for progressive onboarding:


1. [Quickstart](quickstart.md)
2. [Summary](summary.md)
3. [Optimization](optimize.md)
4. [Extensions](extensions.md)
5. [Installation](installation.md)
6. [Scoring](scoring.md)
7. [Notebooks](../notebooks/index.md)
8. [API Reference](../api/modules.md)
9. [Developer Docs](../developers/index.md)
10. [Development](../developers/index.md)
11. [Build Docs](build_docs.md)
12. [Docker](docker.md)
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
optimize
scoring
extensions
installation
build_docs
docker
changelog
```
