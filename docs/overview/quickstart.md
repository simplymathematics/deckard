# Quickstart

Use this page as the launch point for both first-time users and returning
contributors. It links to the highest-value documentation and explains when to
use each page.

## Start Here

If your goal is to run an experiment quickly:

1. Read [Installation](installation) to create a working environment.
1. Read [Summary](summary) for optimization-first architecture.
1. Read [Scoring](scoring) to understand objective metrics and runtime outputs.
1. Open the notebook guide in [Notebooks](../notebooks/index).

If your goal is multi-objective optimization:

1. Compose objective scorers in [Score API](../api/score).
1. Run optimization with [Hydra](https://hydra.cc) overrides and
   [Optuna](https://optuna.org) study storage.
1. Persist artifacts via [File API](../api/file).
1. Run post-hoc Pareto and plotting analysis via [Layers API](../api/layers).

If your goal is to extend deckard:

1. Read [Developer Docs](../developers/development).
1. Review [API](../api/modules).
1. Review extension APIs in [Extensions](extensions).

Core extension docs:

- [PyTorch](../api/pytorch)
- [Fairlearn](../api/fairlearn)
- [Anjana](../api/anjana)
- [Lifelines](../api/lifelines)
- [Seaborn](../api/seaborn)
- [Yellowbrick](../api/yellowbrick)

Base runtime config docs:

- [DataConfig](../api/data)
- [ModelConfig](../api/model)
- [AttackConfig](../api/attack)
- [DetectorConfig](../api/detector)
- [ExperimentConfig](../api/experiment)
- [FileConfig](../api/file)

## Documentation Map

## [Summary](summary)

A conceptual summary of the package.

## [Installation](installation)

Installation instructions for users.

## [API](../api/modules)

Core package documentation

## [Developer Docs](../developers/development)

Documentation for testing and extending this package.

## [Docs Docs](build_docs)

Documentation about how to build this documentation.

## [Extensions](extensions)

Extension points and optional subsystems for additional workflows.

## [Changelog](changelog)

A history of changes.

## Recommended Learning Paths

### Path A: Experiment Users

- Install dependencies (core plus optional stacks as needed).
- Run one notebook workflow ([sklearn](../notebooks/sklearn) or [pytorch](../notebooks/pytorch)).
- Inspect scoring outputs and persisted artifacts.
- Promote the most relevant metrics to multi-objective optimization targets.
- Use [Layers](../api/layers) for post-hoc evaluations (Pareto filtering,
  plotting, survival workflows).
- Adapt one config for a new dataset, model, or metric.

### Path B: Framework Contributors

- Read API module docs for the area you are extending.
- Follow development/testing conventions.
- Add or update docs and notebook examples for new behavior.
- Validate with local workflows and documentation builds.

## Notes On Scope

The pages linked here focus on documentation and architecture orientation.
Executable examples and reproducible runs are covered in notebooks and examples
directories, while module-level behavior is captured in API references.
