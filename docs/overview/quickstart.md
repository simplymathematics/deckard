# Quickstart

Use this page as the launch point for both first-time users and returning
contributors. It follows the same overview flow as the rest of this section:
understand the project, inspect the core runtime surfaces, then move into
experiment orchestration and scoring.

## Overview Flow

The fastest path through the overview section is:

1. {doc}`summary`
2. {doc}`core`
3. {doc}`experiment`
4. {doc}`scoring`
5. {doc}`hydra`
6. {doc}`optimize`
7. {doc}`dvc`
8. {doc}`extensions/index`
9. {doc}`installation`
10. {doc}`docker`
11. {doc}`../developers/index`
12. {doc}`changelog`

## Start Here

If your goal is to run an experiment quickly:

1. Read {doc}`installation` to create a working environment.
2. Read {doc}`summary` for the package model.
3. Read {doc}`core` to identify the API surfaces you will configure.
4. Read {doc}`experiment` for the end-to-end runtime workflow.
5. Read {doc}`scoring` to understand objective metrics and runtime outputs.
6. Open the notebook guide in {doc}`../notebooks/index`.

If your goal is multi-objective optimization:

1. Compose objective scorers in {doc}`../api/score`.
2. Read {doc}`hydra` for run and multirun orchestration.
3. Read {doc}`optimize` for Optuna-backed optimization behavior.
4. Read {doc}`dvc` for pipeline and artifact generation.
5. Persist artifacts via {doc}`../api/file`.
6. Run post-hoc Pareto and plotting analysis via {doc}`../api/layers`.

If your goal is to extend deckard:

1. Read {doc}`../developers/index`.
2. Review {doc}`../api/modules`.
3. Review extension APIs in {doc}`extensions/index`.

## Core Extensions

- [PyTorch](../api/pytorch)
- [Fairlearn](../api/fairlearn)
- [Anjana](../api/anjana)
- [Lifelines](../api/lifelines)
- [Seaborn](../api/seaborn)
- [Yellowbrick](../api/yellowbrick)

Base runtime config docs:

- {class}`deckard.data.DataConfig`
- {class}`deckard.model.ModelConfig`
- {class}`deckard.attack.AttackConfig`
- {class}`deckard.detector.DetectorConfig`
- {class}`deckard.experiment.ExperimentConfig`
- {class}`deckard.file.FileConfig`

## Documentation Map

- {doc}`summary`: high-level architecture and purpose.
- {doc}`core`: compact map of the core runtime and API surfaces.
- {doc}`experiment`: end-to-end orchestration flow through {class}`deckard.experiment.ExperimentConfig`.
- {doc}`scoring`: score outputs, objectives, and persisted runtime metrics.
- {doc}`hydra`: run and multirun config composition.
- {doc}`optimize`: optimization lifecycle and Optuna integration.
- {doc}`dvc`: DVC pipeline autogeneration and output conventions.
- {doc}`extensions/index`: optional framework and plugin families.
- {doc}`installation`: environment setup.
- {doc}`docker`: containerized workflows.
- {doc}`../developers/index`: contributor-facing design docs.
- {doc}`changelog`: project history.

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
