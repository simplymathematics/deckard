# Quickstart

Use this page as the launch point for both first-time users and returning
contributors. It follows the same overview flow as the rest of this section:
understand the project, inspect the core runtime surfaces, then move into
experiment orchestration and scoring.

Use this page for execution-oriented onboarding. For architecture context and
capability mapping, use {doc}`index`.

## Overview Flow

The fastest path through the overview section is:

1. {doc}`index`
2. {doc}`core`
3. {doc}`experiment`
4. {doc}`scoring`
5. {doc}`hydra`
6. {doc}`optimize`
7. {doc}`dvc`
8. {doc}`extensions/index`
9. {doc}`installation`


## Start Here

If your goal is to run an experiment quickly:

1. Read {doc}`installation` to create a working environment.
2. Read {doc}`index` for the package model and capability map.
3. Read {doc}`core` to identify the API surfaces you will configure.
4. Read {doc}`experiment` for the end-to-end runtime workflow.
5. Read {doc}`scoring` to understand objective metrics and runtime outputs.
6. Open the notebook guide in {doc}`../notebooks/index`.

If your goal is multi-objective optimization:

1. Compose objective scorers in {doc}`/api/score/index`.
2. Read {doc}`hydra` for run and multirun orchestration.
3. Read {doc}`optimize` for Optuna-backed optimization behavior.
4. Read {doc}`dvc` for pipeline and artifact generation.
5. Persist artifacts via {doc}`/api/file/index`.
6. Run post-hoc Pareto and plotting analysis via {doc}`/api/layers/index`.

If your goal is to extend deckard:

1. Read {doc}`../developers/index`.
2. Review {doc}`../api/modules`.
3. Review extension APIs in {doc}`extensions/index`.

## Core Extensions

- [PyTorch](/api/pytorch/index)
- [Fairlearn](/api/plugins/fairlearn)
- [Anjana](/api/plugins/anjana)
- [Lifelines](/api/plugins/lifelines)
- [Seaborn](/api/plugins/seaborn)
- [Yellowbrick](/api/plugins/yellowbrick)

## Programmatic Example

```python
from deckard import DataConfig, ModelConfig, AttackConfig, ExperimentConfig

data = DataConfig(name="adult")
model = ModelConfig(name="sklearn.linear_model.LogisticRegression")
attack = AttackConfig()
experiment = ExperimentConfig(data=data, model=model, attack=attack)

scores = experiment()
print(scores)
```

## Command-Line Orientation

Use the package through module entrypoints or the top-level CLI router:

```bash
python -m deckard --help
python -m deckard optimize --help
python -m deckard plot --help
```

Base runtime config docs:

- {class}`deckard.data.DataConfig`
- {class}`deckard.model.ModelConfig`
- {class}`deckard.attack.AttackConfig`
- {class}`deckard.detector.DetectorConfig`
- {class}`deckard.experiment.ExperimentConfig`
- {class}`deckard.file.FileConfig`

## Documentation Map

- {doc}`index`: high-level architecture and capability map.
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

## Experiment Management Snapshot

Typical run composition includes:

1. dataset loading and sampling
2. model training/evaluation
3. optional defense application
4. optional attack and detector execution
5. scoring and artifact persistence

This stage model keeps large parameter sweeps auditable and comparable.

## Optimization-First Workflow

1. Define objective scorers in {doc}`/api/score/index`.
2. Compose experiment config with [Hydra](https://hydra.cc) groups.
3. Run single or multi-objective optimization through [Optuna](https://optuna.org).
4. Persist predictions, scores, and metadata through {doc}`/api/file/index`.
5. Run post-hoc analysis via {doc}`/api/layers/index`.

## Recommended Learning Paths

### Path A: Experiment Users

- Install dependencies (core plus optional stacks as needed).
- Run one notebook workflow ([sklearn](../notebooks/sklearn) or [pytorch](../notebooks/pytorch)).
- Inspect scoring outputs and persisted artifacts.
- Promote the most relevant metrics to multi-objective optimization targets.
- Use [Layers](/api/layers/index) for post-hoc evaluations (Pareto filtering,
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
