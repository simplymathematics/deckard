# Notebook Index

Welcome!
This page links to the published notebooks and companion guides in deckard,
organized as a progressive, end-to-end story for users and researchers.

**How to use these notebooks:**

- Start with the core workflows (sklearn, pytorch) to learn the basics.
- Explore fairness, privacy, robustness, and visualization topics as needed.
- Each notebook is self-contained, but together they form a reproducible
  research pipeline.

Most notebooks follow this shape:

1. Build config objects for data/model/scoring.
1. Run training or evaluation for one trustworthiness question.
1. Inspect metrics and generated artifacts.
1. Adapt one configuration axis (model, attack, scorer, sampler) and rerun.


## Supported Frameworks

Deckard notebooks also serve as the fastest way to understand the project's
extension surfaces.
```{toctree}
:maxdepth: 1
:caption: Frameworks
:hidden:
sklearn
pytorch
huggingface
```


- {doc}`sklearn </notebooks/sklearn>` demonstrates the default tabular pipeline stack, including
  data transforms, model execution, and score composition.
- {doc}`pytorch </notebooks/pytorch>` demonstrates torch-native training and evaluation flows,
  including dataset loaders, tensor models, and attack-aware scoring.
- {doc}`huggingface </notebooks/huggingface>` demonstrates Hugging Face dataset/model workflows
  integrated with Deckard attack and scoring orchestration.



## Security and Robustness
Uses [Adversarial Robustness Toolbox(ART)](https://adversarial-robustness-toolbox.org/) backed attacks and
  defenses.
- {doc}`art_attacks </notebooks/art_attacks>` - Adversarial attack workflows.
- {doc}`art_defenses </notebooks/art_defenses>` - Defense pipeline workflows.
- {doc}`detector </notebooks/detector>` - Detector training and evaluation workflows.
```{toctree}
:maxdepth: 1
:caption: Attacks and Defenses
:hidden:
art_attacks
art_defenses
detector
```

## Privacy-aware models and metrics

Privacy-aware functionality is provided through the [anjana](https://anjana.readthedocs.io/en/latest/) plugin layer.

- {doc}`anjana </notebooks/anjana>` -  anjana anonymization-aware workflows.

```{toctree}
:maxdepth: 1
:caption: Privacy and Anonymization
:hidden:
anjana
```

## Fair Models and Group Scoring

Fairness functionality is provided through the Fairlearn plugin layer.

- {doc}`fairlearn </notebooks/fairlearn>` - Fairness-aware data/model/score workflows.
```{toctree}
:maxdepth: 1
:hidden:
:caption: Fairness and Group Scoring
fairlearn
```
## Survival Analysis

[Survival-analysis](https://lifelines.readthedocs.io/en/latest/Survival%20Analysis%20intro.html) functionality is provided through the [lifelines](https://lifelines.readthedocs.io/en/latest/index.html) plugin layer.

- {doc}`lifelines </notebooks/lifelines>` - Survival analysis and time-to-event modeling.

```{toctree}
:caption: Survival Analysis
:maxdepth: 1
:hidden:
lifelines
```

## Visualization

- {doc}`yellowbrick </notebooks/yellowbrick>` show reporting,
  diagnostics, and visualization integrations on deckard {class}`~deckard.experiment.ExperimentConfig` objects.
- {doc}`seaborn </notebooks/seaborn>` allows users to configure post-hoc plotting for {class}`~deckard.data.DataConfig` objects including pandas-compatible data sources and [optuna](https://optuna.org) databases.

```{toctree}
:maxdepth: 1
:caption: Visualization
:hidden:
seaborn
yellowbrick
```

## Persistence
- {doc}`artifacts </notebooks/artifacts>` - Artifact and output management.
- {doc}`scoring </notebooks/scoring>` - ScoreDict contract, score lifecycle, and persistence views.

```{toctree}
:maxdepth: 1
:caption: Persistence
:hidden:
scoring
artifacts
```



## CLI and Optimization
- {doc}`hydra </notebooks/hydra>` - [Hydra](https://hydra.cc) config composition and overrides.
- {doc}`dvc </notebooks/dvc>` - DVC stage planning, canonical stage contracts, and stage decomposition.
- {doc}`optuna </notebooks/optuna>` - [Optuna](https://optuna.org) hyperparameter optimization workflows.
- {doc}`optimize </notebooks/optimize>` - examples/sklearn walkthrough of single-run and multirun optimization, including {class}`~deckard.layers.optimize.OptimizerConfig` and {class}`~deckard.layers.optimize.DefaultOptimizerCallback`.
<!-- - {doc}`deckard </notebooks/deckard>` - narrative CLI tour of the public `deckard layers` commands in the examples/sklearn context. -->

```{toctree}
:maxdepth: 1
:caption: CLI and Optimization
:hidden:
hydra
optuna
dvc

optimize
```




```{note}
For reproducibility in CI and local testing, notebooks are also wired into
[DVC](https://dvc.org) stages.
When validating notebook updates, prefer forced stage execution for changed
notebooks to avoid stale cached outputs.
```

## Execution Tips

- Execute cells in order unless the notebook explicitly marks independent
  sections.
- Re-run from top after dependency or environment changes.
- Keep generated outputs versioned only when they are intentional documentation
  artifacts.
- Use `dvc repro --force notebook_<name>` ([dvc docs](https://dvc.org)) to delete cached artifacts and force a new build.
