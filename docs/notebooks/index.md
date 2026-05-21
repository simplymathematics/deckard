# Notebook Index

Welcome! This page links to all executable notebooks in Deckard, organized as a progressive, end-to-end story for non-developers and researchers.

**How to use these notebooks:**

- Start with the core workflows (sklearn, pytorch) to learn the basics.
- Explore fairness, privacy, robustness, and visualization topics as needed.
- Each notebook is self-contained, but together they form a reproducible research pipeline.

Most notebooks follow this shape:

1. Build config objects for data/model/scoring.
1. Run training or evaluation for one trustworthiness question.
1. Inspect metrics and generated artifacts.
1. Adapt one configuration axis (model, attack, scorer, sampler) and rerun.

```{note}
For reproducibility in CI and local testing, notebooks are also wired into [DVC](https://dvc.org) stages. When validating notebook updates, prefer forced stage execution for changed notebooks to avoid stale cached outputs.
```

## Core Workflows

- [sklearn](sklearn) - End-to-end sklearn experiments.
- [pytorch](pytorch) - End-to-end PyTorch experiments.
- [hydra](hydra) - [Hydra](https://hydra.cc) config composition and overrides.
- [optuna](optuna) - [Optuna](https://optuna.org) hyperparameter optimization workflows.
- [artifacts](artifacts) - Artifact and output management.

## Fair Models and Group Scoring

- [fairlearn](fairlearn) - Fairness-aware data/model/score workflows.

## Security and Robustness

- Uses [Adversarial Robustness Toolbox (ART)](https://adversarial-robustness-toolbox.org/) backed attacks and defenses.
- [art_attacks](art_attacks) - Adversarial attack workflows.
- [art_defenses](art_defenses) - Defense pipeline workflows.
- {doc}`detector </notebooks/detector>` - Detector training and evaluation workflows.

## Privacy-aware models and metrics

- {doc}`anjana </notebooks/anjana>` - Anjana anonymization-aware workflows.

## Survival Analysis

- [lifelines](lifelines) - Survival analysis and time-to-event modeling.

## Visualization

- [seaborn](seaborn) - Seaborn plotting workflows.
- [yellowbrick](yellowbrick) - Yellowbrick model diagnostics.
- [deckard](deckard) - Deckard layer and script walkthroughs.

______________________________________________________________________

**Developer Docs:** For design docs and architectural standards, see [Developer Documentation](../developers/index.md).

## Suggested Reading Order

For general onboarding:

1. [sklearn](sklearn)
1. [pytorch](pytorch)
1. [hydra](hydra)
1. [optuna](optuna)

For fairness-first users:

1. [fairlearn](fairlearn)

For privacy-first users:

1. {doc}`anjana </notebooks/anjana>`

For command line usage:

1. [hydra](hydra)
1. [optuna](optuna)

For robustness-first users:

1. [art_attacks](art_attacks)
1. [art_defenses](art_defenses)
1. {doc}`detector </notebooks/detector>`

For reporting, explainability, and diagnostics:

1. [seaborn](seaborn)
1. [yellowbrick](yellowbrick)

## Execution Tips

- Execute cells in order unless the notebook explicitly marks independent
  sections.
- Re-run from top after dependency or environment changes.
- Keep generated outputs versioned only when they are intentional documentation
  artifacts.
- Use notebook logs and [DVC](https://dvc.org) stage output together when debugging failures.

```{toctree}
:maxdepth: 1
:hidden:

anjana
art_attacks
art_defenses
artifacts
deckard
detector
fairlearn
hydra
lifelines
optuna
pytorch
seaborn
sklearn
yellowbrick
```
