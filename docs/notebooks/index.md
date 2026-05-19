
# Notebook Index

Welcome! This page links to all executable notebooks in Deckard, organized as a progressive, end-to-end story for non-developers and researchers.

**How to use these notebooks:**
- Start with the core workflows (sklearn, pytorch) to learn the basics.
- Explore fairness, privacy, robustness, and visualization topics as needed.
- Each notebook is self-contained, but together they form a reproducible research pipeline.

Most notebooks follow this shape:
1. Build config objects for data/model/scoring.
2. Run training or evaluation for one trustworthiness question.
3. Inspect metrics and generated artifacts.
4. Adapt one configuration axis (model, attack, scorer, sampler) and rerun.

```{note}
For reproducibility in CI and local testing, notebooks are also wired into DVC stages. When validating notebook updates, prefer forced stage execution for changed notebooks to avoid stale cached outputs.
```


## Core Workflows
- [sklearn](sklearn) - End-to-end sklearn experiments.
- [pytorch](pytorch) - End-to-end PyTorch experiments.
- [hydra](hydra) - Hydra config composition and overrides.
- [optuna](optuna) - Optuna hyperparameter optimization workflows.
- [artifacts](artifacts) - Artifact and output management.

## Fair Models and Group Scoring
- [fairlearn](fairlearn) - Fairness-aware data/model/score workflows.

## Security and Robustness
- [art_attacks](art_attacks) - Adversarial attack workflows.
- [art_defenses](art_defenses) - Defense pipeline workflows.
- {doc}`detector <detector>` - Detector training and evaluation workflows.

## Privacy-aware models and metrics
- {doc}`anjana <anjana>` - Anjana anonymization-aware workflows.

## Survival Analysis
- [lifelines](lifelines) - Survival analysis and time-to-event modeling.

## Visualization
- [seaborn](seaborn) - Seaborn plotting workflows.
- [yellowbrick](yellowbrick) - Yellowbrick model diagnostics.
- [deckard](deckard) - Deckard layer and script walkthroughs.

---

**Developer Docs:** For design docs and architectural standards, see [Developer Documentation](../developers/index.md).


## Suggested Reading Order

For general onboarding:

1. [sklearn](sklearn)
2. [pytorch](pytorch)
3. [fairlearn](fairlearn)
4. {doc}`anjana <anjana>`

For robustness-first users:
1. [art_attacks](art_attacks)
2. [art_defenses](art_defenses)
3. {doc}`detector <detector>`

For reporting, explainability, and diagnostics:
1. [seaborn](seaborn)
2. [yellowbrick](yellowbrick)

## Execution Tips

- Execute cells in order unless the notebook explicitly marks independent
	sections.
- Re-run from top after dependency or environment changes.
- Keep generated outputs versioned only when they are intentional documentation
	artifacts.
- Use notebook logs and DVC stage output together when debugging failures.

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
