# Notebook Index

Welcome!
This page links to all executable notebooks in Deckard, organized as a
progressive, end-to-end story for non-developers and researchers.

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
  - {doc}`seaborn </notebooks/seaborn>`allows users to configure post-hoc plotting for {class}`~deckard.data.DataConfig` objects including pandas-compatible data sources and [optuna](https://optuna.org) databases.

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
<!-- - {doc}`dvc </notebooks/dvclive>` - DVCLive runtime logging, monitoring, and hook-scoped score emission. -->
<!-- - {doc}`optimize </notebooks/optimize>` - demonstrates how to use. -->
<!-- - {doc}`dvc </notebooks/dvclive>` - DVCLive runtime logging, monitoring, and hook-scoped score emission. -->
<!-- - {doc}`deckard </notebooks/deckard>` - [optimize] runtime demonstrations for run and multirun execution. -->

```{toctree}
:maxdepth: 1
:caption: CLI and Optimization
:hidden:
hydra
optuna
dvc
```


## Per-Notebook Run Expectations

Use these expectations when running notebooks locally, in CI, or through the
DVC-backed docs build.

| Notebook | Main purpose | Expected persisted outputs |
| --- | --- | --- |
| {doc}`sklearn </notebooks/sklearn>` | Canonical tabular runtime flow | score artifacts, experiment outputs, and split-aware scoring examples |
| {doc}`pytorch </notebooks/pytorch>` | Torch-native training and checkpoint flow | checkpoint files, score artifacts, and model-state examples |
| {doc}`huggingface </notebooks/huggingface>` | Transformer-native text pipeline and attack flow | transformer checkpoints, attack score summaries, and Hugging Face dataset-driven outputs |
| {doc}`hydra </notebooks/hydra>` | Compose-first config and override behavior | single-run params or score artifacts and resolved override examples |
| {doc}`dvc </notebooks/dvc>` | DVC stage planning and contract decomposition | canonical stage mappings, stage plan summaries, and contract-oriented inspection outputs |
| {doc}`scoring </notebooks/scoring>` | ScoreDict lifecycle and persistence | human-readable score payloads and flat or dotlist projections |
| {doc}`artifacts </notebooks/artifacts>` | Artifact hydration and pretrained reload paths | cached model, attack, and score artifacts for sklearn and torch paths |
| {doc}`art_attacks </notebooks/art_attacks>` | Attack-family execution and timing outputs | attack artifacts, score tables, and attack-family timing summaries |
| {doc}`art_defenses </notebooks/art_defenses>` | Defense execution and evaluation | defended artifact outputs and comparison metrics |
| {doc}`detector </notebooks/detector>` | Detector fit or detect orchestration | detector outputs, filtered artifacts, and detector score payloads |
| {doc}`fairlearn </notebooks/fairlearn>` | Fairness-aware data/model/score flow | group metric score artifacts and fairness-specific runtime outputs |
| {doc}`anjana </notebooks/anjana>` | Privacy-aware preprocessing and privacy scoring | anonymized artifacts, defended outputs, and privacy score files |
| {doc}`lifelines </notebooks/lifelines>` | Survival-analysis runtime flow | survival metrics, tables, and backend-specific persisted outputs |
| {doc}`seaborn </notebooks/seaborn>` | Results-table driven plotting | figure outputs and plot-spec-backed artifacts |
| {doc}`yellowbrick </notebooks/yellowbrick>` | Experiment-backed diagnostics | diagnostic figures and model-analysis outputs |


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
- Use `dvc repro --force notebook_<name>` ([dvc docs](https://dvc.org)) to delete cached artifacts and force a new new.
