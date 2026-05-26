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

```{note}
For reproducibility in CI and local testing, notebooks are also wired into
[DVC](https://dvc.org) stages.
When validating notebook updates, prefer forced stage execution for changed
notebooks to avoid stale cached outputs.
```

## Frameworks and Plugins

Deckard notebooks also serve as the fastest way to understand the project's
extension surfaces.

Framework support:

- {doc}`sklearn </notebooks/sklearn>` demonstrates the default tabular pipeline stack, including
  data transforms, model execution, and score composition.
- {doc}`pytorch </notebooks/pytorch>` demonstrates torch-native training and evaluation flows,
  including dataset loaders, tensor models, and attack-aware scoring.

Plugin functionality:

- {doc}`fairlearn </notebooks/fairlearn>` shows fairness-aware data, model, and group-metric
  workflows.
- {doc}`anjana </notebooks/anjana>` shows anonymization-aware preprocessing and
  privacy-oriented evaluation.
- {doc}`lifelines </notebooks/lifelines>` shows survival-analysis and time-to-event workflows.
- {doc}`seaborn </notebooks/seaborn>` and {doc}`yellowbrick </notebooks/yellowbrick>` show reporting,
  diagnostics, and visualization integrations.
- {doc}`art_attacks </notebooks/art_attacks>`, {doc}`art_defenses </notebooks/art_defenses>`, and
  {doc}`detector </notebooks/detector>` show robustness, defense, and detector
  integrations built around attack and monitoring workflows.

## Core Workflows

- {doc}`sklearn </notebooks/sklearn>` - End-to-end sklearn experiments.
- {doc}`pytorch </notebooks/pytorch>` - End-to-end PyTorch experiments.
- {doc}`hydra </notebooks/hydra>` - [Hydra](https://hydra.cc) config composition and overrides.
<!-- - {doc}`optimize </notebooks/optimize>` - optimize runtime demonstrations for run and multirun execution. -->
- {doc}`dvc </notebooks/dvc>` - DVC pipeline autogeneration.
- {doc}`optuna </notebooks/optuna>` - [Optuna](https://optuna.org) hyperparameter optimization workflows.
- {doc}`artifacts </notebooks/artifacts>` - Artifact and output management.
- {doc}`scoring </notebooks/scoring>` - ScoreDict contract, score lifecycle, and persistence views.

## Contract Coverage Map

The notebook suite is organized so the runtime contract is demonstrated in a
small number of canonical places rather than repeated everywhere.

| Contract question | Primary notebooks | What to look for |
| --- | --- | --- |
| Files-only persistence aliases | {doc}`sklearn </notebooks/sklearn>`, {doc}`hydra </notebooks/hydra>`, {doc}`dvc </notebooks/dvc>` | `files={...}` payloads, `+files.params_file=...`, `+files.score_file=...`, and persisted run artifacts |
| Canonical timing keys plus extensibility | {doc}`scoring </notebooks/scoring>`, {doc}`sklearn </notebooks/sklearn>`, {doc}`art_attacks </notebooks/art_attacks>`, {doc}`fairlearn </notebooks/fairlearn>` | flat timing keys such as `prediction_time` and richer nested timing payloads for plugin or attack-specific execution |
| Stage and mode normalization plus hook ordering | {doc}`scoring </notebooks/scoring>`, {doc}`hydra </notebooks/hydra>`, {doc}`anjana </notebooks/anjana>`, {doc}`fairlearn </notebooks/fairlearn>` | explicit `score_mode`, stage-aware hooks, and plugin hook merge behavior |
| Cache-key determinism and selective invalidation | {doc}`dvc </notebooks/dvc>`, {doc}`artifacts </notebooks/artifacts>` | repeated run or multirun templates, cache reuse expectations, and resolved artifact identity |
| Human-readable YAML and JSON params or score artifacts | {doc}`scoring </notebooks/scoring>`, {doc}`hydra </notebooks/hydra>`, {doc}`dvc </notebooks/dvc>`, {doc}`artifacts </notebooks/artifacts>` | `scores.json`, `params.yaml`, DVC-generated manifests, and round-trip artifact inspection |

<!-- optimize notebook references are temporarily disabled while notebook is under construction. -->

## Per-Notebook Run Expectations

Use these expectations when running notebooks locally, in CI, or through the
DVC-backed docs build.

| Notebook | Main purpose | Expected persisted outputs |
| --- | --- | --- |
| {doc}`sklearn </notebooks/sklearn>` | Canonical tabular runtime flow | score artifacts, experiment outputs, and split-aware scoring examples |
| {doc}`pytorch </notebooks/pytorch>` | Torch-native training and checkpoint flow | checkpoint files, score artifacts, and model-state examples |
| {doc}`hydra </notebooks/hydra>` | Compose-first config and override behavior | single-run params or score artifacts and resolved override examples |
<!-- | {doc}`optimize </notebooks/optimize>` | Run versus multirun optimization flow | `params.yaml`, `scores.json`, Optuna storage examples, and cache reuse templates | -->
| {doc}`dvc </notebooks/dvc>` | DVC stage and report generation | `dvc.yaml`, `params.yaml`, `scores.json`, and Vega-Lite-oriented output wiring |

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
| {doc}`deckard </notebooks/deckard>` | Layer and script walkthroughs | CLI-adjacent params or score examples and generated helper artifacts |

## Fair Models and Group Scoring

Fairness functionality is provided through the Fairlearn plugin layer.

- {doc}`fairlearn </notebooks/fairlearn>` - Fairness-aware data/model/score workflows.

## Security and Robustness

- Uses [Adversarial Robustness Toolbox
  (ART)](https://adversarial-robustness-toolbox.org/) backed attacks and
  defenses.
- {doc}`art_attacks </notebooks/art_attacks>` - Adversarial attack workflows.
- {doc}`art_defenses </notebooks/art_defenses>` - Defense pipeline workflows.
- {doc}`detector </notebooks/detector>` - Detector training and evaluation workflows.

## Privacy-aware models and metrics

Privacy-aware functionality is provided through the Anjana plugin layer.

- {doc}`anjana </notebooks/anjana>` - Anjana anonymization-aware workflows.

## Survival Analysis

Survival-analysis functionality is provided through the Lifelines plugin layer.

- {doc}`lifelines </notebooks/lifelines>` - Survival analysis and time-to-event modeling.

## Visualization

Visualization functionality is provided through the Seaborn and Yellowbrick
plugin layers.

- {doc}`seaborn </notebooks/seaborn>` - Seaborn plotting workflows.
- {doc}`yellowbrick </notebooks/yellowbrick>` - Yellowbrick model diagnostics.
- {doc}`deckard </notebooks/deckard>` - Deckard layer and script walkthroughs.

______________________________________________________________________

**Developer Docs:** For design docs and architectural standards, see {doc}`../developers/index`.

## Suggested Reading Order

For general onboarding:

1. {doc}`sklearn </notebooks/sklearn>`
1. {doc}`pytorch </notebooks/pytorch>`
1. {doc}`hydra </notebooks/hydra>`
<!-- 1. {doc}`optimize </notebooks/optimize>` -->
1. {doc}`optuna </notebooks/optuna>`

For fairness-first users:

1. {doc}`fairlearn </notebooks/fairlearn>`

For privacy-first users:

1. {doc}`anjana </notebooks/anjana>`

For survival models:

1. {doc}`lifelines </notebooks/lifelines>`

For command line usage:

1. {doc}`hydra </notebooks/hydra>`
2. {doc}`optuna </notebooks/optuna>`

For robustness-first users:

1. {doc}`art_attacks </notebooks/art_attacks>`
2. {doc}`art_defenses </notebooks/art_defenses>`
3. {doc}`detector </notebooks/detector>`

For reporting, explainability, and diagnostics:

1. {doc}`seaborn </notebooks/seaborn>`
2. {doc}`yellowbrick </notebooks/yellowbrick>`

## Execution Tips

- Execute cells in order unless the notebook explicitly marks independent
  sections.
- Re-run from top after dependency or environment changes.
- Keep generated outputs versioned only when they are intentional documentation
  artifacts.
- Use notebook logs and [DVC](https://dvc.org) stage output together when
  debugging failures.

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
<!-- optimize -->
dvc
lifelines
optuna
pytorch
seaborn
sklearn
yellowbrick
scoring
```
