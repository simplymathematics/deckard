# Core Modules

This page gives a simple guide to the main deckard runtime objects. Each
section keeps the view high-level and shows the broad path through the object,
including any optional branches.

The full API reference lives in {doc}`../api/modules`, which contains the
module-level docs for each object.

## Data API

The data object starts the run. It loads the data, samples it, and can apply a
pipeline transform before scoring or passing on to the next step.

This overview chart keeps the data path simple and shows the optional transform
step.

```{include} flowcharts.md
:start-after: <!-- core-data-overview-start -->
:end-before: <!-- core-data-overview-end -->
```

Scoping detail:

```{include} flowcharts.md
:start-after: <!-- core-data-scope-start -->
:end-before: <!-- core-data-scope-end -->
```

- [Data](/api/data/index): dataset loading and runtime coordination.
- [Sample](/api/data/sample): sampling helpers.
- [Pipeline](/api/data/pipeline): preprocessing transforms.

## Model API

The model object learns from the data, can train or load a model, and may add
an optional defense before prediction. The final step is to persist outputs.

This overview chart shows the simple model path and where defense fits.

```{include} flowcharts.md
:start-after: <!-- core-model-overview-start -->
:end-before: <!-- core-model-overview-end -->
```

Scoping detail for trainer choices:

```{include} flowcharts.md
:start-after: <!-- core-model-trainer-scope-start -->
:end-before: <!-- core-model-trainer-scope-end -->
```

Scoping detail for defense subtypes:

```{include} flowcharts.md
:start-after: <!-- core-defense-subtypes-start -->
:end-before: <!-- core-defense-subtypes-end -->
```

- [Model](/api/model/index): model setup and runtime behavior.
- [Training](/api/model/train): trainer helpers.
- [Defense](/api/model/defend): defense behavior.

## Attack API

The attack object checks how the model behaves when inputs are changed on
purpose. The main families are evasion, poisoning, inference, and extraction,
and they all end in score output.

This overview chart shows those attack families without going into stage-level
runtime detail.

```{include} flowcharts.md
:start-after: <!-- core-attack-overview-start -->
:end-before: <!-- core-attack-overview-end -->
```

Scoping detail for attack subtypes:

```{include} flowcharts.md
:start-after: <!-- core-attack-family-start -->
:end-before: <!-- core-attack-family-end -->
```

- [Attack](/api/attack/index): attack execution and scoring.

## Detector API

Detectors either train a detector model or filter attack outputs. The most
important split is train versus filter.

```{include} flowcharts.md
:start-after: <!-- core-detector-overview-start -->
:end-before: <!-- core-detector-overview-end -->
```

Scoping detail for detector train and filter modes:

```{include} flowcharts.md
:start-after: <!-- core-detector-mode-start -->
:end-before: <!-- core-detector-mode-end -->
```

- [Detector](/api/detector/index): detector training and filter-mode behavior.

## Score API

Scoring combines outputs from data, model, attack, and optional group scorers
into one score payload.

```{include} flowcharts.md
:start-after: <!-- core-score-overview-start -->
:end-before: <!-- core-score-overview-end -->
```

Scoping detail for data, model, attack, and group scorers:

```{include} flowcharts.md
:start-after: <!-- core-score-composition-start -->
:end-before: <!-- core-score-composition-end -->
```

- [Score](/api/score/index): scorer setup and score payload composition.
- [Fairlearn](/api/plugins/fairlearn): group-aware scorers.

## Experiment API

The experiment object ties data, model, attack, detector, and scoring together
into one run.

Experiment pages show the orchestration layer that connects the smaller pieces.
This is where the runtime order becomes clear: prepare data, run the model,
apply attacks or detectors if needed, and then score the result.

```{include} flowcharts.md
:start-after: <!-- core-experiment-overview-start -->
:end-before: <!-- core-experiment-overview-end -->
```

- [Experiment](/api/experiment/index): end-to-end orchestration runtime.
- [Score](/api/score/index): scorer setup and metric composition.
- [Plot](/api/plot/index): plotting configuration for run outputs.


## Persistence API

The persistence layer saves the run so it can be checked, compared, and
reused later.

Persistence includes file helpers, stored artifacts, and utility code that keep
outputs organized. These pieces make it possible to reopen a run later and
compare it with other experiments.

```{include} flowcharts.md
:start-after: <!-- core-persistence-overview-start -->
:end-before: <!-- core-persistence-overview-end -->
```

- [File](/api/file/index): file paths, persistence helpers, and saved outputs.
- [Artifacts](/api/artifacts/index): artifact handling and stored run data.
- [Utils](/api/utils/index): shared helpers for file and runtime support.
