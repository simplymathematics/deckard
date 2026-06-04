# Default Workflow Flowchart

This page shows the default Deckard workflow as one high-level orchestration
diagram.

Shared semantic color system:

- Data -> green
- Model -> blue
- Attack -> red
- Detector -> purple
- Files -> light gray
- Scores -> orange
- Experiment -> dark gray
- Plots -> dark gray

## Default Workflow

Canonical orchestration entrypoint: {meth}`deckard.experiment.ExperimentConfig.run`.

```mermaid
flowchart TD
  A[ExperimentConfig.run]:::experiment
  B[Data]:::data
  C[Model]:::model
  D[Attack]:::attack
  E[Detect]:::detector
  F[Score stage: train / test / val / all]:::scores
  G[Persist stage: outputs, files, artifacts]:::files
  H[Visualize and review results]:::experiment

  A --> B --> C --> D --> E --> F --> G --> H

```

## Reading Guide

- The default path is `load -> sample -> train -> defense -> attack -> detector -> score -> persist`.
- Attack sits after the model has produced baseline outputs so you can compare
  clean and attacked results.
- Detector and scoring come after the attack path, and persistence is followed
  by visualization and review.

## Core API Flowcharts

The flowcharts below are the canonical source for the core object diagrams used
in [Core Modules](core).

### Data API Overview

<!-- core-data-overview-start -->
```mermaid
flowchart LR
  A[load]:::data -->  B[pre-sample pipeline]:::data --> C[sample]:::data
  C --> D[pipeline]:::data
  D --> E[score]:::scores
```
<!-- core-data-overview-end -->

### Model API Overview

<!-- core-model-overview-start -->
```mermaid
flowchart LR
  A[initialize model]:::model --> B[train or load]:::model
  B --> C[defend]:::model
  C --> D[predict]:::model
  D --> E[persist]:::files
```
<!-- core-model-overview-end -->

### Model Trainer Scope

<!-- core-model-trainer-scope-start -->
```mermaid
flowchart LR
  A[model initialization]:::model --> B[default training]:::model
  A --> C[pretrained]:::model
  A --> D[partial_fit]:::model
  A --> E[partial_fit_pruning]:::model
  A --> F[pruning]:::model
  B --> G[prediction]:::model
  C --> G[prediction]:::model
  D --> G[prediction]:::model
  E --> G[prediction]:::model
  F --> G[prediction]:::model
  G --> H[scoring]:::model
  H --> I[persistence]:::model
  
```
<!-- core-model-trainer-scope-end -->

### Defense Subtypes

<!-- core-defense-subtypes-start -->
```mermaid
flowchart TD
  A[model initialization]:::model --> B[anjana]:::model
  B --> C[fairlearn.reductions]:::model
  C --> D[other defenses]:::model
  D --> E[ART apply_fit]:::model
  E --> F[fairlearn.adversarial]:::model
  F --> G[train]:::model
  G --> H[fairlearn.postprocessing]:::model
  G --> I[ART apply_predict]:::model
  H --> J[predict]:::model
  C --> G
  D --> G
  E --> G
  F --> G
```
<!-- core-defense-subtypes-end -->

### Attack API Overview

<!-- core-attack-overview-start -->
```mermaid
flowchart LR
  A[load]:::data --> B[sample]:::data
  C --> D[model initialization]:::model
  D --> E[train]:::model
  E --> F[predict]:::model
  F --> G[score]:::scores
  EVA[evade]:::attack --> G
  POI[poison]:::attack --> F
  B --> INF[infer]:::attack
  E --> EXT[extract]:::attack
```
<!-- core-attack-overview-end -->

### Detector API Overview

<!-- core-detector-overview-start -->
```mermaid
flowchart LR
  A[DataConfig]:::data --> B[ModelConfig]:::model -> C[AttackConfig]:::attack
  D[train]:::detector --> E[filter poisoning]:::detector
  D --> F[filter evasion]:::detector
  E --> B --> G[score]:::scores
  F --> C --> G
```
<!-- core-detector-overview-end -->

### Score API Overview

<!-- core-score-overview-start -->
```mermaid
flowchart TD
  A[data scorers]:::scores --> D[score_dict]:::scores
  B[model scorers]:::scores --> D
  C[attack scorers]:::scores --> D
  D --> F[score_file]:::files
```
<!-- core-score-overview-end -->

### Experiment API Overview

<!-- core-experiment-overview-start -->
```mermaid
flowchart TD
  A[load]:::data --> B[sample]:::data --> B2[pipeline]:::data
  B2 --> C[apply_fit defense]:::model
  C --> D[train]:::model
  D --> E[apply_predict defense]:::model
  E --> F[predict]:::model
  F --> G[attack]:::attack
  G --> H[detector]:::detector
  H --> I[score]:::scores
  I --> J[persist]:::files
  J --> K[review / post-hoc analysis]:::experiment
```
<!-- core-experiment-overview-end -->

### Persistence API Overview

<!-- core-persistence-overview-start -->
```mermaid
flowchart LR
  A[outputs]:::files --> B[file name resolution]:::files
  B --> C[artifacts]:::files
```
<!-- core-persistence-overview-end -->

## Plugin Flowcharts

### ANJANA Execution Flows
<!-- anjana-data-overview-start -->
```mermaid
flowchart LR
  A[load]:::data -->  B[anonymization]:::data --> C[sample]:::data
  C --> D[pipeline]:::data
  D --> E[score]:::scores
```
<!-- anjana-data-overview-end -->
### Fairlearn Execution Flows
<!-- fairlearn-model-overview-start -->
```mermaid
flowchart LR
  A[defenses]:::model --> B[anjana]:::model
  A --> C[fairlearn.reductions]:::model
  A --> D[fairlearn.adversarial]:::model
  A --> E[fairlearn.postprocessing]:::model
  B --> G[apply defense]:::model
  C --> H[fit]:::model
  D --> H
  E --> I[predict]:::model
  G --> H --> I
```
<!-- fairlearn-model-overview-end -->

<!-- fairlearn-score-overview-start -->
```mermaid
flowchart TD
  A[data scorers]:::scores --> E
  B[model scorers]:::scores --> E
  C[attack scorers]:::scores --> E
  E[by-group scorers]:::scores --> D[score_dict]:::scores
  D --> F[score_file]:::files
```
<!-- fairlearn-score-overview-end -->
### Lifelines Execution Flows

```mermaid
flowchart LR
  A[load]:::data -->  B[fit survival model]:::model --> C[score]:::scores
```

```mermaid
flowchart LR
  A[load historical model data]:::data --> A2[calculate failures benign success (e.g. accuracy)]:::data --> B[fit survival model]:::model --> C[score]:::scores
```

```mermaid
flowchart LR
  A[load historical model data]:::data --> A2[calculate failures from attack success]:::data --> B[fit survival model]:::model --> C[score]:::scores
```

### Yellowbrick Execution Flows

```mermaid
flowchart TD
  A[ExperimentConfig]:::experiment
  B[Data]:::data
  C[Model]:::model
  D[Attack]:::attack
  E[Detect]:::detector
  F[Score stage: train / test / val / all]:::scores
  G[Persist stage: outputs, files, artifacts]:::files
  B --> V1[Data Visualizers] --> G
  C --> V2[Model Visualizers] --> G
  A --> B --> C -. ignored .-> D --> E --> F -. ignored .-> G
```

### Seaborn

```mermaid
flowchart LR
  A[DataConfig]:::data
  A --> V1[Tabular Data Visualizers]
```
