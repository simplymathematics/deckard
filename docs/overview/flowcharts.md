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
- Visualization -> dark gray

## Default Workflow

Canonical orchestration entrypoint: {meth}`deckard.experiment.ExperimentConfig.run`.

```mermaid
flowchart TD
  A[ExperimentConfig.run]:::experiment
  B[Data stage: pre-load]:::data
  C[Data stage: pre-sample]:::data
  D[Data stage: post-sample]:::data
  E[Optional pipeline stage: post-pipeline]:::data
  F[Model stage: initialize_model / train_or_load_model]:::model
  G[Model stage: evaluate_model / __call__]:::model
  H[Optional defense stage: pre_art_defense / pre_fit / post_fit_pre_predict]:::model
  I[Optional attack stage: pre-attack / post-attack]:::attack
  J[Optional detector stage: pre-fit / post-fit / pre-detect / post-detect]:::detector
  K[Score stage: train / test / val / all]:::scores
  L[Persist stage: outputs, files, artifacts]:::files
  M[Visualize and review results]:::experiment

  A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K --> L --> M
  E -. optional .-> F
  G -. optional .-> H
  H -. optional .-> I
  I -. optional .-> J
  J --> K

  class A experiment;
  class B,C data;
  class F,G,H model;
  class I attack;
  class J detector;
  class K scores;
  class L files;
  class M experiment;
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
  A[load]:::data --> B[sample]:::data
  B --> C[optional transform]:::data
  C --> D[score]:::scores
```
<!-- core-data-overview-end -->

### Model API Overview

<!-- core-model-overview-start -->
```mermaid
flowchart LR
  A[initialize model]:::model --> B[train or load]:::model
  B --> C[optional defense]:::model
  C --> D[predict]:::model
  D --> E[persist]:::files
```
<!-- core-model-overview-end -->

### Model Trainer Scope

<!-- core-model-trainer-scope-start -->
```mermaid
flowchart LR
  A[trainer alias]:::model --> B[sklearn]:::model
  A --> C[pretrained]:::model
  A --> D[partial_fit]:::model
  A --> E[partial_fit_pruning]:::model
  A --> F[pruning]:::model
  A --> G[pytorch]:::model
```
<!-- core-model-trainer-scope-end -->

### Defense Subtypes

<!-- core-defense-subtypes-start -->
```mermaid
flowchart LR
  A[defense]:::model --> B[anjana]:::model
  A --> C[fairlearn.reductions]:::model
  A --> D[fairlearn.adversarial]:::model
  A --> E[fairlearn.postprocessing]:::model
  A --> F[other defenses]:::model
  B --> G[pre_art_defense]:::model
  C --> H[pre_fit]:::model
  D --> I[post_fit_pre_predict]:::model
  E --> I
  F --> I
```
<!-- core-defense-subtypes-end -->

### Attack API Overview

<!-- core-attack-overview-start -->
```mermaid
flowchart LR
  A[attack family]:::attack --> B[evasion]:::attack
  A --> C[poisoning]:::attack
  A --> D[inference]:::attack
  A --> E[extraction]:::attack
  B --> F[score]:::scores
  C --> F
  D --> F
  E --> F
```
<!-- core-attack-overview-end -->

### Detector API Overview

<!-- core-detector-overview-start -->
```mermaid
flowchart LR
  A[train]:::detector --> B[filter]:::detector
  B --> C[evasion / poison filtering]:::detector
```
<!-- core-detector-overview-end -->

### Score API Overview

<!-- core-score-overview-start -->
```mermaid
flowchart LR
  A[data scorers]:::scores --> D[score_dict]:::scores
  B[model scorers]:::scores --> D
  C[attack scorers]:::scores --> D
  E[group scorers]:::scores --> D
  D --> F[score_file]:::files
```
<!-- core-score-overview-end -->

### Experiment API Overview

<!-- core-experiment-overview-start -->
```mermaid
flowchart LR
  A[load]:::experiment --> B[sample]:::experiment
  B --> C[train]:::experiment
  C --> D[defense]:::experiment
  D --> E[attack]:::attack
  E --> F[detector]:::detector
  F --> G[score]:::scores
  G --> H[persist]:::files
  H --> I[review / post-hoc analysis]:::experiment
```
<!-- core-experiment-overview-end -->

### Persistence API Overview

<!-- core-persistence-overview-start -->
```mermaid
flowchart LR
  A[outputs]:::files --> B[files]:::files
  B --> C[artifacts]:::files
```
<!-- core-persistence-overview-end -->

## Core API Scoping Flowcharts

The charts below focus on how each core object scopes its smaller pieces.

### Data API

<!-- core-data-scope-start -->
```mermaid
flowchart LR
  A[data]:::data --> B[sampler]:::data
  A --> C[pipeline]:::data
  B --> D[pre-sample]:::data
  D --> E[post-sample]:::data
  C -. optional .-> F[post-pipeline]:::data
```
<!-- core-data-scope-end -->

### Data Runtime Flow

<!-- core-data-flowchart-start -->
```mermaid
flowchart LR
  A[pre-load]:::data --> B[pre-sample]:::data
  B --> C[post-sample]:::data
  C -. optional .-> D[post-pipeline]:::data
  D --> E[data_score]:::scores
```
<!-- core-data-flowchart-end -->

### Model API

<!-- core-model-scope-start -->
```mermaid
flowchart LR
  A[initialize_model]:::model --> B[train_or_load_model / trainer]:::model
  B --> C[evaluate_model]:::model
  B -. optional .-> D[pre_art_defense]:::model
  D --> E[pre_fit]:::model
  E --> F[post_fit_pre_predict]:::model
  F --> C
```
<!-- core-model-scope-end -->

### Model Runtime Flow

<!-- core-model-flowchart-start -->
```mermaid
flowchart LR
  A[initialize_model]:::model --> B[train_or_load_model]:::model
  B --> C[evaluate_model]:::model
  C --> D[persist_outputs]:::files
  B -. optional .-> E[pre_art_defense / pre_fit / post_fit_pre_predict]:::model
  E --> C
```
<!-- core-model-flowchart-end -->

### Attack API

<!-- core-attack-family-start -->
```mermaid
flowchart LR
  A[attack family]:::attack --> B[evasion]:::attack
  A --> C[poisoning]:::attack
  A --> D[inference]:::attack
  A --> E[extraction]:::attack
  B --> F[blackbox_evasion / whitebox_evasion]:::attack
  C --> G[poisoning / PoisoningAttackSVM]:::attack
  D --> H[membership_inference / attribute_inference / model_inversion]:::attack
  H -. optional .-> I[reconstruction]:::attack
```
<!-- core-attack-family-end -->

### Attack Runtime Flow

<!-- core-attack-flowchart-start -->
```mermaid
flowchart LR
  A[pre-attack]:::attack --> B[post-attack]:::attack
  B --> C[score]:::scores
```
<!-- core-attack-flowchart-end -->

### Experiment API

<!-- core-experiment-flowchart-start -->
```mermaid
flowchart LR
  A[load]:::experiment --> B[sample]:::experiment
  B --> C[train]:::experiment
  C --> D[defense]:::experiment
  D --> E[attack]:::attack
  E --> F[detector]:::detector
  F --> G[score]:::scores
  G --> H[persist]:::files
  H --> I[plot / post-hoc analysis]:::experiment
```
<!-- core-experiment-flowchart-end -->

### Persistence API

<!-- core-persistence-flowchart-start -->
```mermaid
flowchart LR
  A[score_file]:::files --> B[data_file / model_file]:::files
  B --> C[attack_file / detector_model_file]:::files
  C --> D[metadata_file / artifacts]:::files
```
<!-- core-persistence-flowchart-end -->

### Detector API

<!-- core-detector-mode-start -->
```mermaid
flowchart LR
  A[mode=train]:::detector --> B[detector_training_time]:::detector
  A --> C[mode=filter]:::detector
  C --> D[filter_mode:auto / poison / evasion]:::detector
  D -. poison .-> E[poison filtering]:::detector
  D -. evasion .-> F[evasion filtering]:::detector
```
<!-- core-detector-mode-end -->

### Score API

<!-- core-score-composition-start -->
```mermaid
flowchart LR
  A[data scorer: pre-sample / post-sample]:::scores --> D[group_scorers / fairlearn]:::scores
  B[model scorer: train / test / val]:::scores --> D
  C[attack scorer: attack / attack-val]:::scores --> D
  D --> E[score_dict]:::scores
  E --> F[score_file]:::files
```
<!-- core-score-composition-end -->

## Extension Flowcharts

### ANJANA Execution Flows

<!-- anjana-execution-flows-start -->
## Execution Flows

### Data Flow

```mermaid
flowchart TD
    A[Data load] --> B[before_sample privacy hook execution]
    B --> C[sampled privacy-aware payload]
```

### Pipeline Flow

```mermaid
flowchart TD
    A[privacy-constrained input] --> B[pipeline stages]
    B --> C[policy-compliant transformed payload]
```

### Defense Flow

```mermaid
flowchart TD
    A[model runtime with ANJANA] --> B[map defense to pre_art_defense]
    B --> C[apply defense path]
```

### Scoring Flow

```mermaid
flowchart TD
    A[privacy run outputs] --> B[anjana scorer execution]
    B --> C[merge privacy metrics into score_dict]
```

### Plot Flow

```mermaid
flowchart TD
    A[persisted privacy metrics/artifacts] --> B[plot adapter]
    B --> C[render privacy diagnostics]
```
<!-- anjana-execution-flows-end -->

### sklearn Execution Flows

<!-- sklearn-execution-flows-start -->
## Execution Flows

### Data Flow

```mermaid
flowchart TD
  A[DataConfig load/sample] --> B[sklearn-ready tabular payload]
  B --> C[pass to sklearn trainer/runtime]
```

### Pipeline Flow

```mermaid
flowchart TD
  A[before_pipeline hook] --> B[fit_pre_sample -> fit_X -> fit_y -> fit_Xy]
  B --> C[after_pipeline hook]
```

### Defense Flow

```mermaid
flowchart TD
  A[trained or loaded sklearn model] --> B{defense configured?}
  B -- yes --> C[map to pre_art_defense/pre_fit/post_fit_pre_predict]
  C --> D[apply defense and optional retrain]
  B -- no --> E[use baseline model path]
```

### Scoring Flow

```mermaid
flowchart TD
  A[predictions available] --> B[mode select train/test/val]
  B --> C[score stage dispatch]
  C --> D[merge score_dict and persist score_file]
```

### Plot Flow

```mermaid
flowchart TD
  A[persisted sklearn artifacts] --> B[PlotConfig backend adapter]
  B --> C[render and persist figure outputs]
```
<!-- sklearn-execution-flows-end -->

### Fairlearn Execution Flows

<!-- fairlearn-execution-flows-start -->
## Execution Flows

### Data Flow

```mermaid
flowchart TD
  A[Data load + sensitive attrs] --> B[fairlearn data policy hooks]
  B --> C[prepared fairness-aware split payload]
```

### Pipeline Flow

```mermaid
flowchart TD
  A[before_pipeline] --> B[fairlearn preprocessing/postprocessing stage]
  B --> C[after_pipeline]
```

### Defense Flow

```mermaid
flowchart TD
  A[fairlearn model runtime] --> B{defense type}
  B -- reductions --> C[pre_fit stage]
  B -- adversarial/postprocessing --> D[post_fit_pre_predict stage]
```

### Scoring Flow

```mermaid
flowchart TD
  A[predictions + groups] --> B[group metric scorer execution]
  B --> C[fairness merge last into score_dict]
```

### Plot Flow

```mermaid
flowchart TD
  A[persisted fairness metrics] --> B[plot adapter]
  B --> C[group fairness visual diagnostics]
```
<!-- fairlearn-execution-flows-end -->

### PyTorch Execution Flows

<!-- pytorch-execution-flows-start -->
## Execution Flows

### Data Flow

```mermaid
flowchart TD
  A[DataConfig load/sample] --> B[tensor conversion and dataloader prep]
  B --> C[pytorch model runtime]
```

### Pipeline Flow

```mermaid
flowchart TD
  A[data pipeline hooks] --> B[feature transforms before torch training]
  B --> C[dataloader consumes transformed payload]
```

### Defense Flow

```mermaid
flowchart TD
  A[torch trainer output] --> B{defense configured?}
  B -- yes --> C[map to canonical defense stage]
  C --> D[apply defense, optional retrain]
  B -- no --> E[baseline inference path]
```

### Scoring Flow

```mermaid
flowchart TD
  A[predictions and logits] --> B[mode train/test/val]
  B --> C[scorer execution]
  C --> D[persist score artifacts]
```

### Plot Flow

```mermaid
flowchart TD
  A[persisted torch artifacts] --> B[plot backend adapter]
  B --> C[render and store outputs]
```
<!-- pytorch-execution-flows-end -->

### Lifelines Execution Flows

<!-- lifelines-execution-flows-start -->
## Execution Flows

### Data Flow

```mermaid
flowchart TD
  A[data load/split] --> B[survival target/time preparation]
  B --> C[lifelines-ready payload]
```

### Pipeline Flow

```mermaid
flowchart TD
  A[pipeline transforms] --> B[survival feature engineering]
  B --> C[model runtime input]
```

### Defense Flow

```mermaid
flowchart TD
  A[lifelines model path] --> B{defense configured?}
  B -- yes --> C[delegate to canonical model defense stages]
  B -- no --> D[baseline survival path]
```

### Scoring Flow

```mermaid
flowchart TD
  A[survival predictions] --> B[c-index and survival metrics]
  B --> C[merge and persist score artifacts]
```

### Plot Flow

```mermaid
flowchart TD
  A[persisted survival outputs] --> B[survival plot backend]
  B --> C[render and persist charts]
```
<!-- lifelines-execution-flows-end -->

### Seaborn Execution Flows

<!-- seaborn-execution-flows-start -->
## Execution Flows

### Data Flow

```mermaid
flowchart TD
    A[data or experiment artifact input] --> B[seaborn data adapter]
    B --> C[plot-ready dataframe payload]
```

### Pipeline Flow

```mermaid
flowchart TD
    A[plot input payload] --> B[optional pre-plot transform]
    B --> C[seaborn render input]
```

### Defense Flow

```mermaid
flowchart TD
    A[source model/experiment defense outputs] --> B[seaborn reads defended artifacts]
    B --> C[no plugin-local defense execution]
```

### Scoring Flow

```mermaid
flowchart TD
    A[persisted score payload] --> B[metric selection for charting]
    B --> C[annotated seaborn visualization]
```

### Plot Flow

```mermaid
flowchart TD
    A[seaborn backend adapter] --> B[render statistical figure]
    B --> C[persist figure asset and metadata]
```
<!-- seaborn-execution-flows-end -->

### Yellowbrick Execution Flows

<!-- yellowbrick-execution-flows-start -->
## Execution Flows

### Data Flow

```mermaid
flowchart TD
    A[experiment/model artifacts] --> B[yellowbrick input adapter]
    B --> C[diagnostic payload]
```

### Pipeline Flow

```mermaid
flowchart TD
    A[diagnostic payload] --> B[optional pre-diagnostic transform]
    B --> C[yellowbrick runtime input]
```

### Defense Flow

```mermaid
flowchart TD
    A[defended model outputs] --> B[yellowbrick consumes defended predictions]
    B --> C[no plugin-local defense execution]
```

### Scoring Flow

```mermaid
flowchart TD
    A[score artifacts] --> B[diagnostic metric selection]
    B --> C[chart annotations and summaries]
```

### Plot Flow

```mermaid
flowchart TD
    A[yellowbrick backend adapter] --> B[render diagnostic figure]
    B --> C[persist figure asset + metadata]
```
<!-- yellowbrick-execution-flows-end -->

### Transformers Execution Flows

<!-- transformers-execution-flows-start -->
## Execution Flows

### Data Flow

```mermaid
flowchart TD
  A[DataConfig split payload] --> B[tokenizer and encoding adapters]
  B --> C[transformers runtime inputs]
```

### Pipeline Flow

```mermaid
flowchart TD
  A[pipeline transforms] --> B[encoded feature payload]
  B --> C[transformers trainer consumption]
```

### Defense Flow

```mermaid
flowchart TD
  A[transformers model state] --> B{defense configured?}
  B -- yes --> C[canonical defense stage dispatch]
  C --> D[defended inference/train path]
  B -- no --> E[baseline transformers path]
```

### Scoring Flow

```mermaid
flowchart TD
  A[predictions/probabilities] --> B[mode and stage selection]
  B --> C[scorer execution]
  C --> D[persist score artifacts]
```

### Plot Flow

```mermaid
flowchart TD
  A[persisted transformer outputs] --> B[plot adapter]
  B --> C[render diagnostics]
```
<!-- transformers-execution-flows-end -->
