# Plugin and Hook Execution Reference

This document is the canonical developer reference for plugin and framework hook
execution in Deckard.

Use this page for implementation details. Overview pages should stay concise and
focus on execution order.

## Scope

This reference covers hook and runtime flow behavior for:

- frameworks: sklearn, pytorch, transformers
- plugins: anjana, fairlearn, lifelines, seaborn, yellowbrick

Across runtime concerns:

- data
- pipeline
- defense
- scoring
- plot

## Hook Ownership Matrix

| Concern | Primary owner | Hook surface examples |
| --- | --- | --- |
| Data load/sample/pipeline | DataConfig | before_load_data, before_sample, before_pipeline, after_pipeline |
| Data score stages | DataConfig + scorer | before_score_pre_sample, after_score_post_pipeline |
| Model trainer lifecycle | ModelConfig + trainer runtime | before_train_or_load_model, after_train_or_load_model |
| Model defense stages | ModelConfig + DefensePipeline | pre_art_defense, pre_fit, post_fit_pre_predict |
| Attack lifecycle | AttackConfig | pre-attack, post-attack, benign, adversarial |
| Detector lifecycle | DetectorConfig | pre-fit, post-fit, pre-detect, post-detect |
| Experiment orchestration | ExperimentConfig | before/after load, sample, train, attack, defense, score, persist |
| Plot backend runtime | PlotConfig + backend wrappers | backend-specific setup/render hooks |

## Canonical End-to-End Composition

```mermaid
flowchart TD
    A[ExperimentConfig.__call__] --> B[DataConfig load/sample/pipeline]
    B --> C[ModelConfig trainer + defense stages]
    C --> D[AttackConfig path]
    D --> E[DetectorConfig path]
    E --> F[Scorer execution by mode and stage]
    F --> G[Plot backends consume runtime artifacts]
    G --> H[Artifacts/File persistence]
```

The framework/plugin layers should adapt behavior at these boundaries without
replacing canonical orchestration ownership.

## Framework Flows

### sklearn

```mermaid
flowchart TD
    A[DataConfig split + pipeline] --> B[sklearn estimator trainer]
    B --> C[defense stage mapping]
    C --> D[scorer mode train/test/val]
    D --> E[persist model/predictions/scores]
```

Typical config:

```yaml
model:
  _target_: deckard.model.base.ModelConfig
  model_type: sklearn.ensemble.RandomForestClassifier
  trainer:
    _target_: deckard.model.trainer.base.SklearnTrainerConfig
  defense:
    pipeline:
      - name: art.defences.preprocessor.FeatureSqueezing
        apply_fit: false
        apply_predict: true
```

### pytorch

```mermaid
flowchart TD
    A[data to tensor/dataloader] --> B[pytorch trainer runtime]
    B --> C[defense stage mapping]
    C --> D[score mode and stage execution]
    D --> E[persist checkpoints and score artifacts]
```

Typical config:

```yaml
model:
  _target_: deckard.frameworks.pytorch.model.PytorchModelConfig
  trainer:
    _target_: deckard.model.trainer.base.PytorchTrainerConfig
  model_type: my_file.py:MyModelClass
  model_params:
    input_features : 3 
    output_classes : 10
  fit_params:
    epochs: 3
    lr: 0.001
```

### transformers

```mermaid
flowchart TD
    A[tokenizer/encoding pipeline] --> B[transformer model wrapper]
    B --> C[defense and scorer dispatch]
    C --> D[persist runtime artifacts]
```

Typical config:

```yaml
model:
  _target_: deckard.frameworks.transformers.model.TransformersModelConfig
  model_type: transformers.AutoModelForSequenceClassification
  tokenizer: transformers.AutoTokenizer
```

## Plugin Flows

### anjana

```mermaid
flowchart TD
    A[data load] --> B[before_sample privacy hook]
    B --> C[sample/pipeline]
    C --> D[privacy scorer merge]
    D --> E[persist outputs]
```

Typical config:

```yaml
data:
  _target_: deckard.plugins.anjana.data.AnjanaDataConfig
  anjana_defense:
    k: 2
score:
  data:
    _target_: deckard.plugins.anjana.score.DefaultAnjanaScorerDictConfig
```

### fairlearn

```mermaid
flowchart TD
    A[data pipeline] --> B[fairness preprocessing/postprocessing hook]
    B --> C[fairlearn model/defense stage]
    C --> D[group metric scoring]
    D --> E[persist fairness metrics]
```

Typical config:

```yaml
data:
  _target_: deckard.plugins.fairlearn.data.FairlearnDataConfig
model:
  _target_: deckard.plugins.fairlearn.model.FairlearnModelConfig
score:
  model:
    _target_: deckard.plugins.fairlearn.score.FairlearnScorerDictConfig
```

### lifelines

```mermaid
flowchart TD
    A[survival dataset prep] --> B[lifelines model runtime]
    B --> C[survival scoring metrics]
    C --> D[survival plot backend integration]
```

Typical config:

```yaml
model:
  _target_: deckard.plugins.lifelines.model.SurvivalModelConfig
score:
  c_index:
    score_function: lifelines.utils.concordance_index
    stage : [test, val]
```

### seaborn

```mermaid
flowchart TD
    A[data/score artifacts] --> B[seaborn plot adapter]
    B --> C[render statistical plots]
    C --> D[persist plot artifacts]
```

Typical config:

```yaml
plot:
  _target_: deckard.plugins.seaborn.plot.SeabornPlotConfig
  files:
    plot_file: outputs/seaborn_summary.png
```

### yellowbrick

```mermaid
flowchart TD
    A[experiment/model artifacts] --> B[yellowbrick diagnostics adapter]
    B --> C[diagnostic scoring plots]
    C --> D[persist visual artifacts]
```

Typical config:

```yaml
plot:
  _target_: deckard.plugins.yellowbrick.plot.YellowbrickPlotConfig
  files:
    plot_file: outputs/yellowbrick_diagnostic.png
```

## Standardization Rules

- Keep core orchestration in base Config runtimes.
- Keep framework/plugin behavior policy-specific and thin.
- Emit stage-aware hooks, but keep score mode split-scoped.
- Persist through canonical file aliases and artifacts utilities.
- Add contract tests when introducing new hook stages or plugin branches.
