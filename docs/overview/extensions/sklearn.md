# sklearn Framework Overview

This overview focuses on execution order for sklearn framework wrappers.

For comprehensive hook ownership and policy details, see
[Plugin and Hook Execution Reference](../../developers/hooks.md).

Related docs:

- [Data API](../../api/data)
- [Pipeline API](../../api/pipeline)
- [Model API](../../api/model)
- [Training API](../../api/train)
- [Defense API](../../api/defend)
- [Attack API](../../api/attack)
- [Detector API](../../api/detector)
- [Scoring Overview](../scoring.md)
- [File API](../../api/file)
- [Artifacts API](../../api/artifacts)
- [Experiment Guide](../experiment.md)
- [Plot API](../../api/plot)
- [Utils API](../../api/utils)

## Execution Order

1. Data split and optional pipeline transform execution.
2. Model trainer strategy resolution and training/load path.
3. Defense stage mapping and application.
4. Split-scoped scoring and score merge.
5. Artifact/file persistence and optional downstream plotting.

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

## YAML Examples

```yaml
data:
    _target_: deckard.data.base.DataConfig
    pipeline:
        scale:
            name: sklearn.preprocessing.StandardScaler
            fit_X: true

model:
    _target_: deckard.model.base.ModelConfig
    model_type: sklearn.ensemble.RandomForestClassifier
    trainer:
        _target_: deckard.model.trainer.base.SklearnTrainerConfig
```

```yaml
score:
    model:
        scorers:
            accuracy:
                score_name: accuracy
                score_function: sklearn.metrics.accuracy_score
```