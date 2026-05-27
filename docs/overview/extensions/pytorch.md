# PyTorch Framework Overview

This overview focuses on execution order for PyTorch framework wrappers.

For comprehensive hook ownership and policy details, see
[Plugin and Hook Execution Reference](../../developers/hooks).

Related docs:

- [Data API](../../api/data)
- [Pipeline API](../../api/pipeline)
- [Model API](../../api/model)
- [Training API](../../api/train)
- [Defense API](../../api/defend)
- [Attack API](../../api/attack)
- [Detector API](../../api/detector)
- [Scoring Overview](../scoring)
- [File API](../../api/file)
- [Artifacts API](../../api/artifacts)
- [Experiment Guide](../experiment)
- [Plot API](../../api/plot)

## Execution Order

1. Data tensors/dataloader payload preparation.
2. PyTorch trainer execution (fit/load strategy).
3. Defense stage mapping and application.
4. Mode/stage scoring and score merge.
5. Checkpoint/artifact persistence and plot consumption.

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

## YAML Examples

```yaml
model:
    _target_: deckard.frameworks.pytorch.model.PytorchModelConfig
    trainer:
        _target_: deckard.model.trainer.base.PytorchTrainerConfig
    model_params:
        epochs: 3
        lr: 0.001
```

```yaml
score:
    model:
        scorers:
            accuracy:
                score_name: accuracy
                score_function: sklearn.metrics.accuracy_score
```
