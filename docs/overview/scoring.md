# Scoring Guide for Base Config Objects

This guide documents scoring behavior for the base runtime configs:

- `DataConfig`
- `ModelConfig`
- `DetectorConfig`
- `AttackConfig`

It covers scoring defaults, `mode` and `stage` semantics, output conventions, and custom scorer examples.

## Core Concepts

### `mode` vs `stage`

- `mode` selects the split/runtime context (`train`, `test`, `val`, plus scope-specific values like `attack` and `pre-sample`).
- `stage` identifies lifecycle boundaries (`pre-sample`, `post-sample`, `pre-defense`, `post-defense`, `pre-filter`, `post-filter`, `benign`, `adversarial`, etc).

Rule:

- `mode` answers: where does this data come from?
- `stage` answers: what step of the pipeline/runtime produced this score?

### Scorer Interface

Most runtime scorers are `ScorerDictConfig` (or subclasses) receiving:

- `y_true`
- `y_pred`
- optional `y_proba`
- `mode`
- `stage`

### Output Shape

Scorers may produce nested stage/mode outputs internally. Runtime owners can flatten for backward compatibility.

## Defaults by Base Config

## `DataConfig`

Defaults:

- `score_mode`: `pre-sample`
- `scorer`:
  - classification: `deckard.score.data.DefaultDataClassificationConfig`
  - regression: `deckard.score.data.DefaultDataRegressionConfig`

Typical usage:

- pre/post sampling diagnostics
- split-aware data diagnostics on train/test/val

Example:

```python
from deckard.data.base import DataConfig

cfg = DataConfig(
    dataset_name="make_classification",
    data_params={"n_samples": 100, "n_features": 10},
)

scores = cfg._score(mode="pre-sample", stage="pre-sample")
```

## `ModelConfig`

Defaults:

- `score_mode`: `test`
- `scorer`:
  - classification: `deckard.score.base.DefaultClassifierConfig`
  - regression: `deckard.score.base.DefaultRegressorConfig`

Typical usage:

- score train/test/val predictions
- include probability-based metrics when `y_proba` is available

Example:

```python
from deckard.model.base import ModelConfig

model_cfg = ModelConfig(
    model_type="sklearn.linear_model.LogisticRegression",
    classifier=True,
    defense={
        ""
    }
)

out = model_cfg._score(
    y_true=[0, 1, 1],
    y_pred=[0, 1, 0],
    mode="test",
    stage="post-defense",
)
```

## `DetectorConfig`

Defaults:

- `scorer`: `DetectorScorerConfig(classifier=True)` when not provided

Typical usage:

- emit pre-filter baseline scores
- emit post-filter detector scores
- prefix detector outputs with `detector_`

Example:

```python
from deckard.detector.base import DetectorConfig

detector_cfg = DetectorConfig(
    detector_type="art.defences.detector.evasion.BinaryInputDetector",
)

# runtime emits stage="pre-filter" and stage="post-filter"
```

## `AttackConfig`

Defaults:

- `mode`: `auto`
- `scorer`: `deckard.score.attack.AttackScorerConfig`

Typical usage:

- attack-profile scoring for evasion/membership/attribute paths
- stage-aware attack metrics (`benign`, `adversarial`)
- shared configurable comparison scoring for benign-vs-adversarial and victim-vs-extracted paths


Example:

```python
from deckard.attack.base import AttackConfig

attack_cfg = AttackConfig(
    attack_type="art.attacks.evasion.FastGradientMethod",
    attack_params={"eps": 0.1},
    mode="test",
)

scores = attack_cfg(data=my_data_cfg, model=my_model_cfg)
```

## Custom Scorers

## 1. Custom `ScorerConfig`

```python
from deckard.score.base import ScorerConfig

my_acc = ScorerConfig(
    score_name="accuracy",
    score_function="sklearn.metrics.accuracy_score",
)
```

## 2. Custom `ScorerDictConfig`

```python
from deckard.score.base import ScorerConfig, ScorerDictConfig

custom = ScorerDictConfig(
    scorers={
        "accuracy": ScorerConfig(
            score_name="accuracy",
            score_function="sklearn.metrics.accuracy_score",
        ),
        "f1": ScorerConfig(
            score_name="f1",
            score_function="sklearn.metrics.f1_score",
            score_params={"average": "weighted", "zero_division": 0},
        ),
    },
)
```

Attach to a base config:

```python
model_cfg = ModelConfig(
    model_type="sklearn.linear_model.LogisticRegression",
    classifier=True,
    scorer=custom,
)
```

## 3. Attack comparison scorer override

For attack comparison paths, configure `attack_params["comparison_scorer"]`:

```python
attack_cfg = AttackConfig(
    attack_type="art.attacks.extraction.CopycatCNN",
    attack_params={
        "comparison_scorer": {
            "scorers": {
                "accuracy": {
                    "score_name": "accuracy",
                    "score_function": "sklearn.metrics.accuracy_score",
                }
            }
        }
    },
)
```

## YAML Example (Hydra Style)

```yaml
scorers:
  accuracy:
    score_name: accuracy
    score_function: sklearn.metrics.accuracy_score
  recall:
    score_name: recall
    score_function: sklearn.metrics.recall_score
    score_params:
      average: weighted
      zero_division: 0
```

## Recommended Practices

- Keep stage emission explicit at runtime boundaries.
- Use `mode` for split selection and `stage` for lifecycle semantics.
- Raise explicit errors for unsupported stage/mode combinations.
- Keep scoring configurable via scorer configs, not ad-hoc subtype metric code.
- Preserve compatibility aliases only at output boundaries.

## Quick Checklist

- Is `mode` selecting the intended split?
- Is `stage` describing the intended runtime boundary?
- Is scoring flowing through shared config-based scorer paths?
- Are compatibility keys preserved where downstream consumers require them?
