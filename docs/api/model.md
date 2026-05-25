# Model

## Introduction

This page is the canonical home for model module behavior and API details.
It includes runtime lifecycle behavior, defense-stage semantics, persistence,
and extension points.

The {mod}`~deckard.model` module defines the {class}`~deckard.model.ModelConfig`
dataclass, which provides a complete pipeline for **model configuration,
training, evaluation, defense application, and persistence**.
It supports dynamic scikit-learn model instantiation, configurable parameters,
CLI execution,
and integration with the {mod}`deckard.data` module.

The canonical model runtime contract, trainer flow, and defense-stage behavior
are documented in {doc}`../developers/model`.

```{eval-rst}
.. automodule:: deckard.model
   :members:
   :show-inheritance:
```

## Extensions

### Fairlearn Extension

The fairlearn extension provides fairness-aware model behavior, including
group-sensitive fitting, scoring, and fairlearn defense wrappers. The public
model mixin is {class}`~deckard.plugins.fairlearn.model.FairnessBehaviorMixin`.
See also: {doc}`fairlearn`.

```{eval-rst}
.. automodule:: deckard.plugins.fairlearn.model
   :members:
   :show-inheritance:
```

### Pytorch extension

The Pytorch extension provides PyTorch-native model training, prediction, and
scoring through a {class}`~deckard.model.ModelConfig`-compatible API.
See also: {doc}`pytorch`.

```{eval-rst}
.. automodule:: deckard.frameworks.pytorch.model
   :members:
   :show-inheritance:
```

## Lifelines plugin

Survival-specific experiment orchestration is split into a dedicated optional
module.
See also: {doc}`lifelines`.

```{eval-rst}
.. automodule:: deckard.plugins.lifelines.model
   :members:
   :show-inheritance:
```

## Overview

{class}`~deckard.model.ModelConfig` automates the following steps:

- Dynamic instantiation of scikit-learn models via import strings (e.g. `sklearn.svm.SVC`)
- Training, prediction, and evaluation for both classification and regression
- Timing instrumentation for training, prediction, and scoring
- Config persistence via YAML state-machine artifacts
- Runtime model persistence via framework-native artifacts
- Hydra/YAML configuration for reproducibility and experiment management
- CLI support for one-line model training and testing

### Model scoring mode

{class}`~deckard.model.ModelConfig` supports split-aware scoring with
`score_mode` set to one of:

- `train`
- `test`
- `val`

The experiment layer can propagate this mode automatically so model scoring is
performed on the active split selected by experiment scoring policy.

### Supported frameworks

Currently supports:

- **scikit-learn** — via {class}`~deckard.model.ModelConfig`
- **PyTorch** — via {class}`~deckard.frameworks.pytorch.model.PytorchModelConfig`
- **Fairlearn (sklearn)** — via {class}`~deckard.plugins.fairlearn.model.FairlearnModelConfig`
- **Fairlearn (PyTorch)** — via {class}`~deckard.plugins.fairlearn.model.FairlearnPytorchModelConfig`

### Defense pipeline integration

Model configs can compose deckard defense pipelines used during robustness
evaluation. See {doc}`attack` for paired attack orchestration and {doc}`score`
for attack-aware scorer profiles.

Defense application is stage-aware and follows the canonical model stages:

- `pre_art_defense`
- `pre_fit`
- `post_fit_pre_predict`

Pretrained models that receive a fit-time defense are retrained after a
pre-defense snapshot is cached, so the old timing and prediction state remain
available for analysis.

Common ART defense components referenced by deckard model defenses:

- [`FeatureSqueezing`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/preprocessor.html#feature-squeezing)
- [`SpatialSmoothing`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/preprocessor.html#spatial-smoothing)
- [`GaussianAugmentation`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/preprocessor.html#gaussian-augmentation)
- [`AdversarialTrainer`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/trainer.html#adversarial-training)

## Examples

```{seealso}

   Notebook-based model workflows, including sklearn, defense pipelines,
   Fairlearn, and PyTorch models, are documented in:

   - {doc}`notebooks/sklearn.ipynb </notebooks/sklearn>`
   - {doc}`notebooks/fairlearn.ipynb </notebooks/fairlearn>`
   - {doc}`notebooks/pytorch.ipynb </notebooks/pytorch>`
   - {doc}`notebooks/art_defenses.ipynb </notebooks/art_defenses>`

```

## Minimal YAML Example

```yaml
model:
   _target_: deckard.model.base.ModelConfig
   model_type: sklearn.linear_model.LogisticRegression
   classifier: true
   model_params:
      max_iter: 500
```

## Internals

### Timing and logging

All major operations (training, prediction, scoring, saving/loading) record
wall-clock time
and log via Python’s `logging` module.

### Scoring

- For classifiers: accuracy, precision, recall, and F1 score.
- For regressors: MSE, RMSE, and MAE.

### Persistence

Use the public model persistence interfaces:

- :meth:`deckard.model.ModelConfig.save`
- :meth:`deckard.model.ModelConfig.load`
- :meth:`deckard.model.ModelConfig.save_model`
- :meth:`deckard.model.ModelConfig.load_model`
- `model(data, model_file=...)` for automatic load-or-train behavior

Canonical policy:

- `save`/`load` persist and restore config objects as `.yaml`/`.yml`.
- `save_model`/`load_model` persist and restore runtime model objects.
- Runtime model extensions are framework-specific:
  - PyTorch runtime artifacts use `.pt`.
  - scikit-learn runtime artifacts use `.pkl` or `.joblib`.

For {class}`~deckard.frameworks.pytorch.model.PytorchModelConfig`, checkpointing
produces YAML config records that reference runtime model-state artifacts:

- `model_file` entries point to YAML config artifacts.
- `model_state_file` entries point to `.pt` runtime state artifacts.

Public API example (automatic load-or-train): see the
{doc}`notebooks/sklearn.ipynb </notebooks/sklearn>`
notebook for executed save/load examples.

Public API example (PyTorch config + runtime save/load): see the
{doc}`notebooks/pytorch.ipynb </notebooks/pytorch>`
notebook for executed YAML + `save_model`/`load_model` patterns.

#### Pre-trained torch models

There are two supported patterns:

1. Load a previously saved deckard PyTorch config via `load(filepath)` and
load runtime state via `load_model(filepath)`.
1. Point `model_type` to a custom constructor/class that returns an already
   initialized {class}`torch.nn.Module` (for example, one that internally loads external
   pre-trained weights), then run normal deckard training/evaluation.

If you want inference-only behavior from a pre-trained checkpoint, load it via
`load` + `load_model` and then call the model with `model_file`/prediction
outputs as
needed, without requiring private methods.

## Troubleshooting

- **Model not fitted error** — train the model before calling
  :meth:`deckard.model.ModelConfig.save_model` or predictions.
- **Hydra config not found** — ensure the YAML file path is valid or use inline overrides.
- **Artifact deserialization errors** — verify runtime artifact type and extension
  match the framework policy (PyTorch `.pt`, sklearn `.pkl`/`.joblib`).
- **CLI argument conflicts** — use `conflict_handler='resolve'` when composing parsers.
- **Probability prediction errors** — set `--probability` only for models that
  support `predict_proba()`.

### See also

- {doc}`data` — data configuration and loading
- {doc}`train` — training runtime mixins and trainer-defense behavior
- {doc}`defend` — defense pipeline and defense-family mixins
- {doc}`experiment` — experiment orchestration
- {doc}`attack` — attack configuration
- {doc}`score` — scoring framework
- {doc}`pytorch` — PyTorch model integration
- {doc}`anjana` — anonymization-aware models
- {doc}`lifelines` — survival model configuration
- {doc}`utils` — utility functions
