# Model

## Basic flow state

`initialize -> train -> predict/score -> persist`.

## Purpose

Define user-facing model runtime owner behavior, including split mode scoring,
defense stage execution, persistence outputs, and boundaries for framework
adapters and plugin integrations.

## Capabilities

- Resolve and initialize model implementations from configuration.
- Train and evaluate classifiers/regressors on canonical split modes.
- Apply defense stages in deterministic runtime order.
- Persist config/state artifacts and expose reusable prediction payloads.
- Integrate with {doc}`/api/model/train` and {doc}`/api/model/defend` runtime sub-objects.

## Outputs

- Model artifacts (`model_file`, runtime state files).
- Prediction/probability payloads for train/test/val splits.
- Runtime timings for training, prediction, and scoring.
- Model score dictionaries and persisted score outputs.

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
are documented in {doc}`/developers/model/model`.

```{eval-rst}
.. automodule:: deckard.model
   :members:
   :show-inheritance:
```

## Integrations

Integration-specific capabilities are documented in dedicated pages so this
core API page remains focused on base model behavior:

- Framework integration: {doc}`../pytorch`
- Plugin integrations: {doc}`../fairlearn`, {doc}`../lifelines`, {doc}`../anjana`

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
evaluation. See {doc}`../attack` for paired attack orchestration and {doc}`../score`
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
  name: sklearn.linear_model.LogisticRegression
   classifier: true
   model_params:
      max_iter: 500
```

## Implementation Notes

Detailed model runtime contracts (stage ordering, trainer/defense internals,
and framework adapter boundaries) are documented in
{doc}`/developers/model/model`.

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

- {doc}`../data` — data configuration and loading
- {doc}`/api/model/train` — training runtime mixins and trainer-defense behavior
- {doc}`/api/model/defend` — defense pipeline and defense-family mixins
- {doc}`../experiment` — experiment orchestration
- {doc}`../attack` — attack configuration
- {doc}`../score` — scoring framework
- {doc}`../pytorch` — PyTorch model integration
- {doc}`../anjana` — anonymization-aware models
- {doc}`../lifelines` — survival model configuration
- {doc}`../utils` — utility functions
