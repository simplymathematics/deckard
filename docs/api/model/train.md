# Training Runtime

## Contract References

- Canonical API contract page: {doc}`train`
- Developer authoring contract: {doc}`../developers/trainers`
- Shared config/mixin/plugin contracts: {doc}`../developers/configs`, {doc}`../developers/mixins`, {doc}`../developers/plugins`

## Introduction

This page is the canonical home for training runtime behavior and API details.
It covers model trainer composition, pretrained/pruning pathways,
and trainer-defense integration.

## Overview

Deckard training behavior is split between core model orchestration in
{class}`deckard.model.base.ModelConfig` and reusable training/runtime mixins in
{mod}`deckard.model._mixins`.

Trainer defenses are configured separately through
{class}`deckard.model.defense.trainer.TrainerDefenseConfig`.

Important distinction:

- Trainer configuration objects in this page define how model fitting is
  orchestrated.
- ART retrainer defenses are defense objects (configured under `model.defense`)
  and are documented in {doc}`defend`.

## Parent Config and Mixin Map

- {class}`deckard.model.base.ModelConfig` is the parent runtime config for
  fit/predict/score orchestration.
- {class}`deckard.model._mixins.PretrainedModelMixin` provides cached model load
  behavior.
- {class}`deckard.model._mixins.ModelPrunerMixin` provides Optuna-style pruning checks.
- {class}`deckard.model.base.ModelConfig` owns fit entrypoints, plugin hook
  execution, and runtime-state propagation directly.
- {class}`deckard.model.defense.trainer.TrainerDefenseConfig` composes
  {class}`deckard.model.defense.trainer.TrainerDefenseMixin` with
  {class}`deckard.model.defense.base.DefensePipelineConfig` for adversarial training
  defenses.

## External References

- [Optuna Trial API](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html)
- [ART adversarial training](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/trainer.html#adversarial-training)

## API Reference

```{eval-rst}
.. automodule:: deckard.model._mixins
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.model.defense.trainer
   :members:
   :show-inheritance:
```

## Minimal YAML Example

```yaml
model:
  _target_: deckard.model.base.ModelConfig
  model_type: sklearn.ensemble.RandomForestClassifier
  classifier: true
  fit_params:
    sample_weight: null
```

## See also

- {doc}`model`
- {doc}`defend`
- {doc}`score`
- {doc}`../developers/trainers`
