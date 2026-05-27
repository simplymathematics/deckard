# Defense Runtime

## Contract References

- Canonical API contract page: {doc}`defend`
- Developer authoring contract: {doc}`../developers/defenses`
- Shared config/mixin/plugin contracts: {doc}`configs`, {doc}`../developers/mixins`, {doc}`../developers/plugins`

## Introduction

This page is the canonical home for defense runtime behavior and API details.
It covers defense-family composition, stage-aware application, and
runtime mixin dispatch.

## Overview

Deckard defense orchestration is centered on
{class}`deckard.model.defense.base.DefensePipelineConfig` and
{class}`deckard.model.defense.base.DefenseConfig`.

This layer composes model wrappers, ART defenses, and plugin hooks with
runtime mixin dispatch.

Important distinction:

- Trainer runtime configuration objects are documented in {doc}`train`.
- ART retrainer defenses are defense-family objects handled in this page under
  the `trainer` defense family.

## Parent Config and Mixin Map

- {class}`deckard.model.defense.base.DefensePipelineConfig` uses
  {class}`deckard.model.defense.base.DefensePipelineConfigBehaviorMixin` to normalize
  defense chains, enforce ordering, and run before/after hooks.
- {class}`deckard.model.defense.base.DefenseConfig` uses
  {class}`deckard.model.defense.base.ARTDefenseBehaviorMixin` to resolve ART wrapper
  classes, parse defense families, and dispatch subtype handlers.
- {class}`deckard.model.defense.preprocessor.PreprocessorDefenseConfig` inherits
  {class}`deckard.model.defense.preprocessor.PreprocessorDefenseMixin` and
  {class}`deckard.model.defense.base.DefensePipelineConfig`.
- {class}`deckard.model.defense.postprocessor.PostprocessorDefenseConfig` inherits
  {class}`deckard.model.defense.postprocessor.PostprocessorDefenseMixin` and
  {class}`deckard.model.defense.base.DefensePipelineConfig`.
- {class}`deckard.model.defense.trainer.TrainerDefenseConfig` inherits
  {class}`deckard.model.defense.trainer.TrainerDefenseMixin` and
  {class}`deckard.model.defense.base.DefensePipelineConfig`.
- {class}`deckard.model.transformer.TransformerDefenseConfig` inherits
  {class}`deckard.model.transformer.TransformerDefenseMixin` and
  {class}`deckard.model.defense.base.DefensePipelineConfig`.
- {class}`deckard.model.defense.detector.DetectorDefenseConfig` inherits
  {class}`deckard.model.defense.detector.DetectorDefenseMixin` and
  {class}`deckard.model.defense.base.DefensePipelineConfig`.
- {class}`deckard.model.defense.regularizer.RegularizerDefenseConfig` inherits
  {class}`deckard.model.defense.regularizer.RegularizerDefenseMixin` and
  {class}`deckard.model.defense.base.DefensePipelineConfig`.

## Defense Families

- `preprocessor`
- `postprocessor`
- `trainer`
- `transformer`
- `detector`
- `regularizer`

## Public Method Naming

Defense runtime APIs prefer verb-mode public methods.

- Detector defenses expose `detect(...)`, `detect_evasion(...)`, and
  `detect_poison(...)`.
- Noun-mode detector method names are removed from the public API.

External ART references:

- [ART preprocessor defenses](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/preprocessor.html)
- [ART postprocessor defenses](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/postprocessor.html)
- [ART trainer defenses](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/trainer.html)

## Minimal YAML Example

```yaml
model:
  defense:
    _target_: deckard.model.defense.base.DefensePipelineConfig
    defenses:
      - defense_name: art.defences.preprocessor.FeatureSqueezing
        defense_params:
          bit_depth: 4
```

## API Reference

```{eval-rst}
.. automodule:: deckard.model.defense.base
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.model.defense.preprocessor
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.model.defense.postprocessor
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.model.defense.transformer
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.model.defense.detector
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.model.defense.regularizer
   :members:
   :show-inheritance:
```

## See also

- {doc}`model`
- {doc}`attack`
- {doc}`score`
- {doc}`train`
- {doc}`../developers/defenses`
