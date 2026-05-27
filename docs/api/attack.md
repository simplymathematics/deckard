# Attack

## Basic flow state

`resolve mode/stage -> generate -> predict -> score -> persist`.

## Purpose

Define user-facing attack runtime owner behavior, including stage versus mode
semantics, scoring outputs, persistence paths, and boundaries for framework
adapters and plugin integrations.

## Capabilities

- Normalize attack mode and stage tokens for consistent execution.
- Generate attacked samples across supported attack families.
- Produce attack prediction payloads for downstream evaluation.
- Emit attack-scoped metrics and persist canonical attack artifacts.
- Consume model outputs from {doc}`model` and emit metrics used by {doc}`score`.

Implementation-level runtime contracts are documented in {doc}`../developers/attack`.

## Outputs

- Attack artifacts (`attack_file`, `attack_predictions_file`, `score_file`).
- Attacked labels/predictions and score-ready payloads.
- Attack timing fields (`attack_generation_time`, `attack_prediction_time`, `attack_score_time`).
- Attack score dictionaries merged with runtime timing metadata.

## Introduction

This page is the canonical home for attack module behavior and API details.
It documents attack-family dispatch, mode/stage semantics, persistence
contracts, and framework extensions.

## Overview

The attack module orchestrates adversarial example generation across supported
backends and attack families.

It provides:

- attack configuration and instantiation
- attack execution over model/data outputs
- artifact persistence for attacked samples and labels
- attack-aware scoring hooks used by experiment orchestration

Canonical runtime contract:

- files: attack artifacts persist through files-only paths (`attack_file`, `attack_predictions_file`, `score_file`)
- times: `attack_generation_time`, `attack_prediction_time`, `attack_score_time`
- scores: attack metrics merged with runtime timing metadata
- stage: canonical stage tokens (`pre-attack`, `post-attack`) with compatibility aliases normalized at runtime
- mode: split scope (`auto`, `train`, `test`, `val`) distinct from stage/hook lifecycle

## Attack Families

Deckard runtime attack dispatch is centered on
{class}`deckard.attack.base.AttackConfig`, which resolves a family/subtype and
dispatches to mixin handlers via plugins.

Attack execution ordering is explicit in runtime metadata and defaults to
`post-defense`, so downstream scoring and detector layers can reason about
defense/attack sequencing consistently.

Supported attack families:

- `evasion`
- `poisoning`
- `extraction`
- `inference`

Supported inference subtypes:

- `membership_inference`
- `attribute_inference`
- `model_inversion`
- `reconstruction`

Scoring keys include:

- `blackbox_membership_inference`
- `blackbox_evasion`
- `whitebox_evasion`
- `blackbox_attribute_inference`
- `whitebox_attribute_inference`

## Common Attack and Defense Components

Attack configuration in this module is typically composed with
{doc}`model` defense pipelines and {doc}`score` attack scorer profiles.

Common ART evasion attack classes used with deckard attack configs:

- [`FastGradientMethod`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#fast-gradient-method-fgm)
- [`ProjectedGradientDescent`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#projected-gradient-descent-pgd)
- [`BasicIterativeMethod`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#basic-iterative-method-bim)

Common ART defenses that are paired with attack runs in deckard:

- [`FeatureSqueezing`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/preprocessor.html#feature-squeezing)
- [`SpatialSmoothing`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/preprocessor.html#spatial-smoothing)
- [`AdversarialTrainer`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/trainer.html#adversarial-training)

Related Deckard docs:

- {doc}`model` for configuring defense pipelines
- {doc}`defend` for defense pipeline and defense mixin dispatch
- {doc}`score` for attack-specific scorer profiles
- {doc}`pytorch` for ART estimator integration in torch workflows
- {doc}`/overview/extensions/index` for cross-framework extensions map

## Integrations

- Framework integration: {doc}`pytorch`
- Plugin integrations: {doc}`fairlearn`, {doc}`lifelines`, {doc}`anjana`

## Examples

```{seealso}

   Notebook-based attack workflows are documented in:

   - {doc}`notebooks/art_attacks.ipynb </notebooks/art_attacks>`
   - {doc}`notebooks/fairlearn.ipynb </notebooks/fairlearn>`
   - {doc}`notebooks/pytorch.ipynb </notebooks/pytorch>`

```

## Minimal YAML Example

```yaml
attack:
   _target_: deckard.attack.base.AttackConfig
   attack_type: art.attacks.evasion.FastGradientMethod
   attack_params:
      eps: 0.1
   attack_size: 100
```

## API Reference

```{eval-rst}
.. automodule:: deckard.attack.base
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.attack.evasion
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.attack.poisoning
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.attack.extraction
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.attack.inference
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: deckard.attack.reconstruction
   :members:
   :show-inheritance:
```

Framework-specific attack adapters are documented in {doc}`pytorch`.

## Troubleshooting

- Ensure the selected attack backend matches the active model backend.
- Confirm attack parameters are valid for the chosen ART/Fairlearn attack type.
- Verify the attack receives compatible input shapes and labels.

### See also

- {doc}`experiment` — experiment orchestration
- {doc}`model` — model configuration and execution
- {doc}`defend` — defense pipeline configuration and mixin behavior
- {doc}`data` — data loading and split handling
- {doc}`sample` — sampling/split strategy definitions
- {doc}`score` — attack scoring profiles
