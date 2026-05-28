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
- mode: split scope (auto, train, test, val) distinct from stage/hook lifecycle

## Attack Families

Deckard runtime attack dispatch is centered on
{class}`deckard.attack.base.AttackConfig`, which resolves a family/subtype and
dispatches to mixin handlers via plugins.

Attack execution ordering is explicit in runtime metadata and defaults to
`post-defense`, so downstream scoring and detector layers can reason about
defense/attack sequencing consistently.

Supported attack families:

- [`evasion`](#evasion-attacks)
- [`poisoning`](#poisoning-attacks)
- [`extraction`](#extraction-attacks)
- [`inference`](#inference-attacks)

Supported inference subtypes:

- [`membership_inference`](#membership-inference-attacks)
- [`attribute_inference`](#attribute-inference-attacks)
- [`model_inversion`](#model-inversion-attacks)
- [`reconstruction`](#reconstruction-attacks)

(evasion-attacks)=
### Evasion attacks

Evasion attacks measure post-attack model robustness by perturbing evaluation
inputs while preserving their semantic class intent. In deckard outputs, these
typically map to evasion score contexts such as `blackbox_evasion` and
`whitebox_evasion`.

Common implementations include gradient- and decision-based methods such as
`FastGradientMethod`, `ProjectedGradientDescent`, `HopSkipJump`, and
`BoundaryAttack`.

(poisoning-attacks)=
### Poisoning attacks

Poisoning attacks measure training-time robustness by injecting crafted samples
or labels into the training process and evaluating downstream degradation (for
example, accuracy drop or calibration drift) after retraining.

In notebook and config workflows this commonly includes `PoisoningAttackSVM`.

(extraction-attacks)=
### Extraction attacks

Extraction attacks measure model theft risk by querying a target model and
training a surrogate to replicate the target's behavior. Evaluation focuses on
fidelity between target and surrogate predictions, plus downstream task metrics
on held-out data.

In notebook and config workflows this commonly includes `CopycatCNN`.

(inference-attacks)=
### Inference attacks

Inference attacks measure privacy leakage from model behavior or outputs rather
than only predictive robustness.

(membership-inference-attacks)=
#### Membership inference attacks

Membership inference measures whether an attacker can determine if a sample was
part of the target model's training set. Deckard emits membership-scoped
metrics under the `membership_inference_*` namespace.

Common implementation: `MembershipInferenceBlackBox`.

(attribute-inference-attacks)=
#### Attribute inference attacks

Attribute inference measures whether sensitive or hidden attributes can be
recovered from model predictions and observed features. Deckard emits these as
`inferred_<attribute>_*` metrics.

Common implementation: `AttributeInferenceBlackBox`.

(model-inversion-attacks)=
#### Model inversion attacks

Model inversion measures how much feature-level information can be reconstructed
about target classes or individuals from model outputs or gradients.

Common implementation: `MIFace`.

(reconstruction-attacks)=
#### Reconstruction attacks

Reconstruction attacks measure the ability to recover representative training
inputs or latent data structure from model behavior. These are tracked as
reconstruction-oriented attack outputs in deckard workflows.

Scoring keys include:

- `blackbox_membership_inference`
- `blackbox_evasion`
- `whitebox_evasion`
- `blackbox_<attribute>_inference`
- `whitebox_<attribute>_inference`

## Common Attack and Defense Components

Attack configuration in this module is typically composed with
{doc}`model` defense pipelines and {doc}`score` attack scorer profiles.

Common ART evasion attack classes used with deckard attack configs:

- [`FastGradientMethod`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#fast-gradient-method-fgm)
- [`HopSkipJump`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#hopskipjump)
- [`BoundaryAttack`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#boundary-attack-decision-based-attack)
- [`ProjectedGradientDescent`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#projected-gradient-descent-pgd)
- [`BasicIterativeMethod`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#basic-iterative-method-bim)

Common ART poisoning attack classes used with deckard attack configs:

- [`PoisoningAttackSVM`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/poisoning.html#poisoning-attack-svm)

Common ART extraction attack classes used with deckard attack configs:

- [`CopycatCNN`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/extraction.html#copycat-cnn)

Common ART inference attack classes used with deckard attack configs:

- [`MembershipInferenceBlackBox`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/membership_inference.html#membership-inference-black-box)
- [`AttributeInferenceBlackBox`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/attribute_inference.html#attribute-inference-black-box)
- [`MIFace`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/model_inversion.html#miface-model-inversion-attack)

Common ART defenses that are paired with attack runs in deckard:

- [`FeatureSqueezing`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/preprocessor.html#feature-squeezing)
- [`SpatialSmoothing`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/preprocessor.html#spatial-smoothing)
- [`AdversarialTrainer`](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/trainer.html#adversarial-training)

Related Deckard docs:

- {doc}`model` for configuring defense pipelines
- {doc}`defend` for defense pipeline and defense mixin dispatch
- {doc}`score` for attack-specific scorer profiles
- {doc}`pytorch` for ART estimator integration in torch workflows
- {doc}`fairlearn`, {doc}`lifelines`, and {doc}`anjana` for plugin-specific attack/scoring integrations

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
