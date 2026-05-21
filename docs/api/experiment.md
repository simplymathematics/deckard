# Experiment

The {mod}`deckard.experiment` module contains the high-level orchestration
entrypoints for end-to-end experiment execution.

```{eval-rst}
.. automodule:: deckard.experiment
   :members:
   :show-inheritance:
```

## Torch Framework

PyTorch-specific experiment orchestration is available via
{class}`deckard.frameworks.pytorch.experiment.TorchExperimentConfig` in the
optional {mod}`deckard.frameworks.pytorch.experiment` module.
See also: {doc}`pytorch`.

Use this extension when you need PyTorch model/data orchestration while keeping
the same high-level experiment lifecycle as {class}`deckard.experiment.ExperimentConfig`.

## Survival Plugin

Survival-specific experiment orchestration is split into a dedicated optional
module.
See also: {doc}`lifelines`.

```{eval-rst}
.. automodule:: deckard.plugins.lifelines.experiment
   :members:
   :show-inheritance:
```

## Overview

The experiment layer coordinates the full deckard workflow by composing:

- data loading and sampling via {mod}`deckard.data`
- model training/evaluation via {mod}`deckard.model`
- optional attack execution via {mod}`deckard.attack`
- optional detector execution via {mod}`deckard.detector`
- score aggregation and file outputs via {mod}`deckard.file`

It is the primary integration point for reproducible end-to-end runs.

Experiment configs are typically composed with [Hydra](https://hydra.cc) and
[OmegaConf](https://omegaconf.readthedocs.io), including config-group overrides
for data, model, attack, detector, score, and file targets.

For config-group organization details, see
{doc}`/developers/config_declaration_architecture`.

Available experiment entrypoints:

- {class}`~deckard.experiment.ExperimentConfig` (default)
- {class}`~deckard.frameworks.pytorch.experiment.TorchExperimentConfig` (PyTorch)
- {class}`~deckard.plugins.lifelines.experiment.SurvivalExperimentConfig` (survival)

## Examples

```{seealso}

   Notebook-based experiment workflows (single-attack, multi-attack,
   detector phase, and backend-specific runs) are documented in:

   - {doc}`notebooks/sklearn.ipynb </notebooks/sklearn>`
   - {doc}`notebooks/art_attacks.ipynb </notebooks/art_attacks>`
   - {doc}`notebooks/pytorch.ipynb </notebooks/pytorch>`

```

## Minimal YAML Example

```yaml
experiment:
   _target_: deckard.experiment.base.ExperimentConfig
   data:
      _target_: deckard.data.base.DataConfig
      dataset_name: make_classification
   model:
      _target_: deckard.model.base.ModelConfig
      model_type: sklearn.linear_model.LogisticRegression
      classifier: true
   attack:
      _target_: deckard.attack.base.AttackConfig
      attack_type: art.attacks.evasion.FastGradientMethod
```

## Internals

The module resolves nested config objects, applies runtime overrides, and
normalizes outputs for downstream scoring/serialization.

Hydra override patterns commonly used with experiments include:

- selecting alternate attack/score profiles per run
- composing plugin configs (for example fairlearn or lifelines)
- switching runtime backends (sklearn, pytorch, survival)

## Troubleshooting

- Verify config paths and override keys when Hydra/OmegaConf resolution fails.
- Ensure optional dependencies are installed for selected model/attack backends.
- Check file output paths in {class}`deckard.file.FileConfig` if artifacts are missing.

### See also

- {doc}`data` — data configuration and loading
- {doc}`model` — model configuration and training
- {doc}`attack` — attack configuration
- {doc}`file` — result serialization
- {doc}`score` — scoring framework
- {doc}`plot` — backend plotting configuration and outputs
- {doc}`layers` — CLI orchestration layers (including pareto and survival)
- {doc}`pytorch` — PyTorch experiment orchestration
- {doc}`lifelines` — survival experiment orchestration
- {doc}`utils` — utility functions
