# Experiment

## Basic flow state

`load -> sample -> train -> defense -> attack -> score -> persist`.

## Purpose

Define user-facing experiment runtime owner behavior, including stage ordering,
mode propagation to child components, hook orchestration boundaries, and
persistence outputs across framework adapters and plugin integrations.

## Capabilities

- Orchestrate end-to-end execution across core runtime modules.
- Route split/mode/stage context into component runtimes.
- Compose reproducible runs via declarative config objects and overrides.
- Persist aggregated run artifacts, timing, and score outputs.
- Coordinate child components across {doc}`data`, {doc}`model`, {doc}`attack`, {doc}`detector`, and {doc}`score`.

## Outputs

- Composed runtime outputs from data/model/attack/detector components.
- Aggregated score payloads and persisted score files.
- Canonical experiment timing and stage execution metadata.
- Persisted artifact paths managed through file configuration objects.

Implementation-level runtime contracts are documented in
{doc}`../developers/experiment`.

## Introduction

This page is the canonical home for experiment orchestration behavior and API
details. It documents end-to-end workflow composition across data, model,
attack, detector, scoring, and persistence layers.

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

Integration-specific orchestration behavior is documented in integration pages
to keep this page focused on core experiment flow.

## Integrations

- Framework integration: {doc}`pytorch`
- Plugin integrations: {doc}`lifelines`

## Overview

The experiment layer coordinates the full deckard workflow by composing:

- data loading and sampling via {mod}`deckard.data`
- model training/evaluation via {mod}`deckard.model`
- optional attack execution via {mod}`deckard.attack`
- optional detector execution via {mod}`deckard.detector`
- score aggregation and file outputs via {mod}`deckard.file`

It is the primary integration point for reproducible end-to-end runs.

Canonical public execution entrypoint:

- {meth}`deckard.experiment.ExperimentConfig.run`

{meth}`deckard.experiment.ExperimentConfig.__call__` remains available as a backward-compatible alias.

Experiment configs are typically composed with [Hydra](https://hydra.cc) and
[OmegaConf](https://omegaconf.readthedocs.io), including config-group overrides
for data, model, attack, detector, score, and file targets.

For config-group organization details, see
{doc}`/developers/declarations`.

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
      name: make_classification
   model:
      _target_: deckard.model.base.ModelConfig
      name: sklearn.linear_model.LogisticRegression
      classifier: true
   attack:
      _target_: deckard.attack.base.AttackConfig
      name: art.attacks.evasion.FastGradientMethod
```

## Implementation Notes

Detailed experiment internals (hook contracts, cache schema, and runtime
serialization policy) are documented in {doc}`../developers/experiment`.

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
