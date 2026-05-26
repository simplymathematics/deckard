# Core Modules

This page is a compact guide to the core runtime surfaces that power the
experiment workflow. Use it after {doc}`summary` and before {doc}`experiment`
to understand which configuration objects participate in an end-to-end run.

Detailed core-module documentation lives in {doc}`../api/modules`.

## Overview Flow

The core overview path is:

1. {doc}`summary`
2. {doc}`core`
3. {doc}`experiment`
4. {doc}`scoring`

## Core API Map

- [Data](../api/data): dataset loading, sampling, pipelines, and data scoring.
- [Pipeline](../api/pipeline): preprocessing pipeline runtime and stage execution.
- [Model](../api/model): model orchestration, scoring, persistence, and defense stages.
- [Training](../api/train): training mixins and trainer-defense integration.
- [Defense](../api/defend): defense pipeline runtime and defense-family dispatch.
- [Attack](../api/attack): attack family execution and attack scoring integration.
- [Detector](../api/detector): detector runtime, filtering, and detector metrics.
- [Score](../api/score): scorer configs, mode normalization, and metric composition.
- [File](../api/file): canonical artifact path management and persistence helpers.
- [Experiment](../api/experiment): end-to-end experiment orchestration runtime.
- [Plot](../api/plot): backend routing and plotting runtime configs.
- [Utils](../api/utils): shared runtime utilities and config helpers.

## How This Connects To Experiments

`ExperimentConfig` composes the core runtime surfaces listed above. In normal
use, most runs combine:

1. data
2. model
3. optional attack and detector branches
4. scoring
5. file-backed persistence

See {doc}`experiment` for the orchestration flow and {doc}`scoring` for score
outputs and optimization targets.

## Related

- {doc}`index`
- {doc}`summary`
- {doc}`experiment`
- {doc}`scoring`
- {doc}`../api/modules`
