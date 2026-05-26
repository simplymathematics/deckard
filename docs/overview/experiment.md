# Experiment Workflow Overview

{class}`deckard.experiment.ExperimentConfig` is the top-level orchestration entrypoint in deckard.
It composes component configs into one end-to-end runtime workflow and persists
results through canonical file aliases.

Primary API reference:

- {doc}`../api/experiment`

## Overview Flow

This page sits in the same core overview path as:

1. {doc}`summary`
2. {doc}`core`
3. {doc}`experiment`
4. {doc}`scoring`

## Workflow Stages

At a high level, experiment execution follows this sequence:

1. load data
2. sample or split data
3. train or load model
4. apply fit-time defense when configured
5. apply predict-time defense when configured
6. run attack and detector branches when configured
7. score and aggregate outputs
8. persist runtime artifacts

`apply_fit_defense` and `apply_predict_defense` are distinct stages.
Only fit-time defense can trigger retraining behavior.

## Component Configs

{class}`deckard.experiment.ExperimentConfig` composes these core module configs:

- Data: {doc}`../api/data`
- Model: {doc}`../api/model`
- Attack: {doc}`../api/attack`
- Detector: {doc}`../api/detector`
- Score: {doc}`../api/score`
- Artifacts and paths: {doc}`../api/file`
- Plot and layers (post-hoc): {doc}`../api/plot`, {doc}`../api/layers`

## Runtime Controls

Typical runtime controls are set through Hydra and OmegaConf overrides:

- stage selection and stage fan-out
- single run vs multirun
- optimizer directions and objectives
- output identity and persistence paths

See also:

- {doc}`optimize`
- {doc}`hydra`
- {doc}`dvc`
- {doc}`../developers/experiment`
