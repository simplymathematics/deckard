# Experiment Workflow Overview

{class}`deckard.experiment.ExperimentConfig` is the top-level orchestration entrypoint in deckard.
It composes component configs into one end-to-end runtime workflow and persists
results through canonical file aliases.

Primary API reference:

- {doc}`/api/experiment/index`

## Overview Flow

This page sits in the same core overview path as:

1. {doc}`index`
2. {doc}`core`
3. {doc}`experiment`
4. {doc}`scoring`

## Workflow Stages

At a high level, experiment execution follows this sequence:

1. load data
2. sample or split data
3. apply fit-time defense(s) when configured (pre-processing, training, re-training, multi-objective optimization)
4. train or load model
5. apply predict-time defense(s) when configured (post-processing)
6. run attack  when configured (evasion, poisoning, inference, extraction)
7. run the detection branches (poisoning and evasion attack filtering)
8. score and aggregate outputs (scores, initialization parameters, metadata)
9. persist runtime artifacts (data, models, predictions, attacked samples)

`apply_fit_defense` and `apply_predict_defense` are distinct stages.
Only fit-time defense can trigger retraining behavior.

## Component Configs

{class}`deckard.experiment.ExperimentConfig` composes these core module configs:

- Data: {doc}`/api/data/index`
- Model: {doc}`/api/model/index`
- Attack: {doc}`/api/attack/index`
- Detector: {doc}`/api/detector/index`
- Score: {doc}`/api/score/index`
- Artifacts and paths: {doc}`/api/file/index`

In addition, there are tools for plotting and post-hoc analysis across many experiments.
- Plot and visualization {doc}`/api/plot/index`
- Survival analysis and pareto analysis {doc}`/api/layers/index`

## Runtime Controls

Typical runtime controls are set through Hydra and OmegaConf overrides:

- data, model, attack, detector, file, and scoring components
- single run vs multirun
- optimizer directions and objectives
- output identity and persistence paths


See also:

- {doc}`optimize`
- {doc}`hydra`
- {doc}`dvc`
