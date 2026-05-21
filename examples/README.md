# `examples/sklearn/` — End-to-end deckard experiment (Scikit-learn pipeline)

This example demonstrates a full deckard workflow using a
scikit-learn-compatible experiment setup, including training, adversarial
evaluation, fairness analysis, and visualization generation.
It is a fully reproducible experiment workspace rather than a minimal tutorial.

______________________________________________________________________

## Directory layout

```text
examples/*/
├── pycache
├── attack/
├── config/
├── data/
├── deckard.log
├── dvc.lock
├── dvc.yaml
├── error.log
├── model/
├── optuna.db
├── outputs/
└── plots/
└── .deckard_rc
```

## Configuration (`config/`)

The `config/` directory is a Hydra-based configuration tree defining all aspects
of the experiment.
Run `source .deckard_rc` in the target directory to set the environment
variables that specify the default configuration folder and file.

### Core files

- `default.yaml` — main experiment entry point
- `meta.yaml` — global experiment metadata
- `aft.yaml` — survival model configuration
- `survival.yaml` — survival analysis settings

### Submodules

- `model/` — model definitions and training parameters
- `data/` — dataset loading and preprocessing
- `attack/` — adversarial attack configurations
- `defense/` — robustness strategies
- `fairness-default.yaml` — fairness evaluation settings
- `inference-default.yaml` — membership inference configuration
- `attribute-inference-default.yaml` — attribute inference attacks
- `score/` — evaluation metrics
- `search/` — hyperparameter search spaces (Optuna/Hydra)
- `plot/` — visualization configuration
- `sample/` — sampling strategies
- `files/` — file/path overrides

______________________________________________________________________

## Artifacts (`data/`, `model/`, `attack/`)

```text
data/
└── c0be3853fdb3dc49358a957cbed21c68.pkl
model/
└── cf1b02f6045425909ffb86402e68e0a9.pkl
attack/
└── 0c7864e2681dfea328635131b2de0920.pkl
```

## Outputs (`outputs/`)

- model outputs
- evaluation results
- intermediate artifacts
- run-specific configuration snapshots

## Plots (`plots/`)

Generated analysis and visualization outputs.

### General diagnostics

- `jointplot.png`
- `pca.png`
- `radviz.png`
- `pcoords.png`
- `rank1d.png`
- `rank2d.png`

### Pareto analysis (multi-objective tradeoffs)

- `pareto_accuracy.png`
- `pareto_accuracy_vs_evasion.png`
- `pareto_accuracy_vs_membership_inference.png`
- `pareto_membership_inference_vs_evasion_colored_by_accuracy.png`

These visualize tradeoffs between:

- predictive performance
- evasion robustness
- privacy leakage (membership inference)

______________________________________________________________________

### Partial effect / sensitivity analysis

- `partial_effects_benign_accuracy.png`
- `partial_effects_evasion.png`
- `partial_effects_evasion_attacks.png`
- `partial_effects_evasion_numeric.png`
- `partial_effects_membership_inference.png`
- `partial_effects_membership_inference_attacks.png`
- `partial_effects_membership_numeric.png`

Used to analyze parameter sensitivity across:

- accuracy
- adversarial robustness
- privacy leakage

______________________________________________________________________

### Survival analysis plots

Survival modeling outputs (hazard curves, survival functions, time-to-event analysis).

______________________________________________________________________

## Logs and tracking

- `deckard.log` — main execution log
- `error.log` — error traces and failures
- `optuna.db` — Optuna hyperparameter optimization database

______________________________________________________________________

## Pipeline definitions (DVC)

- `dvc.yaml` — pipeline definition (stages, dependencies)
- `dvc.lock` — locked reproducible pipeline state

Supports:

- dataset versioning
- reproducible training pipelines
- cached computation graphs

______________________________________________________________________

## Summary

This example is a full experimental pipeline integrating:

- Hydra configuration management
- scikit-learn model training
- adversarial robustness evaluation
- fairness and privacy analysis
- Optuna hyperparameter optimization
- DVC reproducibility
- structured logging and artifact tracking
- extensive visualization suite

It represents a complete deckard research workflow from configuration ->
execution -> evaluation -> analysis.

## See also

[Installation help](../README.md)
[Full Documentation](../docs/overview/build_docs.md)
[Developer Documentation](../develop.md)
[Jupyter Notebook Examples](../notebooks/README.md)
