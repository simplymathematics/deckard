# DVC Overview

This page summarizes Deckard's DVC pipeline autogeneration contract.

Related docs:

- [Developer DVC Spec](../developers/dvc.md)
- [Optimization Contract](../developers/optimization.md)
- [Hydra Contract](../developers/hydra.md)
- [Pruning Contract](../developers/pruning.md)

## What It Does

Deckard can generate a deterministic `dvc.yaml` from experiment runtime metadata.

Core capabilities:

- canonical stage naming (`<component>__<stage>`)
- deterministic deps/outs/params wiring
- single-run and multirun command emission
- optional runtime cache alias wiring
- metrics policy aligned with optimizer behavior
- Vega-Lite (`*.vl.json`) plot artifact targets
- canonical plot naming patterns (for example `<attack_alias>_<attack_param>_vs_<metric>.vl.json`)

## Current Implementation Notes

- DVC integration is optional and wrapper-driven via `DVCExperimentConfig`.
- Base `ExperimentConfig` hashing excludes `dvc_plugin`, so DVC policy toggles do not change experiment identity.
- Generated structured params payloads use top-level `__target__` and a single top-level `dvc_plugin` block.
- Wrapped `experiment` payloads do not duplicate `dvc_plugin`.
- Persisted DVC/DVCLive path fields are normalized to relative paths to avoid absolute host-path leakage.

## Canonical Stages

Generated stage names follow experiment canon:

- `data__load`
- `data__sample`
- `model__train`
- `detector__defense`
- `attack__attack`
- `experiment__score`
- `experiment__persist`

## Identity-Derived Output Paths

DVC and DVCLive output paths must be identity-derived:

- run mode: `experiment_name`
- multirun mode: stage-dependent experiment hash

Reference layout:

- `outputs/logs/<run_identity>/...`

## Public API

```python
from deckard.experiment import generate_dvc_pipeline

generate_dvc_pipeline(
    experiment=experiment_cfg,
    output_file="dvc.yaml",
    stage_selection=None,
    include_cache_aliases=True,
    mode="single",
    multirun_count=None,
    overwrite=False,
)
```

Command-generation note:

- Stage commands no longer emit explicit `--multirun` flags.
- Run mode is carried through canonical stage/mode metadata and params payloads.

## Quick Checklist

- Are generated stage names canonical?
- Are output paths identity-derived for run and multirun?
- Are metrics file-only by default?
- Are optimizer-keyed metrics selectors only enabled when optimizers are configured?
- Do plots point to Vega-Lite specs (`*.vl.json`)?
