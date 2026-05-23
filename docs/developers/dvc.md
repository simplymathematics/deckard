# DVC Pipeline Autogeneration Spec

This document defines the permanent developer contract for DVC pipeline autogeneration.

It records current experiment-runtime behavior from:

- native config + HookPlugin + HookBundle composition
- stage caching and reuse across runs/trials

It then defines a concrete design for generating reproducible `dvc.yaml` pipelines from runtime metadata.

Related contracts:

- [Optimization Runtime Contract](optimization.md)
- [Hydra and Optuna Orchestration Contract](hydra.md)
- [Pruning Runtime Contract](pruning.md)

## Goals

- Generate deterministic `dvc.yaml` stages from canonical experiment stage contracts.
- Reuse canonical runtime metadata (`params`, `outputs`, `files`, hook graph, cache keys).
- Support both single-run and multi-trial execution commands.
- Keep DVC generation as an additive utility, not a replacement runtime.
- Make cache-aware stage skipping explicit and inspectable in DVC outputs.
- Prefer hook-driven DVCLive logging for per-trial runtime reporting.
- Keep layer-level execution paths for study-level and batch aggregate plotting.

## Frozen Contract (Current)

The current implementation is frozen around these invariants:

- `DVCExperimentConfig` is an optional wrapper and not a replacement for core runtime execution.
- Base `ExperimentConfig` hashing excludes DVC policy fields (`dvc_plugin`) so DVC toggles do not change experiment identity.
- DVC hook wrappers are only composed when the DVC plugin is explicitly enabled.
- Structured DVC params payloads use top-level `__target__` with a single top-level `dvc_plugin` block.
- Wrapped `experiment` payloads do not duplicate `dvc_plugin`.
- Persisted DVC/DVCLive path fields are normalized to repository-relative paths.

### Current command/mode behavior

- Generated stage commands do not emit explicit `--multirun` flags.
- Mode (`single`/multirun-equivalent semantics) is represented through stage planning and params payload metadata.

### Runtime manifest interpretation notes

- `params.attack` represents the primary attack fingerprint for the experiment.
- `params.attack_chain` represents the full ordered attack chain.
- For single-attack runs, these entries can contain the same attack fingerprint by design.

## Existing Runtime Behavior: Composition and Hooks

### Native component composition

`ExperimentConfig.compose_components(...)` already supports runtime overrides for:

- `data`, `model`, `attack`, `detector`, `score`, `defense`, `files`
- `hook_plugins`, `hook_bundles`
- `evaluation_mode`, `score_mode`, `cache_enabled`

After applying overrides, runtime contracts are recomposed and a fresh `params` manifest is built.

### Canonical hook graph and bundle composition

Current orchestration behavior already provides:

- programmatic hook graph generation via `build_experiment_hook_graph()`
- canonical hook bundle via `build_experiment_hook_bundle()`
- additive user bundle + hook plugin composition through `compose_hook_plugins(...)`
- deterministic execution order: canonical bundle, then user bundles, then explicit plugins

Runtime hook metadata is captured under `outputs["hooks"]`:

- `graph`: canonical stage nodes with before/after hook names
- `trace`: execution-time events (`component`, `stage`, `event`, `run`)

This gives enough structure to derive stage boundaries for DVC without hard-coded stage lists.

## Existing Runtime Behavior: Caching and Reuse

### Canonical cache stages and keys

Experiment runtime already caches stage payloads for:

- `sample`, `train`, `defense`, `attack`, `score`

Cache keys are deterministic and built from:

- normalized stage + component
- stage identity (for example run/fold index)
- params manifest fingerprints (`build_experiment_params_manifest`)

### Cache persistence and visibility

Cache behavior currently includes:

- runtime cache storage next to params file as `*.runtime_cache.pkl`
- cache hit/write tracking in `outputs["cache"]["hits"]` and `outputs["cache"]["writes"]`
- cache metadata in runtime state YAML (`enabled`, path, hit/write counts)

### Rehydratable stage payloads

Current cached payloads already include enough data to avoid recomputation:

- sample payloads: splits, sample times, sample-related data fields
- train payloads: scores, predictions/probabilities
- attack payloads: attack predictions + scores
- defense/detector payloads: detector score payloads
- score payloads: experiment scorer outputs

This behavior is a direct input to DVC stage outs and cache reuse semantics.

## Scope

### In scope

- A utility that writes `dvc.yaml` from an `ExperimentConfig` instance and/or persisted runtime state.
- Canonical stage-to-DVC-stage mapping using experiment canon helpers.
- Deterministic deps/outs/params mapping from `FileConfig`, runtime params, and optional cache metadata.
- Command emission for:
  - single experiment run
  - multi-trial/fan-out execution
- Optional cache-reuse mode that points DVC outs/deps at canonical cache aliases.

### Out of scope

- Replacing ExperimentConfig execution with DVC-native Python code.
- DVC remote configuration automation.
- Re-implementing runtime cache internals in DVC logic.

## Proposed API

### Utility entrypoint

```python
from deckard.experiment.dvc import generate_dvc_pipeline

generate_dvc_pipeline(
    experiment=exp,
    output_file="dvc.yaml",
    params_file="params.yaml",
    stage_selection=None,           # None -> canonical default stages
    include_cache_aliases=True,
    mode="single",                # "single" | "multirun"
    multirun_count=None,
    overwrite=False,
)
```

### Runtime DVCLive integration API (proposed)

```python
from deckard.experiment.dvc import configure_dvclive_runtime

configure_dvclive_runtime(
  experiment=exp,
  enabled=True,
  dir="outputs/dvclive",
  monitor_system=True,
  make_dvcyaml=False,
  make_report=False,
  make_summary=False,
)
```

Behavioral contract:

- Primary path: hook-driven DVCLive integration attached to experiment runtime hooks.
- Secondary path: layer-level aggregate plotting/report stages for study outputs.
- Use DVCLive APIs directly (`log_*`, `next_step`, `end`, `make_dvcyaml`, `make_report`, `make_summary`, `monitor_system`).

### Supporting helpers

- `build_dvc_stage_plan(experiment, stage_selection=None, include_cache_aliases=True)`
  - returns normalized stage plan with deps/outs/params/cmd blocks
- `build_dvc_stage_name(component, stage)`
  - canonical naming: `<component>__<stage>`
- `extract_dvc_file_aliases(file_dict, cache_path=None)`
  - normalizes runtime file aliases for DVC deps/outs
- `build_dvc_cmd(experiment, stage_plan, mode, multirun_count=None)`
  - emits reproducible CLI command strings

## Canonical Stage Mapping

Map canonical experiment stages to DVC stages as follows.

- `load`
  - stage name: `data__load`
  - deps: source config and optional raw data files
  - outs: loaded/persisted data artifacts when configured
- `sample`
  - stage name: `data__sample`
  - deps: load outputs + sampler params
  - outs: split/sampled data artifacts, sample cache payload
- `train`
  - stage name: `model__train`
  - deps: sample outputs + model/defense params
  - outs: model artifact, predictions, train cache payload
- `defense`
  - stage name: `detector__defense` (or `model__defense` when model defense stages are selected)
  - deps: train outputs + detector/defense params
  - outs: defense/detector outputs + cache payload
- `attack`
  - stage name: `attack__attack`
  - deps: model outputs + attack params
  - outs: attack artifacts/predictions + cache payload
- `score`
  - stage name: `experiment__score`
  - deps: upstream outputs + scorer params
  - outs: score artifacts + score cache payload
- `persist`
  - stage name: `experiment__persist`
  - deps: all selected stage outputs
  - outs: score file, params/runtime YAML, runtime cache pointer artifact

Notes:

- Multi-attack runs should emit one stage per attack alias when aliases are configured.
- Stage names must be stable across runs for deterministic DVC lock files.

## Deps / Outs / Params Contract

### deps

Always include:

- resolved experiment config snapshot (or source config path)
- code entrypoint module path(s) used by the generated command
- upstream stage artifact outputs required for stage execution

Conditionally include:

- cache file path when `include_cache_aliases=True`
- hook bundle/plugin declaration files when present in config

### outs

Include configured file aliases from `FileConfig` that are written by the stage.

When cache aliases are enabled, include synthetic outs/deps for cache payload continuity:

- runtime cache file (`*.runtime_cache.pkl`)
- params/runtime YAML state file

### params

Write a DVC params file (YAML) containing:

- stage selection
- run mode (`single`/`multirun`)
- key experiment manifest fields (`experiment_name`, `library`, `random_state`, score/eval mode)
- component fingerprints from `build_experiment_params_manifest`

Current payload shape includes:

- top-level `__target__` for wrapper identity
- top-level `experiment` payload for constructor-safe runtime state
- top-level `dvc_plugin` policy payload
- top-level `_dvc` metadata (`stage_selection`, `run_mode`, `params_manifest`)

This ensures DVC stage invalidation aligns with experiment cache invalidation.

Params MUST be parsed and cached according to stage (e.g. pre-defense does not include defense params)

## Metrics and Plot Policy

### Metrics policy

- Default: file-only metrics entries (for example `scores.json`, `timing.json`, `metadata.json`).
- Add keyed metric selectors only when `optimizers` is explicitly configured.

### Metrics payload expansion

In addition to score values, include timing/count/metadata fields not already emitted
by `ScoreDict` into DVC metrics artifacts.

Recommended payload fields include:

- stage timings and aggregate runtime fields
- sample/training/prediction counts
- cache hit/write metadata
- experiment/trial identity metadata

### Plot coverage target

Minimum targeted plot families:

- roc_auc
- covariance
- epochs vs loss
- feature importance
- metric vs attack strength
- metric vs defense strength
- adversarial vs benign metrics
- attack-vs-defense comparison heatmaps

Canonical Vega-Lite naming examples:

- `roc_auc.vl.json`
- `<attack_alias>_<attack_param>_vs_<metric>.vl.json`
- `<defense_alias>_<defense_param>_vs_<metric>.vl.json`
- `adversarial_vs_benign_<metric>.vl.json`
- `attack_vs_defense_<metric>_heatmap.vl.json`

Runnable Hydra YAML spec configs are stored under:

- `examples/sklearn/config/dvc/plot_specs/roc_auc.yaml`
- `examples/sklearn/config/dvc/plot_specs/hsj_max_iter_vs_accuracy.yaml`
- `examples/sklearn/config/dvc/plot_specs/class_labels_apply_fit_vs_accuracy.yaml`
- `examples/sklearn/config/dvc/plot_specs/adversarial_vs_benign_accuracy.yaml`
- `examples/sklearn/config/dvc/plot_specs/attack_vs_defense_accuracy_heatmap.yaml`
- `examples/sklearn/config/dvc/plot_specs/epochs_vs_loss.yaml`
- `examples/sklearn/config/dvc/plot_specs/feature_importance.yaml`
- `examples/sklearn/config/dvc/plot_specs/covariance.yaml`

Plot artifact contract:

- Plot outputs should be Vega-Lite specification files for browser rendering.
- Preferred extension: `.vl.json`.
- DVC `plots` entries should point to Vega-Lite spec files, not static image files.

### Plan: DVC + DVCLive integration

- Primary integration path: hook-driven DVCLive integration in experiment runtime.
- Secondary integration path: layer-driven study and batch aggregate plotting.
- Use DVCLive APIs directly rather than reimplementing wrapper behavior.

DVCLive API coverage target:

- `log_*`
- `next_step`
- `end`
- `monitor_system`
- `make_dvcyaml`
- `make_report`
- `make_summary`

### DVCLive output directory naming

Do not hard-code a shared directory like `dvclive_runtime/`.

Use an identity-derived directory key:

- Run mode: use `experiment_name`.
- Multirun mode: use a deterministic hash derived from stage-dependent
  experiment parameters (for example stage/component fingerprints from the
  params manifest).

Reference pattern:

- `outputs/logs/<run_identity>/...`

Where `<run_identity>` resolves to:

- `<experiment_name>` in run mode
- `<stage_dependent_experiment_hash>` in multirun mode

### DVC metrics policy

- Default to file-only metrics entries.
- Add keyed metrics selectors only when `optimizers` is explicitly configured.

### Metrics payload expansion

Include timing, count, and metadata fields not already emitted in the core score
payload.

### Plot coverage target

- roc_auc
- covariance
- epochs vs loss
- feature importance
- metric vs attack strength
- metric vs defense strength
- adversarial vs benign metrics
- attack-vs-defense comparison heatmaps

### Canonical stage name example

When representing this flow in `dvc.yaml`, use canonical stage naming:

- `experiment__persist`

Example stage shape:

```yaml
stages:
  <canon_stage_name>:
    cmd: >
      mkdir -p outputs/logs/<run_identity>/plots &&
      deckard optimize
      data=adult
      model=rf
      attack=boundary
      defense=gaussian-noise
      hydra.sweeper.study_name=dvclive_adult_rf_gaussian-noise_boundary
      +files.params_file=outputs/logs/<run_identity>/params.yaml
      +files.score_file=outputs/logs/<run_identity>/scores.json
      +files.log_file=outputs/logs/<run_identity>/run.log
      +files.error_file=outputs/logs/<run_identity>/error.log
    deps:
      - .deckard_rc
      - ../../deckard
      - ./config/default.yaml
    outs:
      - outputs/logs/<run_identity>/
    metrics:
      - outputs/logs/<run_identity>/scores.json
      - outputs/logs/<run_identity>/timing.json
      - outputs/logs/<run_identity>/counts.json
      - outputs/logs/<run_identity>/metadata.json
    plots:
      - outputs/logs/<run_identity>/plots/roc_auc.vl.json
      - outputs/logs/<run_identity>/plots/<attack_alias>_<attack_param>_vs_<metric>.vl.json
      - outputs/logs/<run_identity>/plots/<defense_alias>_<defense_param>_vs_<metric>.vl.json
      - outputs/logs/<run_identity>/plots/adversarial_vs_benign_<metric>.vl.json
      - outputs/logs/<run_identity>/plots/attack_vs_defense_<metric>_heatmap.vl.json
      - outputs/logs/<run_identity>/plots/epochs_vs_loss.vl.json
      - outputs/logs/<run_identity>/plots/feature_importance.vl.json
      - outputs/logs/<run_identity>/plots/covariance.vl.json
    params:
      - ./config/default.yaml:
        - defaults
        - hydra
        - optimizers
```

      Example identity resolution:

      - run mode: `<run_identity> = <experiment_name>`
      - multirun mode: `<run_identity> = <stage_dependent_experiment_hash>`

## Command Templates

### Single run
Parses from 
```bash
python -m deckard optimize <optional hydra overrides> stage=<canonical_stage_or_all> # None = all
params_file=<existing_or_desired> # defaults to params.yaml
dvc_file=<existing_or_desired> #defaults to dvc.yaml
```


### Multi-trial

```bash
python -m deckard optimize <optional hydra overrides> 
stage=<canonical_stage_or_all> 
params_file=<existing_or_desired> # defaults to params.yaml
dvc_file=<existing_or_desired> #defaults to dvc.yaml
```

Requirements:

- Command generation must be deterministic from the same stage plan + params.
- Command strings should avoid ephemeral values unless explicitly requested.

## Determinism and Compatibility Rules

- Stage ordering must follow canonical stage order.
- Generated `dvc.yaml` content must be stable for equivalent manifests.
- DVC generation must not mutate runtime config hashes.
- If runtime schema version is newer than supported, generation should fail with a clear error.
- If optional components are absent (no attack, no detector), omit their DVC stages cleanly.

## Failure Handling

Generation should fail fast on:

- unknown requested stage tokens
- missing required file aliases for selected stages
- unsupported run mode values
- incompatible runtime schema/version metadata

Errors should include:

- failing stage token/name
- missing dep/out/param key
- remediation hint

## Test Plan

Minimum tests:

- stage plan generation from canonical stage set
- deterministic output equality for identical manifests
- stage omission for absent optional components
- multi-attack alias stage expansion
- cache alias inclusion/exclusion toggles
- single vs multirun command generation
- integration test: generated `dvc.yaml` + `dvc repro` dry run command shape validation

## Acceptance Criteria

This contract is satisfied when all are true:

- utility generates valid `dvc.yaml` from experiment runtime metadata
- canonical stage mapping is fully implemented and tested
- single and multirun command templates are emitted deterministically
- cache-reuse alias mode is implemented and tested
- design and usage are documented and linked from developer docs index
- DVCLive integration path is documented with clear hook/layer ownership boundaries
- metrics and plot policies are documented and reflected in generated/maintained `dvc.yaml`

## Related Work

- Hydra stage/trial orchestration should consume the same stage-plan builder and command-generation helpers.
- Expanded integration coverage should build on this contract with end-to-end DVC validation.
- Optimization runtime behavior is specified in [Optimization Runtime Contract](optimization.md).
- Hydra and sweeper lifecycle behavior is specified in [Hydra and Optuna Orchestration Contract](hydra.md).
- Early-stop behavior and prune termination semantics are specified in [Pruning Runtime Contract](pruning.md).

